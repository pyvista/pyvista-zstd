"""Compress VTK objects using zstandard."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
import struct
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pyvista as pv
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import UnstructuredGrid
from tqdm import tqdm
from vtkmodules.util.numpy_support import numpy_to_vtk as numpy_to_vtk  # noqa: PLC0414
from vtkmodules.vtkCommonCore import vtkTypeInt32Array
from vtkmodules.vtkCommonCore import vtkTypeInt64Array
from vtkmodules.vtkCommonDataModel import vtkCellArray
import zstandard as zstd
from zstandard import BufferSegment
from zstandard import BufferWithSegments

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyvista.core.dataset import DataSet

VTK_UNSIGNED_CHAR = 3
POINT_DATA_SUFFIX = "__point_data"
CELL_DATA_SUFFIX = "__cell_data"
FIELD_DATA_SUFFIX = "__field_data"


def compress(ds: DataSet, filename: Path | str) -> None:
    """Compress a PyVista or VTK dataset."""
    ds = pv.wrap(ds)
    filename = Path(filename)

    arrays = {}
    if isinstance(ds, PolyData):
        ds_type = "PolyData"
        arrays["points"] = ds.points
        arrays["offset"] = ds._offset_array  # noqa: SLF001
        arrays["cell_connectivity"] = ds._connectivity_array  # noqa: SLF001
    elif isinstance(ds, UnstructuredGrid):
        ds_type = "UnstructuredGrid"
        arrays["points"] = ds.points
        arrays["offset"] = ds.offset
        arrays["cell_connectivity"] = ds.cell_connectivity
        arrays["celltypes"] = ds.celltypes

    point_data = ds.point_data
    for key, array in point_data.items():
        arrays[key + POINT_DATA_SUFFIX] = array
    cell_data = ds.cell_data
    for key, array in cell_data.items():
        arrays[key + CELL_DATA_SUFFIX] = array
    field_data = ds.field_data
    for key, array in field_data.items():
        arrays[key + FIELD_DATA_SUFFIX] = array

    # dataset metadata
    meta_dict = {
        "type": ds_type,
        "VERSION": 0,
        "point_data_active_scalars_name": point_data.active_scalars_name,
        "point_data_active_vectors_name": point_data.active_vectors_name,
        "point_data_active_texture_coordinates_name": point_data.active_texture_coordinates_name,
        "point_data_active_normals_name": point_data.active_normals_name,
        "cell_data_active_scalars_name": cell_data.active_scalars_name,
        "cell_data_active_vectors_name": cell_data.active_vectors_name,
        "cell_data_active_texture_coordinates_name": cell_data.active_texture_coordinates_name,
        "cell_data_active_normals_name": cell_data.active_normals_name,
    }
    meta_bytes = json.dumps(meta_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    arrays["__metadata__"] = np.frombuffer(meta_bytes, dtype=np.uint8)

    cctx = zstd.ZstdCompressor(level=0, threads=8)
    frame_meta = []  # list of tuples: (compressed_end, decompressed_size)
    with filename.open("wb") as fout, cctx.stream_writer(fout) as compressor:
        for name, arr in tqdm(arrays.items(), desc="Compressing"):
            # Prepare metadata
            meta = struct.pack("<I", len(name)) + name.encode("utf-8")
            meta += struct.pack("<I", arr.ndim)
            for dim in arr.shape:
                meta += struct.pack("<Q", dim)
            meta += arr.dtype.str.encode("utf-8").ljust(16, b" ")

            # Write metadata to the compressed stream
            compressor.write(meta)
            compressor.write(arr.data)

            # Record current frame end offset in compressed stream
            # NOTE: stream_writer does not expose written bytes directly,
            # so we track offsets by flushing using file tell()
            compressor.flush(zstd.FLUSH_FRAME)  # ensures one frame
            frame_end = fout.tell()
            # record compressed end + decompressed size
            frame_meta.append((frame_end, arr.nbytes + len(meta)))

    # Write final metadata
    with filename.open("ab") as fout:
        fout.writelines(
            struct.pack("<QQ", off, dsz) for off, dsz in frame_meta
        )  # 16 bytes per frame
        fout.write(struct.pack("<Q", len(frame_meta)))  # total frames at very end


def _reconstruct_array(segment: BufferSegment) -> np.ndarray:
    """
    Reconstruct a NumPy array from a single decompressed Zstd frame.

    Frame layout:
    ``[name_len:uint32][name:bytes][ndim:uint32][shape:Q*ndim][dtype:16 bytes][array data]``.

    """
    buf = memoryview(segment)  # get a bytes-like view

    offset = 0
    name_len = struct.unpack_from("<I", buf, offset)[0]
    offset += 4
    name = buf[offset : offset + name_len].tobytes().decode("utf-8")
    offset += name_len

    ndim = struct.unpack_from("<I", buf, offset)[0]
    offset += 4

    shape = tuple(struct.unpack_from(f"<{ndim}Q", buf, offset))
    offset += 8 * ndim

    dtype_str = buf[offset : offset + 16].tobytes().strip().decode("utf-8")
    offset += 16

    data = np.frombuffer(buf[offset:], dtype=np.dtype(dtype_str)).reshape(shape)
    return name, data


def _get_or_raise(the_dict: dict[str, Any], key: str) -> NDArray:
    """Extract a key and raise a helpful error if missing it."""
    # extract critical arrays
    if key not in the_dict:
        msg = f"zvtk file missing `{key}` array"
        raise RuntimeError(msg)
    return the_dict[key]


def _add_data(ds: DataSet, segment_dict: dict[str, Any]) -> None:
    # add point and cell data
    point_data = ds.point_data
    cell_data = ds.cell_data
    field_data = ds.field_data
    for key, array in segment_dict.items():
        if key.endswith(POINT_DATA_SUFFIX):
            name = key[: -len(POINT_DATA_SUFFIX)]
            point_data.set_array(array, name)
        if key.endswith(CELL_DATA_SUFFIX):
            name = key[: -len(CELL_DATA_SUFFIX)]
            cell_data.set_array(array, name)
        if key.endswith(FIELD_DATA_SUFFIX):
            name = key[: -len(FIELD_DATA_SUFFIX)]
            field_data.set_array(array, name)


def _segments_to_ugrid(segment_dict: dict[str, Any]) -> UnstructuredGrid:
    # convert to vtk arrays without copying
    offset = _get_or_raise(segment_dict, "offset")
    dtype = offset.dtype
    if dtype == np.int32:
        vtk_dtype = vtkTypeInt32Array().GetDataType()
    elif dtype == np.int64:
        vtk_dtype = vtkTypeInt64Array().GetDataType()
    offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)

    connectivity = _get_or_raise(segment_dict, "cell_connectivity")
    connectivity_vtk = numpy_to_vtk(connectivity, deep=False, array_type=vtk_dtype)

    cell_array = vtkCellArray()
    cell_array.SetData(offset_vtk, connectivity_vtk)

    celltypes = _get_or_raise(segment_dict, "celltypes")
    celltypes_vtk = numpy_to_vtk(celltypes, deep=False, array_type=VTK_UNSIGNED_CHAR)

    ugrid = UnstructuredGrid()
    ugrid.SetCells(celltypes_vtk, cell_array)
    ugrid.points = _get_or_raise(segment_dict, "points")

    _add_data(ugrid, segment_dict)
    return ugrid


def _segments_to_polydata(segment_dict: dict[str, Any]) -> PolyData:
    pdata = PolyData()
    pdata.points = _get_or_raise(segment_dict, "points")

    # convert to vtk arrays without copying
    connectivity = _get_or_raise(segment_dict, "cell_connectivity")
    dtype = connectivity.dtype
    if dtype == np.int32:
        vtk_dtype = vtkTypeInt32Array().GetDataType()
    elif dtype == np.int64:
        vtk_dtype = vtkTypeInt64Array().GetDataType()
    else:
        msg = f"Invalid faces dtype {dtype}. Expected np.int32 or np.int64"
        raise ValueError(msg)
    connectivity_vtk = numpy_to_vtk(connectivity, deep=False, array_type=vtk_dtype)

    offset = _get_or_raise(segment_dict, "offset")
    offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)

    carr = vtkCellArray()
    carr.SetData(offset_vtk, connectivity_vtk)
    pdata.SetPolys(carr)

    _add_data(pdata, segment_dict)
    return pdata


def _apply_metadata(ds: DataSet, metadata: dict[str, Any]) -> None:
    """Apply metadata to a dataset."""
    pd = ds.point_data
    if metadata.get("point_data_active_scalars_name"):
        pd.active_scalars_name = metadata["point_data_active_scalars_name"]
    if metadata.get("point_data_active_vectors_name"):
        pd.active_vectors_name = metadata["point_data_active_vectors_name"]
    if metadata.get("point_data_active_texture_coordinates_name"):
        pd.active_texture_coordinates_name = metadata["point_data_active_texture_coordinates_name"]
    if metadata.get("point_data_active_normals_name"):
        pd.active_normals_name = metadata["point_data_active_normals_name"]

    cd = ds.cell_data
    if metadata.get("cell_data_active_scalars_name"):
        cd.active_scalars_name = metadata["cell_data_active_scalars_name"]
    if metadata.get("cell_data_active_vectors_name"):
        cd.active_vectors_name = metadata["cell_data_active_vectors_name"]
    if metadata.get("cell_data_active_texture_coordinates_name"):
        cd.active_texture_coordinates_name = metadata["cell_data_active_texture_coordinates_name"]
    if metadata.get("cell_data_active_normals_name"):
        cd.active_normals_name = metadata["cell_data_active_normals_name"]


def decompress(filename: Path | str) -> DataSet:
    """Decompress a ``zvtk`` file."""
    filename = Path(filename)
    with filename.open("rb") as f:
        f.seek(-8, 2)
        num_frames = struct.unpack("<Q", f.read(8))[0]

        # Each frame has 16 bytes: (compressed_end_offset, decompressed_size)
        f.seek(-(8 + num_frames * 16), 2)
        meta_data = f.read(num_frames * 16)

        # unpack as list of tuples
        frame_meta = [
            struct.unpack("<QQ", meta_data[i * 16 : (i + 1) * 16]) for i in range(num_frames)
        ]

        # compute start/end of each frame for compressed segments
        frame_starts = [0] + [end for end, _ in frame_meta[:-1]]
        frame_ends = [end for end, _ in frame_meta]

        # decompressed sizes
        sizes = [dsz for _, dsz in frame_meta]
        decompressed_sizes = struct.pack(f"={len(sizes)}Q", *sizes)

        # mmap the file for BufferWithSegments
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # construct BufferWithSegments
    segments_bytes = b"".join(
        struct.pack("=QQ", start, end - start) for start, end in zip(frame_starts, frame_ends)
    )
    frames = BufferWithSegments(mm, segments_bytes)

    # decompress with multi-threaded buffer API
    dctx = zstd.ZstdDecompressor()
    segments = dctx.multi_decompress_to_buffer(
        frames, decompressed_sizes=decompressed_sizes, threads=8
    )

    segment_dict = dict(_reconstruct_array(s) for s in segments)

    # metadata array is JSON
    metadata_raw = segment_dict.pop("__metadata__")
    metadata = json.loads(metadata_raw.tobytes().decode("utf-8"))

    # convert this to match when Python 3.9 goes EOL
    ds_type = metadata["type"]
    if ds_type == "UnstructuredGrid":
        ds = _segments_to_ugrid(segment_dict)
    elif ds_type == "PolyData":
        ds = _segments_to_polydata(segment_dict)
    else:
        msg = f"Unsupported DataSet type `{ds_type}`"
        raise RuntimeError(msg)

    # dataset metadata
    _apply_metadata(ds, metadata)

    return ds
