"""Compress VTK objects using zstandard."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
import struct
from typing import TYPE_CHECKING
from typing import Any
import warnings

import numpy as np
import pyvista as pv
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import UnstructuredGrid
from tqdm import tqdm
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkTypeInt32Array
from vtkmodules.vtkCommonCore import vtkTypeInt64Array
from vtkmodules.vtkCommonDataModel import vtkCellArray
import zstandard as zstd
from zstandard import BufferSegment
from zstandard import BufferWithSegments

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyvista.core.dataset import DataSet


FILE_VERSION = 0
FILE_VERSION_KEY = "FILE_VERSION"
VTK_UNSIGNED_CHAR = 3
POINT_DATA_SUFFIX = "__point_data"
CELL_DATA_SUFFIX = "__cell_data"
FIELD_DATA_SUFFIX = "__field_data"
IMAGE_DATA_SUFFIX = "__image_data"
OFFSET_SUFFIX = "_offset"
CONNECTIVITY_SUFFIX = "_connectivity"

# for all
POINTS = "points"

# for UnstructuredGrid
CELLS = "cells"
POLYHEDRON = "polyhedron"
POLYHEDRON_LOCATION = "polyhedron_locaction"

# for PolyData
POLYS = "polys"
LINES = "lines"
STRIPS = "strips"
VERTS = "verts"


def _add_cell_array(
    arrays: dict[str, np.ndarray],
    name: str,
    cell_array: vtkCellArray,
    *,
    force_int32: bool = False,
) -> None:
    if not cell_array:
        return

    offsets = vtk_to_numpy(cell_array.GetOffsetsArray())
    connectivity = vtk_to_numpy(cell_array.GetConnectivityArray())

    # compress to int32 whenever possible
    if force_int32 and connectivity.size <= np.iinfo(np.int32).max:
        offsets = offsets.astype(np.int32, copy=False)
        connectivity = connectivity.astype(np.int32, copy=False)

    arrays[f"{name}{OFFSET_SUFFIX}"] = offsets
    arrays[f"{name}{CONNECTIVITY_SUFFIX}"] = connectivity


def _extract_cell_array(
    name: str, segments: dict[str, Any], *, require: bool = False
) -> vtkCellArray | None:
    conn_key = f"{name}{CONNECTIVITY_SUFFIX}"
    if conn_key not in segments:
        if require:
            msg = f"Missing {name} array"
            raise RuntimeError(msg)
        return None

    return _numpy_to_vtk_cells(segments[f"{name}{OFFSET_SUFFIX}"], segments[conn_key])


def _prepare_arrays_pointset(ds: PointSet, arrays: dict[str, NDArray[Any]]) -> None:
    arrays[POINTS] = ds.points


def _prepare_arrays_rgrid(ds: RectilinearGrid, arrays: dict[str, NDArray[Any]]) -> None:
    arrays["x"] = ds.x
    arrays["y"] = ds.y
    arrays["z"] = ds.z


def _prepare_arrays_polydata(
    ds: PolyData, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True
) -> None:
    arrays[POINTS] = ds.points
    _add_cell_array(arrays, POLYS, ds.GetPolys(), force_int32=force_int32)
    _add_cell_array(arrays, LINES, ds.GetLines(), force_int32=force_int32)
    _add_cell_array(arrays, STRIPS, ds.GetStrips(), force_int32=force_int32)
    _add_cell_array(arrays, VERTS, ds.GetVerts(), force_int32=force_int32)


def _prepare_arrays_ugrid(
    ds: UnstructuredGrid, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True
) -> None:
    arrays[POINTS] = ds.points
    arrays["celltypes"] = ds.celltypes

    _add_cell_array(arrays, CELLS, ds.GetCells(), force_int32=force_int32)
    _add_cell_array(
        arrays,
        POLYHEDRON,
        ds.GetPolyhedronFaces(),
        force_int32=force_int32,
    )
    _add_cell_array(
        arrays,
        POLYHEDRON_LOCATION,
        ds.GetPolyhedronFaceLocations(),
        force_int32=force_int32,
    )


def _prepare_metadata_imagedata(ds: ImageData, metadata: dict[str, Any]) -> None:
    metadata[f"dimensions{IMAGE_DATA_SUFFIX}"] = ds.dimensions
    metadata[f"origin{IMAGE_DATA_SUFFIX}"] = ds.origin
    metadata[f"spacing{IMAGE_DATA_SUFFIX}"] = ds.spacing
    metadata[f"direction_matrix{IMAGE_DATA_SUFFIX}"] = ds.direction_matrix.tolist()
    metadata[f"offset{IMAGE_DATA_SUFFIX}"] = ds.offset


def compress(  # noqa: C901, PLR0915, PLR0913
    ds: DataSet,
    filename: Path | str,
    *,
    progress_bar: bool = False,
    force_int32: bool = True,
    level: int = 3,
    n_threads: int = 4,
) -> None:
    """
    Compress a PyVista or VTK dataset.

    Supports the following file types.

    * ImageData
    * PolyData
    * RectilinearGrid
    * StructuredGrid
    * UnstructuredGrid

    All file types should end in ``.zvtk``, borrowing both from the legacy
    VTK extension and the ``zst`` file type.

    Parameters
    ----------
    ds : pyvista.DataSet
        Dataset to compress. All PyVista dataset types except for
        :class:`pyvista.MultiBlock` are supported.
    filename : pathlib.Path | str
        Path to the file.
    force_int32 : bool, default: True
        Write offset and connectivity arrays as int32 whenever possible. Only
        applies to :class:`pyvista.PolyData` and
        :class:`pyvista.UnstructuredGrid`.
    progress_bar : bool, default: True
        Show a progress bar while downloading.
    level : int, default: 3
        Compression level. Valid values are all negative integers through
        22. Lower values generally yield faster operations with lower
        compression ratios. Higher values are generally slower but compress
        better.
    n_threads : int, default: 4
        Number of threads to use when compressing. A value of ``-1`` uses all
        available cores.

    """
    metadata: dict[str, int | tuple | str] = {FILE_VERSION_KEY: FILE_VERSION}

    ds = pv.wrap(ds)
    filename = Path(filename)

    if filename.suffix != ".zvtk":
        msg = f"Filename must end in '.zvtk', not '{filename.suffix}'"
        raise ValueError(msg)

    arrays: dict[str, NDArray[Any]] = {}
    if isinstance(ds, PolyData):
        ds_type = "PolyData"
        _prepare_arrays_polydata(ds, arrays, force_int32=force_int32)
    elif isinstance(ds, UnstructuredGrid):
        ds_type = "UnstructuredGrid"
        _prepare_arrays_ugrid(ds, arrays, force_int32=force_int32)
    elif isinstance(ds, ImageData):
        ds_type = "ImageData"
        _prepare_metadata_imagedata(ds, metadata)
    elif isinstance(ds, PointSet):
        ds_type = "PointSet"
        _prepare_arrays_pointset(ds, arrays)
    elif isinstance(ds, RectilinearGrid):
        ds_type = "RectilinearGrid"
        _prepare_arrays_rgrid(ds, arrays)
    else:
        msg = f"Unsupported type {type(ds)}"
        raise TypeError(msg)

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
    metadata["type"] = ds_type
    metadata["COMPRESSION"] = "zstandard"
    metadata["point_data_active_scalars_name"] = point_data.active_scalars_name
    metadata["point_data_active_vectors_name"] = point_data.active_vectors_name
    metadata["point_data_active_texture_coordinates_name"] = (
        point_data.active_texture_coordinates_name
    )
    metadata["point_data_active_normals_name"] = point_data.active_normals_name
    metadata["cell_data_active_scalars_name"] = cell_data.active_scalars_name
    metadata["cell_data_active_vectors_name"] = cell_data.active_vectors_name
    metadata["cell_data_active_texture_coordinates_name"] = (
        cell_data.active_texture_coordinates_name
    )
    metadata["cell_data_active_normals_name"] = cell_data.active_normals_name
    meta_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    arrays["__metadata__"] = np.frombuffer(meta_bytes, dtype=np.uint8)

    cctx = zstd.ZstdCompressor(level=level, threads=n_threads)
    frame_meta = []  # list of tuples: (compressed_end, decompressed_size)
    with filename.open("wb") as fout, cctx.stream_writer(fout) as compressor:
        for name, arr in tqdm(arrays.items(), desc="Compressing", disable=not progress_bar):
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


def _segments_to_ugrid(segments: dict[str, Any]) -> UnstructuredGrid:
    cells = _extract_cell_array(CELLS, segments)

    celltypes = _get_or_raise(segments, "celltypes")
    celltypes_vtk = numpy_to_vtk(celltypes, deep=False, array_type=VTK_UNSIGNED_CHAR)

    ugrid = UnstructuredGrid()
    ugrid.points = _get_or_raise(segments, POINTS)

    poly = _extract_cell_array(POLYHEDRON, segments)
    poly_loc = _extract_cell_array(POLYHEDRON_LOCATION, segments)

    if poly and poly_loc:
        ugrid.SetPolyhedralCells(
            celltypes_vtk,
            cells,
            poly_loc,
            poly,
        )
    else:
        ugrid.SetCells(celltypes_vtk, cells)

    _add_data(ugrid, segments)
    return ugrid


def _numpy_to_vtk_cells(
    offset: NDArray[np.int32] | NDArray[np.int64],
    connectivity: NDArray[np.int32] | NDArray[np.int64],
) -> vtkCellArray:
    # convert to vtk arrays without copying
    dtype = connectivity.dtype
    if dtype == np.int32:
        vtk_dtype = vtkTypeInt32Array().GetDataType()
    elif dtype == np.int64:
        vtk_dtype = vtkTypeInt64Array().GetDataType()
    else:
        msg = f"Invalid faces dtype {dtype}. Expected np.int32 or np.int64"
        raise ValueError(msg)
    connectivity_vtk = numpy_to_vtk(connectivity, deep=False, array_type=vtk_dtype)

    offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)
    carr = vtkCellArray()
    carr.SetData(offset_vtk, connectivity_vtk)
    return carr


def _segments_to_polydata(segments: dict[str, Any]) -> PolyData:
    pdata = PolyData()
    pdata.points = _get_or_raise(segments, POINTS)

    pdata.SetPolys(_extract_cell_array(POLYS, segments))
    pdata.SetLines(_extract_cell_array(LINES, segments))
    pdata.SetStrips(_extract_cell_array(STRIPS, segments))
    pdata.SetVerts(_extract_cell_array(VERTS, segments))

    _add_data(pdata, segments)
    return pdata


def _segments_to_pointset(segments: dict[str, Any]) -> PointSet:
    pset = PointSet(_get_or_raise(segments, POINTS))
    _add_data(pset, segments)
    return pset


def _segments_to_imagedata(segments: dict[str, Any], metadata: dict[str, Any]) -> ImageData:
    image_data = ImageData(
        dimensions=metadata[f"dimensions{IMAGE_DATA_SUFFIX}"],
        origin=metadata[f"origin{IMAGE_DATA_SUFFIX}"],
        spacing=metadata[f"spacing{IMAGE_DATA_SUFFIX}"],
        direction_matrix=metadata[f"direction_matrix{IMAGE_DATA_SUFFIX}"],
        offset=metadata[f"offset{IMAGE_DATA_SUFFIX}"],
    )

    _add_data(image_data, segments)
    return image_data


def _segments_to_rgrid(segments: dict[str, Any]) -> RectilinearGrid:
    rgrid = RectilinearGrid(
        _get_or_raise(segments, "x"),
        _get_or_raise(segments, "y"),
        _get_or_raise(segments, "z"),
    )
    _add_data(rgrid, segments)
    return rgrid


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

    if metadata["FILE_VERSION"] > FILE_VERSION:
        warnings.warn(
            f"The file version {metadata['FILE_VERSION']} of this zvtk file is newer "
            f"than the version supported by this library {FILE_VERSION}. This file "
            " may fail to read. Consider upgrading `zvtk`.",
            stacklevel=0,
        )

    # convert this to match when Python 3.9 goes EOL
    ds_type = metadata["type"]
    if ds_type == "UnstructuredGrid":
        ds = _segments_to_ugrid(segment_dict)
    elif ds_type == "PolyData":
        ds = _segments_to_polydata(segment_dict)
    elif ds_type == "ImageData":
        ds = _segments_to_imagedata(segment_dict, metadata)
    elif ds_type == "PointSet":
        ds = _segments_to_pointset(segment_dict)
    elif ds_type == "RectilinearGrid":
        ds = _segments_to_rgrid(segment_dict)
    else:
        msg = f"zvtk does not support DataSet type `{ds_type}` for compression"
        raise RuntimeError(msg)

    # dataset metadata
    _apply_metadata(ds, metadata)

    return ds
