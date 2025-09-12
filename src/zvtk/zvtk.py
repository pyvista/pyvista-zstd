import json
from typing import Any
import mmap
from tqdm import tqdm
from pathlib import Path
from zstandard import BufferWithSegments, BufferSegment
import zstandard as zstd
import struct
import numpy as np
import pyvista as pv
from pyvista.core.dataset import DataSet
from pyvista.core.pointset import UnstructuredGrid
from vtkmodules.util.numpy_support import numpy_to_vtk as numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkTypeInt32Array, vtkTypeInt64Array
from vtkmodules.vtkCommonDataModel import vtkCellArray

VTK_UNSIGNED_CHAR = 3
POINT_DATA_SUFFIX = "__point_data"
CELL_DATA_SUFFIX = "__cell_data"


def compress(ds, filename: Path | str) -> None:
    arrays = {
        "points": ds.points,
        "offset": ds.offset,
        "cell_connectivity": ds.cell_connectivity,
        "celltypes": ds.celltypes,
    }

    point_data = ds.point_data
    for key, array in point_data.items():
        arrays[key + POINT_DATA_SUFFIX] = array

    cell_data = ds.cell_data
    for key, array in cell_data.items():
        arrays[key + CELL_DATA_SUFFIX] = array

    # dataset metadata
    meta_dict = {
        "type": str(type(ds)),
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
    with open(filename, "wb") as fout, cctx.stream_writer(fout) as compressor:
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
    with open(filename, "ab") as fout:
        for off, dsz in frame_meta:
            fout.write(struct.pack("<QQ", off, dsz))  # 16 bytes per frame
        fout.write(struct.pack("<Q", len(frame_meta)))  # total frames at very end


def _reconstruct_array(segment: BufferSegment) -> np.ndarray:
    """
    Reconstruct a NumPy array from a single decompressed Zstd frame.
    Frame layout:
      [name_len:uint32][name:bytes][ndim:uint32][shape:Q*ndim][dtype:16 bytes][array data]
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


def _get_or_raise(the_dict: dict[str, Any], key: str) -> Any:
    # extract critical arrays
    if key not in the_dict:
        raise RuntimeError(f"zvtk file missing `{key}` array")
    return the_dict[key]


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

    # add point and cell data
    point_data = ugrid.point_data
    cell_data = ugrid.cell_data
    for key, array in segment_dict.items():
        if key.endswith(POINT_DATA_SUFFIX):
            name = key[: -len(POINT_DATA_SUFFIX)]
            point_data.set_array(array, name)
        if key.endswith(CELL_DATA_SUFFIX):
            name = key[: -len(CELL_DATA_SUFFIX)]
            cell_data.set_array(array, name)

    return ugrid


def decompress(filename: Path | str) -> DataSet:
    with open(filename, "rb") as f:
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

    ds_type = metadata["type"]
    if "UnstructuredGrid" in ds_type:
        ds = _segments_to_ugrid(segment_dict)
    else:
        raise RuntimeError(f"Unsupported DataSet type {ds_type}")

    # dataset metadata
    pd = ds.point_data
    if metadata.get("point_data_active_scalars_name"):
        pd.set_active_scalars(metadata["point_data_active_scalars_name"])
    if metadata.get("point_data_active_vectors_name"):
        pd.set_active_vectors(metadata["point_data_active_vectors_name"])
    if metadata.get("point_data_active_normals_name"):
        pd.set_active_normals(metadata["point_data_active_normals_name"])

    cd = ds.cell_data
    if metadata.get("cell_data_active_scalars_name"):
        cd.set_active_scalars(metadata["cell_data_active_scalars_name"])
    if metadata.get("cell_data_active_vectors_name"):
        cd.set_active_vectors(metadata["cell_data_active_vectors_name"])
    if metadata.get("cell_data_active_texture_name"):
        cd.set_active_texture(metadata["cell_data_active_texture_name"])
    if metadata.get("cell_data_active_normals_name"):
        cd.set_active_normals(metadata["cell_data_active_normals_name"])

    return ds
