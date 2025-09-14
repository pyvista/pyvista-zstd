"""Compress VTK objects using zstandard."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
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
DS_TYPE_KEY = "ds_type"
VTK_UNSIGNED_CHAR = 3
POINT_DATA_SUFFIX = "__point_data"
CELL_DATA_SUFFIX = "__cell_data"
FIELD_DATA_SUFFIX = "__field_data"
IMAGE_DATA_SUFFIX = "__image_data"
OFFSET_SUFFIX = "_offset"
CONNECTIVITY_SUFFIX = "_connectivity"
METADATA_KEY_COMPRESSION = "COMPRESSION"
METADATA_KEY_COMPRESSION_LVL = "COMPRESSION_LEVEL"
CELL_TYPES_KEY = "celltypes"

# for all
POINTS_KEY = "points"

# for UnstructuredGrid
CELLS = "cells"
POLYHEDRON = "polyhedron"
POLYHEDRON_LOCATION = "polyhedron_locaction"

# for PolyData
POLYS = "polys"
LINES = "lines"
STRIPS = "strips"
VERTS = "verts"


@dataclass
class ArrayInfo:
    """Array metadata."""

    shape: tuple[int, ...]
    dtype: str


@dataclass
class Metadata:
    """DataSet metadata."""

    file_version: int
    ds_type: str
    compression: str
    compression_level: int
    n_points: int
    points_dtype: str | None
    n_cells: int
    celltypes_dtype: str | None
    point_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)
    cell_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)
    field_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)
    point_data_active_scalars_name: str | None = None
    point_data_active_vectors_name: str | None = None
    point_data_active_texture_coordinates_name: str | None = None
    point_data_active_normals_name: str | None = None
    cell_data_active_scalars_name: str | None = None
    cell_data_active_vectors_name: str | None = None
    cell_data_active_texture_coordinates_name: str | None = None
    cell_data_active_normals_name: str | None = None

    # Optional ImageData metadata
    dimensions: tuple[int, int, int] | None = None
    origin: tuple[float, float, float] | None = None
    spacing: tuple[float, float, float] | None = None
    direction_matrix: list[list[float]] | None = None
    offset: int | None = None

    # required, but not initialized from dataset
    frame_names: list[str] | None = None

    @classmethod
    def from_dataset(cls, ds: pv.DataSet, level: int) -> Metadata:
        """Create metadata from a dataset."""

        def _summarize(arrays: pv.DataSetAttributes) -> dict[str, ArrayInfo]:
            return {k: ArrayInfo(shape=a.shape, dtype=str(a.dtype)) for k, a in arrays.items()}

        kwargs: dict[str, Any] = {
            "file_version": FILE_VERSION,
            "ds_type": type(ds).__name__,
            "compression": "zstandard",
            "compression_level": level,
            "n_points": ds.n_points,
            "points_dtype": str(ds.points.dtype) if ds.n_points else None,
            "n_cells": ds.n_cells,
            "celltypes_dtype": str(ds.celltypes.dtype) if hasattr(ds, "celltypes") else None,
            "point_data_keys": _summarize(ds.point_data),
            "cell_data_keys": _summarize(ds.cell_data),
            "field_data_keys": _summarize(ds.field_data),
            "point_data_active_scalars_name": ds.point_data.active_scalars_name,
            "point_data_active_vectors_name": ds.point_data.active_vectors_name,
            "point_data_active_texture_coordinates_name": ds.point_data.active_texture_coordinates_name,  # noqa: E501
            "point_data_active_normals_name": ds.point_data.active_normals_name,
            "cell_data_active_scalars_name": ds.cell_data.active_scalars_name,
            "cell_data_active_vectors_name": ds.cell_data.active_vectors_name,
            "cell_data_active_texture_coordinates_name": ds.cell_data.active_texture_coordinates_name,  # noqa: E501
            "cell_data_active_normals_name": ds.cell_data.active_normals_name,
        }

        if isinstance(ds, pv.ImageData):
            kwargs.update(
                dimensions=tuple(ds.dimensions),
                origin=tuple(ds.origin),
                spacing=tuple(ds.spacing),
                direction_matrix=ds.direction_matrix.tolist(),
                offset=ds.offset,
            )

        return cls(**kwargs)

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> Metadata:
        """Create from JSON."""
        raw = json.loads(s)

        def decode_mapping(m: dict[str, Any]) -> dict[str, ArrayInfo]:
            return {k: ArrayInfo(**v) for k, v in m.items()}

        raw["point_data_keys"] = decode_mapping(raw.get("point_data_keys", {}))
        raw["cell_data_keys"] = decode_mapping(raw.get("cell_data_keys", {}))
        raw["field_data_keys"] = decode_mapping(raw.get("field_data_keys", {}))
        return cls(**raw)


def _set_n_threads(n_threads: int | None, n_bytes: int, max_manual_threads: int = 8) -> int:
    # Maximum number of set threads before relying on zstandard to
    # automatically set them

    if n_threads is None:
        size_mb = n_bytes / 1024**2
        n_threads = int(size_mb // 2)  # rough guess
        n_threads = -1 if n_threads > max_manual_threads else n_threads

    return n_threads


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


def _extract_cell_array(name: str, segments: dict[str, Any]) -> vtkCellArray | None:
    conn_key = f"{name}{CONNECTIVITY_SUFFIX}"
    if conn_key not in segments:
        return None

    return _numpy_to_vtk_cells(segments[f"{name}{OFFSET_SUFFIX}"], segments[conn_key])


def _prepare_arrays_pointset(ds: PointSet, arrays: dict[str, NDArray[Any]]) -> None:
    arrays[POINTS_KEY] = ds.points


def _prepare_arrays_rgrid(ds: RectilinearGrid, arrays: dict[str, NDArray[Any]]) -> None:
    if ds.n_points:
        arrays["x"] = ds.x
        arrays["y"] = ds.y
        arrays["z"] = ds.z


def _prepare_arrays_polydata(
    ds: PolyData, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True
) -> None:
    arrays[POINTS_KEY] = ds.points
    _add_cell_array(arrays, POLYS, ds.GetPolys(), force_int32=force_int32)
    _add_cell_array(arrays, LINES, ds.GetLines(), force_int32=force_int32)
    _add_cell_array(arrays, STRIPS, ds.GetStrips(), force_int32=force_int32)
    _add_cell_array(arrays, VERTS, ds.GetVerts(), force_int32=force_int32)


def _prepare_arrays_ugrid(
    ds: UnstructuredGrid, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True
) -> None:
    arrays[POINTS_KEY] = ds.points
    arrays[CELL_TYPES_KEY] = ds.celltypes

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


def write(  # noqa: C901, PLR0913
    ds: DataSet,
    filename: Path | str,
    *,
    progress_bar: bool = False,
    force_int32: bool = True,
    level: int = 3,
    n_threads: int | None = None,
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
    n_threads : int, optional
        Number of threads to use when compressing. A value of ``-1`` uses all
        available cores and ``0`` disables multi-threading.

    """
    ds = pv.wrap(ds)
    filename = Path(filename)

    if filename.suffix != ".zvtk":
        msg = f"Filename must end in '.zvtk', not '{filename.suffix}'"
        raise ValueError(msg)

    arrays: dict[str, NDArray[Any]] = {}
    if isinstance(ds, PolyData):
        _prepare_arrays_polydata(ds, arrays, force_int32=force_int32)
    elif isinstance(ds, UnstructuredGrid):
        _prepare_arrays_ugrid(ds, arrays, force_int32=force_int32)
    elif isinstance(ds, ImageData):
        pass
    elif isinstance(ds, PointSet):
        _prepare_arrays_pointset(ds, arrays)
    elif isinstance(ds, RectilinearGrid):
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

    n_bytes = sum([arr.nbytes for arr in arrays.values()])
    n_threads = _set_n_threads(n_threads, n_bytes)

    # dataset metadata
    metadata = Metadata.from_dataset(ds, level)
    metadata.frame_names = list(arrays.keys())  # order matters
    meta_bytes = metadata.to_json().encode("utf-8")

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

    celltypes = segments[CELL_TYPES_KEY]
    celltypes_vtk = numpy_to_vtk(celltypes, deep=False, array_type=VTK_UNSIGNED_CHAR)

    ugrid = UnstructuredGrid()
    ugrid.points = segments[POINTS_KEY]

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
    else:  # pragma: no cover
        msg = f"Invalid faces dtype {dtype}. Expected np.int32 or np.int64"
        raise ValueError(msg)
    connectivity_vtk = numpy_to_vtk(connectivity, deep=False, array_type=vtk_dtype)

    offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)
    carr = vtkCellArray()
    carr.SetData(offset_vtk, connectivity_vtk)
    return carr


def _segments_to_polydata(segments: dict[str, Any]) -> PolyData:
    pdata = PolyData()
    pdata.points = segments[POINTS_KEY]

    pdata.SetPolys(_extract_cell_array(POLYS, segments))
    pdata.SetLines(_extract_cell_array(LINES, segments))
    pdata.SetStrips(_extract_cell_array(STRIPS, segments))
    pdata.SetVerts(_extract_cell_array(VERTS, segments))

    _add_data(pdata, segments)
    return pdata


def _segments_to_pointset(segments: dict[str, Any]) -> PointSet:
    pset = PointSet(segments[POINTS_KEY])
    _add_data(pset, segments)
    return pset


def _segments_to_imagedata(segments: dict[str, Any], metadata: Metadata) -> ImageData:
    image_data = ImageData(
        dimensions=metadata.dimensions,
        origin=metadata.origin,
        spacing=metadata.spacing,
        direction_matrix=metadata.direction_matrix,
        offset=metadata.offset,
    )

    _add_data(image_data, segments)
    return image_data


def _segments_to_rgrid(segments: dict[str, Any]) -> RectilinearGrid:
    if "x" in segments:
        rgrid = RectilinearGrid(
            segments.get("x"),
            segments.get("y"),
            segments.get("z"),
        )
    else:
        rgrid = RectilinearGrid()
    _add_data(rgrid, segments)
    return rgrid


def _apply_metadata(ds: DataSet, metadata: Metadata) -> None:
    """Apply metadata to a dataset."""
    pd = ds.point_data
    if metadata.point_data_active_scalars_name in pd:
        pd.active_scalars_name = metadata.point_data_active_scalars_name
    if metadata.point_data_active_vectors_name in pd:
        pd.active_vectors_name = metadata.point_data_active_vectors_name
    if metadata.point_data_active_texture_coordinates_name in pd:
        pd.active_texture_coordinates_name = metadata.point_data_active_texture_coordinates_name
    if metadata.point_data_active_normals_name in pd:
        pd.active_normals_name = metadata.point_data_active_normals_name

    cd = ds.cell_data
    if metadata.cell_data_active_scalars_name in cd:
        cd.active_scalars_name = metadata.cell_data_active_scalars_name
    if metadata.cell_data_active_vectors_name in cd:
        cd.active_vectors_name = metadata.cell_data_active_vectors_name
    if metadata.cell_data_active_texture_coordinates_name in cd:
        cd.active_texture_coordinates_name = metadata.cell_data_active_texture_coordinates_name
    if metadata.cell_data_active_normals_name in cd:
        cd.active_normals_name = metadata.cell_data_active_normals_name


def read(filename: Path | str, n_threads: int | None = None) -> DataSet:
    """
    Decompress a ``zvtk`` file.

    This is a convenience function that uses :class:`Reader`. Use that class to
    finely tune reading in a file.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to the file.
    n_threads : None | int, optional
        Number of threads to use. If omitted, the best number of threads to
        decompress the file will be used.

    """
    return Reader(filename).read(n_threads=n_threads)


class Reader:
    """
    Class to control zvtk file decompression.

    Use this class in lieu of :func:`zvtk.read` to fine-tune reading in
    compressed files. With this you can:
    - Inspect the dataset before reading it.
    - Control which arrays to read in.
    - For files containing a :class:`pyvista.MultiBlock`, select which blocks
      to read in.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to the file. Must end in ``.zvtk``.

    """

    def __init__(self, filename: Path | str) -> None:
        """Initialize the decompressor."""
        self._filename = Path(filename)

        if self._filename.suffix != ".zvtk":
            msg = f"Filename must end in '.zvtk', not '{self._filename.suffix}'"
            raise ValueError(msg)

        with self._filename.open("rb") as f:
            f.seek(-8, 2)
            num_frames = struct.unpack("<Q", f.read(8))[0]

            max_frames = 1_000_000
            if num_frames > max_frames:
                msg = "Bad number of frames. File may be corrupted."
                raise RuntimeError(msg)

            f.seek(-(8 + num_frames * 16), 2)
            meta_data = f.read(num_frames * 16)
            frame_meta = [
                struct.unpack("<QQ", meta_data[i * 16 : (i + 1) * 16]) for i in range(num_frames)
            ]
            frame_starts = [0] + [end for end, _ in frame_meta[:-1]]
            frame_ends = [end for end, _ in frame_meta]
            sizes = [dsz for _, dsz in frame_meta]
            self._decompressed_sizes = struct.pack(f"={len(sizes)}Q", *sizes)
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # prepare the metadata frame and decompress it
        segments_bytes = b"".join(
            struct.pack("=QQ", start, end - start) for start, end in zip(frame_starts, frame_ends)
        )

        if not segments_bytes:
            msg = "Empty segments. File may be corrupted."
            raise RuntimeError(msg)

        self._frames = BufferWithSegments(self._mm, segments_bytes)
        self._metadata = self._load_metadata()

    @property
    def decompressed_sizes(self) -> NDArray[np.uint64]:
        """
        Return decompressed frame sizes.

        This an array containing 64-bit unsigned integers containing the
        decompressed sizes in bytes of each frame.
        """
        return np.frombuffer(self._decompressed_sizes, dtype=np.uint64)

    @property
    def nbytes(self) -> int:
        """Return the size of the decompressed dataset."""
        return int(self.decompressed_sizes.sum())

    def _load_metadata(self) -> Metadata:
        """Load the metadata from the zvtk file without full decompression."""
        dctx = zstd.ZstdDecompressor()

        # read in only the last segment
        segments = dctx.multi_decompress_to_buffer(
            [self._frames[-1]],
            decompressed_sizes=self._decompressed_sizes[-8:],
            threads=0,  # tiny
        )
        name, arr = _reconstruct_array(segments[0])
        if name != "__metadata__":  # pragma: no cover
            msg = "Metadata not found in zvtk file"
            raise RuntimeError(msg)

        metadata = Metadata.from_json(arr.tobytes().decode("utf-8"))

        if metadata.file_version > FILE_VERSION:
            warnings.warn(
                f"The file version {metadata.file_version} of this zvtk file is "
                f"newer than the version supported by this library {FILE_VERSION}. "
                "This file may fail to read. Consider upgrading `zvtk`.",
                stacklevel=0,
            )

        return metadata

    def read(self, n_threads: int | None = None) -> DataSet:
        """Read in the dataset from the zvtk file."""
        n_threads = _set_n_threads(n_threads, self.nbytes)

        # map frame indices to names using metadata
        frame_names = self._metadata.frame_names
        if frame_names is None:  # pragma: no cover
            msg = "Frame names not found in metadata."
            raise RuntimeError(msg)

        excluded = set()
        for name in self.available_point_arrays - self.selected_point_arrays:
            excluded.add(name + POINT_DATA_SUFFIX)
        for name in self.available_cell_arrays - self.selected_cell_arrays:
            excluded.add(name + CELL_DATA_SUFFIX)
        for name in self.available_field_arrays - self.selected_field_arrays:
            excluded.add(name + FIELD_DATA_SUFFIX)

        selected_frames = []
        sizes = []
        selected_names = set(frame_names) - excluded
        for ii, name in enumerate(frame_names):
            if name in selected_names:
                selected_frames.append(self._frames[ii])
                # 8 bytes per frame
                sizes.append(self._decompressed_sizes[ii * 8 : (ii + 1) * 8])

        # Decompress with multi-threaded buffer API
        if selected_frames:
            dctx = zstd.ZstdDecompressor()
            segments_raw = dctx.multi_decompress_to_buffer(
                selected_frames,
                decompressed_sizes=b"".join(sizes),
                threads=n_threads,
            )
            segments = dict(_reconstruct_array(segment) for segment in segments_raw)
        else:
            segments = {}

        return self._segments_to_ds(segments)

    def _segments_to_ds(self, segments: dict[str, Any]) -> DataSet:
        # convert this to match when Python 3.9 goes EOL
        ds_type = self._metadata.ds_type
        if ds_type == "UnstructuredGrid":
            ds = _segments_to_ugrid(segments)
        elif ds_type == "PolyData":
            ds = _segments_to_polydata(segments)
        elif ds_type == "ImageData":
            ds = _segments_to_imagedata(segments, self._metadata)
        elif ds_type == "PointSet":
            ds = _segments_to_pointset(segments)
        elif ds_type == "RectilinearGrid":
            ds = _segments_to_rgrid(segments)
        else:
            msg = f"zvtk does not support DataSet type `{ds_type}` for compression"
            raise RuntimeError(msg)
        _apply_metadata(ds, self._metadata)

    @property
    def available_point_arrays(self) -> set[str]:
        """Return a set of all point array names available in the dataset."""
        return set(self._metadata.point_data_keys)

    @property
    def available_cell_arrays(self) -> set[str]:
        """Return a set of all cell array names available in the dataset."""
        return set(self._metadata.cell_data_keys)

    @property
    def available_field_arrays(self) -> set[str]:
        """Return a set of all field array names available in the dataset."""
        return set(self._metadata.field_data_keys)

    @property
    def selected_point_arrays(self) -> set[str]:
        """
        Return the set of currently selected point arrays to read.

        Defaults to all available arrays.
        """
        if not hasattr(self, "_selected_point_arrays"):
            return self.available_point_arrays.copy()
        return self._selected_point_arrays

    @selected_point_arrays.setter
    def selected_point_arrays(self, value: set[str]) -> None:
        """
        Set the point arrays to read from the file.

        Parameters
        ----------
        value : set[str]
            A set of point array names to read. All names must exist in
            `available_point_arrays`.

        Raises
        ------
        ValueError
            If any name in `value` is not available in the file.

        """
        invalid = value - self.available_point_arrays
        if invalid:
            msg = f"The following point array(s) are not available: {invalid}"
            raise ValueError(msg)
        self._selected_point_arrays = value.copy()

    @property
    def selected_cell_arrays(self) -> set[str]:
        """
        Return the set of currently selected cell arrays to read.

        Defaults to all available arrays.
        """
        if not hasattr(self, "_selected_cell_arrays"):
            return self.available_cell_arrays.copy()
        return self._selected_cell_arrays

    @selected_cell_arrays.setter
    def selected_cell_arrays(self, value: set[str]) -> None:
        """
        Set the cell arrays to read from the file.

        Parameters
        ----------
        value : set[str]
            A set of cell array names to read. All names must exist in `available_cell_arrays`.

        Raises
        ------
        ValueError
            If any name in `value` is not available in the file.

        """
        invalid = value - self.available_cell_arrays
        if invalid:
            msg = f"The following cell array(s) are not available: {invalid}"
            raise ValueError(msg)
        self._selected_cell_arrays = value.copy()

    @property
    def selected_field_arrays(self) -> set[str]:
        """
        Return the set of currently selected field arrays to read.

        Defaults to all available arrays.
        """
        if not hasattr(self, "_selected_field_arrays"):
            return self.available_field_arrays.copy()
        return self._selected_field_arrays

    @selected_field_arrays.setter
    def selected_field_arrays(self, value: set[str]) -> None:
        """
        Set the field arrays to read from the file.

        Parameters
        ----------
        value : set[str]
            A set of field array names to read. All names must exist in `available_field_arrays`.

        Raises
        ------
        ValueError
            If any name in `value` is not available in the file.

        """
        invalid = value - self.available_field_arrays
        if invalid:
            msg = f"The following field array(s) are not available: {invalid}"
            raise ValueError(msg)
        self._selected_field_arrays = value.copy()

    def __repr__(self) -> str:
        """Return a representation of the dataset's metadata."""
        md = self._metadata
        header = [
            f"zvtk.Decompressor ({hex(id(self))})",
            f"  File:               {self._filename}",
            f"  Dataset Type:       {md.ds_type}",
            f"  File Version:       {md.file_version}",
            f"  Compression:        {md.compression}",
            f"  Compression Level:  {md.compression_level}",
            f"  N Points:           {md.n_points} ({md.points_dtype})",
            f"  N Cells:            {md.n_cells}",
        ]

        def _format_dsa(name: str, arrays: dict[str, ArrayInfo]) -> list[str]:
            if not arrays:
                return []
            lines = []
            if arrays:
                lines.append(f"  {name} arrays:")
                for k, info in arrays.items():
                    shape = tuple(info.shape)
                    lines.append(f"      {k:<24} {info.dtype:<10} {shape}")
            return lines

        # Point data
        header.extend(_format_dsa("Point", md.point_data_keys))

        # Cell data
        header.extend(_format_dsa("Cell", md.cell_data_keys))

        # Field data
        if md.field_data_keys:
            lines = ["  Field arrays"]
            for k, info in md.field_data_keys.items():
                shape = tuple(info.shape)
                lines.append(f"      {k:<24} {info.dtype:<10} {shape}")
            header.extend(lines)

        return "\n".join(header)
