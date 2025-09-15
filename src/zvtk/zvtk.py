"""
Compress VTK objects using zstandard.

We're writing everything out using `zstandard frames
<https://python-zstandard.readthedocs.io/en/latest/concepts.html>`_.

"""

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
from pyvista.core.composite import MultiBlock
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
DS_METADATA_KEY = "__ds_metadata"
MULTIBLOCK_METADATA_KEY = "__multiblock__ds_metadata"
FILE_METADATA_KEY = "__zvtk_metadata"

RGRID_X_SUFFIX = "_x_rgrid"
RGRID_Y_SUFFIX = "_y_rgrid"
RGRID_Z_SUFFIX = "_z_rgrid"


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
class ZvtkFileMetadata:
    """Zvtk file metadata."""

    frame_names: list[str]
    compression_level: int
    compression: str = "zstandard"  # only option at the moment
    file_version: int = FILE_VERSION

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> MultiBlockMetadata:
        """Create from JSON."""
        return cls(**json.loads(s))


@dataclass
class MultiBlockMetadata:
    """MultiBlock metadata."""

    uid: str
    children: dict[str]  # key: value
    ds_type = "MultiBlock"

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> MultiBlockMetadata:
        """Create from JSON."""
        return cls(**json.loads(s))

    @classmethod
    def from_array(cls, arr: str) -> MultiBlockMetadata:
        """Create from a numpy uint8 array."""
        raw_json = arr.tobytes().decode("utf-8")  # copy, but it's tiny
        return MultiBlockMetadata.from_json(raw_json)

    @classmethod
    def from_multiblock(cls, mblock: MultiBlock) -> MultiBlockMetadata:
        """Create from a multiblock."""
        return cls(
            uid=_make_ds_id(mblock),
            children=[_make_ds_id(ds) for ds in mblock],
        )


@dataclass
class DataSetMetadata:
    """DataSet metadata."""

    ds_type: str
    uid: str
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

    @classmethod
    def from_dataset(cls, ds: pv.DataSet) -> DataSetMetadata:
        """Create metadata from a dataset."""

        def _summarize(arrays: pv.DataSetAttributes) -> dict[str, ArrayInfo]:
            return {k: ArrayInfo(shape=a.shape, dtype=str(a.dtype)) for k, a in arrays.items()}

        kwargs: dict[str, Any] = {
            "ds_type": type(ds).__name__,
            "uid": _make_ds_id(ds),
            "n_points": ds.n_points,
            "points_dtype": str(ds.points.dtype) if ds.n_points else None,
            "n_cells": ds.n_cells,
            "celltypes_dtype": str(ds.celltypes.dtype) if hasattr(ds, "celltypes") else None,
            "point_data_keys": _summarize(ds.point_data),
            "cell_data_keys": _summarize(ds.cell_data),
            "field_data_keys": _summarize(ds.field_data),
            "point_data_active_scalars_name": ds.point_data.active_scalars_name,
            "point_data_active_vectors_name": ds.point_data.active_vectors_name,
            "point_data_active_texture_coordinates_name": ds.point_data.active_texture_coordinates_name,
            "point_data_active_normals_name": ds.point_data.active_normals_name,
            "cell_data_active_scalars_name": ds.cell_data.active_scalars_name,
            "cell_data_active_vectors_name": ds.cell_data.active_vectors_name,
            "cell_data_active_texture_coordinates_name": ds.cell_data.active_texture_coordinates_name,
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
    def from_array(cls, arr: str) -> DataSetMetadata:
        """Create from a numpy uint8 array."""
        raw_json = arr.tobytes().decode("utf-8")  # copy, but it's tiny
        return DataSetMetadata.from_json(raw_json)

    @classmethod
    def from_json(cls, s: str) -> DataSetMetadata:
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
    ds_id: str,
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

    arrays[f"{ds_id}{name}{OFFSET_SUFFIX}"] = offsets
    arrays[f"{ds_id}{name}{CONNECTIVITY_SUFFIX}"] = connectivity


def _extract_cell_array(ds_id: str, name: str, segments: dict[str, Any]) -> vtkCellArray | None:
    conn_key = f"{ds_id}{name}{CONNECTIVITY_SUFFIX}"
    if conn_key not in segments:
        return None

    offset_key = f"{ds_id}{name}{OFFSET_SUFFIX}"
    return _numpy_to_vtk_cells(segments[offset_key], segments[conn_key])


def _add_arrays_pointset(ds: PointSet, arrays: dict[str, NDArray[Any]]) -> None:
    arrays[f"{_make_ds_id(ds)}{POINTS_KEY}"] = ds.points


def _add_arrays_rgrid(ds: RectilinearGrid, arrays: dict[str, NDArray[Any]]) -> None:
    if ds.n_points:
        ds_id = _make_ds_id(ds)
        arrays[f"{ds_id}{RGRID_X_SUFFIX}"] = ds.x
        arrays[f"{ds_id}{RGRID_Y_SUFFIX}"] = ds.y
        arrays[f"{ds_id}{RGRID_Z_SUFFIX}"] = ds.z


def _add_arrays_polydata(ds: PolyData, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True) -> None:
    ds_id = _make_ds_id(ds)
    arrays[f"{ds_id}{POINTS_KEY}"] = ds.points
    _add_cell_array(ds_id, arrays, POLYS, ds.GetPolys(), force_int32=force_int32)
    _add_cell_array(ds_id, arrays, LINES, ds.GetLines(), force_int32=force_int32)
    _add_cell_array(ds_id, arrays, STRIPS, ds.GetStrips(), force_int32=force_int32)
    _add_cell_array(ds_id, arrays, VERTS, ds.GetVerts(), force_int32=force_int32)


def _add_arrays_ugrid(ds: UnstructuredGrid, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True) -> None:
    ds_id = _make_ds_id(ds)
    arrays[f"{ds_id}{POINTS_KEY}"] = ds.points
    arrays[f"{ds_id}{CELL_TYPES_KEY}"] = ds.celltypes

    _add_cell_array(ds_id, arrays, CELLS, ds.GetCells(), force_int32=force_int32)
    _add_cell_array(
        ds_id,
        arrays,
        POLYHEDRON,
        ds.GetPolyhedronFaces(),
        force_int32=force_int32,
    )
    _add_cell_array(
        ds_id,
        arrays,
        POLYHEDRON_LOCATION,
        ds.GetPolyhedronFaceLocations(),
        force_int32=force_int32,
    )


def write(  # noqa: PLR0913
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

    Supports the following classes.

    * :class:`pyvista.ImageData`
    * :class:`pyvista.PolyData`
    * :class:`pyvista.RectilinearGrid`
    * :class:`pyvista.StructuredGrid`
    * :class:`pyvista.UnstructuredGrid`
    * :class:`pyvista.MultiBlock`

    All file types should end in ``.zvtk``, borrowing both from the legacy
    VTK extension ``.vtk`` and the ``.zst`` file types.

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
    Writer(ds, filename).write(
        progress_bar=progress_bar,
        force_int32=force_int32,
        level=level,
        n_threads=n_threads,
    )


def _make_ds_id(ds: DataSet) -> str:
    """Make a unique dataset ID using the memory address."""
    # padded for 32-bit
    return f"{id(ds):016x}"


class Writer:
    """Class to write a zvtk file."""

    def __init__(self, ds: DataSet, filename: Path | str) -> None:
        """Initialize the writer."""
        self._filename = Path(filename)

        if self._filename.suffix != ".zvtk":
            msg = f"Filename must end in '.zvtk', not '{self._filename.suffix}'"
            raise ValueError(msg)

        self._arrays: dict[str, NDArray[Any]] = {}
        self._ds = pv.wrap(ds)

    def _add_ds_arrays(self, ds: DataSet, *, force_int32: bool) -> None:
        """Extract dataset data as arrays."""
        ds_id = _make_ds_id(ds)

        if isinstance(ds, PolyData):
            _add_arrays_polydata(ds, self._arrays, force_int32=force_int32)
        elif isinstance(ds, UnstructuredGrid):
            _add_arrays_ugrid(ds, self._arrays, force_int32=force_int32)
        elif isinstance(ds, ImageData):
            pass
        elif isinstance(ds, PointSet):
            _add_arrays_pointset(ds, self._arrays)
        elif isinstance(ds, RectilinearGrid):
            _add_arrays_rgrid(ds, self._arrays)
        elif isinstance(ds, MultiBlock):
            multi_meta = MultiBlockMetadata.from_multiblock(ds)
            meta_bytes = multi_meta.to_json().encode("utf-8")
            self._arrays[f"{ds_id}{MULTIBLOCK_METADATA_KEY}"] = np.frombuffer(meta_bytes, dtype=np.uint8)

            for ds_child in ds:
                self._add_ds_arrays(ds_child, force_int32=force_int32)
            return
        else:
            msg = f"Unsupported type {type(ds)}"
            raise TypeError(msg)

        for key, array in ds.point_data.items():
            self._arrays[f"{ds_id}{key}{POINT_DATA_SUFFIX}"] = array
        for key, array in ds.cell_data.items():
            self._arrays[f"{ds_id}{key}{CELL_DATA_SUFFIX}"] = array
        for key, array in ds.field_data.items():
            self._arrays[f"{ds_id}{key}{FIELD_DATA_SUFFIX}"] = array

        # dataset metadata
        metadata = DataSetMetadata.from_dataset(ds)
        meta_bytes = metadata.to_json().encode("utf-8")
        self._arrays[f"{ds_id}{DS_METADATA_KEY}"] = np.frombuffer(meta_bytes, dtype=np.uint8)

    def write(
        self,
        *,
        progress_bar: bool = False,
        force_int32: bool = True,
        level: int = 3,
        n_threads: int | None = None,
    ) -> None:
        """Write the dataset."""
        self._add_ds_arrays(self._ds, force_int32=force_int32)

        # optimal number of threads is based on how much we're writing to disk
        n_bytes = sum([arr.nbytes for arr in self._arrays.values()])
        n_threads = _set_n_threads(n_threads, n_bytes)

        # finally, append file metadata as the final frame
        file_meta = ZvtkFileMetadata(
            frame_names=list(self._arrays.keys()),  # frame order matters
            compression_level=level,
        )
        meta_bytes = file_meta.to_json().encode("utf-8")
        self._arrays[FILE_METADATA_KEY] = np.frombuffer(meta_bytes, dtype=np.uint8)

        cctx = zstd.ZstdCompressor(level=level, threads=n_threads)
        frame_meta = []  # list of tuples: (compressed_end, decompressed_size)
        with self._filename.open("wb") as fout, cctx.stream_writer(fout) as compressor:
            for name, arr in tqdm(self._arrays.items(), desc="Compressing", disable=not progress_bar):
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

        # Write out zvtk frame sizes as the last bit. This isn't compatible
        # with zstandard compression, but we're using this for decompression
        with self._filename.open("ab") as fout:
            fout.writelines(struct.pack("<QQ", off, dsz) for off, dsz in frame_meta)  # 16 bytes per frame
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


def _add_data(ds_id: str, ds: DataSet, segment_dict: dict[str, Any]) -> None:
    # add point and cell data
    point_data = ds.point_data
    cell_data = ds.cell_data
    field_data = ds.field_data
    for key, array in segment_dict.items():
        if not key.startswith(ds_id):
            continue

        # uid size is 16
        if key.endswith(POINT_DATA_SUFFIX):
            point_data.set_array(array, key[16 : -len(POINT_DATA_SUFFIX)])
        if key.endswith(CELL_DATA_SUFFIX):
            cell_data.set_array(array, key[16 : -len(CELL_DATA_SUFFIX)])
        if key.endswith(FIELD_DATA_SUFFIX):
            field_data.set_array(array, key[16 : -len(FIELD_DATA_SUFFIX)])


def _segments_to_ugrid(ds_id: str, segments: dict[str, Any]) -> UnstructuredGrid:
    cells = _extract_cell_array(ds_id, CELLS, segments)

    celltypes = segments[f"{ds_id}{CELL_TYPES_KEY}"]
    celltypes_vtk = numpy_to_vtk(celltypes, deep=False, array_type=VTK_UNSIGNED_CHAR)

    ugrid = UnstructuredGrid()
    ugrid.points = segments[f"{ds_id}{POINTS_KEY}"]

    poly = _extract_cell_array(ds_id, POLYHEDRON, segments)
    poly_loc = _extract_cell_array(ds_id, POLYHEDRON_LOCATION, segments)

    if poly and poly_loc:
        ugrid.SetPolyhedralCells(
            celltypes_vtk,
            cells,
            poly_loc,
            poly,
        )
    else:
        ugrid.SetCells(celltypes_vtk, cells)

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


def _segments_to_polydata(ds_id: str, segments: dict[str, Any]) -> PolyData:
    pdata = PolyData()
    pdata.points = segments[f"{ds_id}{POINTS_KEY}"]

    pdata.SetPolys(_extract_cell_array(ds_id, POLYS, segments))
    pdata.SetLines(_extract_cell_array(ds_id, LINES, segments))
    pdata.SetStrips(_extract_cell_array(ds_id, STRIPS, segments))
    pdata.SetVerts(_extract_cell_array(ds_id, VERTS, segments))

    return pdata


def _segments_to_pointset(ds_id: str, segments: dict[str, Any]) -> PointSet:
    return PointSet(segments[f"{ds_id}{POINTS_KEY}"])


def _metadata_to_imagedata(metadata: DataSetMetadata) -> ImageData:
    return ImageData(
        dimensions=metadata.dimensions,
        origin=metadata.origin,
        spacing=metadata.spacing,
        direction_matrix=metadata.direction_matrix,
        offset=metadata.offset,
    )


def _segments_to_rgrid(ds_id: str, segments: dict[str, Any]) -> RectilinearGrid:
    if f"{ds_id}{RGRID_X_SUFFIX}" in segments:
        rgrid = RectilinearGrid(
            segments.get(f"{ds_id}{RGRID_X_SUFFIX}"),
            segments.get(f"{ds_id}{RGRID_Y_SUFFIX}"),
            segments.get(f"{ds_id}{RGRID_Z_SUFFIX}"),
        )
    else:
        rgrid = RectilinearGrid()

    return rgrid


def _apply_metadata(ds: DataSet, metadata: DataSetMetadata) -> None:
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


class _DataSetReader:
    def __init__(
        self,
        metadata: MultiBlockMetadata | DataSetMetadata,
        parent: Reader,
    ) -> None:
        self._meta = metadata
        self._parent = parent
        self._children: list[_DataSetReader] = []

        if isinstance(metadata, MultiBlockMetadata):
            for child in metadata.children.values():
                self._children.append(_DataSetReader(child, parent))

    def __getitem__(self, idx: int) -> _DataSetReader:
        if not isinstance(self._meta, MultiBlockMetadata):
            raise TypeError("Only MultiBlock nodes are indexable")
        return self._children[idx]

    def __len__(self) -> int:
        if not isinstance(self._meta, MultiBlockMetadata):
            return 0
        return len(self._children)

    @property
    def uid(self) -> str:
        return self._meta.uid

    @property
    def children(self) -> list[_DataSetReader]:
        return self._children

    def read(self) -> DataSet | MultiBlock:
        if isinstance(self._meta, DataSetMetadata):
            return self._parent._read_ds(self.uid)  # noqa: SLF001
        if isinstance(self._meta, MultiBlockMetadata):
            mb = MultiBlock()
            for child in self._children:
                mb.append(child.read())
            return mb
        raise RuntimeError("Unknown metadata type")

    def __repr__(self) -> str:
        return self._repr_recursive(prefix="", is_last=True)

    def _repr_recursive(self, prefix: str = "", is_last: bool = True) -> str:
        connector = "└─ " if is_last else "├─ "
        if isinstance(self._meta, DataSetMetadata):
            return f"{prefix}{connector}{self._meta.ds_type}"
        if isinstance(self._meta, MultiBlockMetadata):
            lines = [f"{prefix}{connector}MultiBlock(children={len(self._children)})"]
            for i, child in enumerate(self._children):
                last = i == len(self._children) - 1
                child_prefix = prefix + ("   " if is_last else "│  ")
                lines.append(child._repr_recursive(child_prefix, last))  # noqa: SLF001
            return "\n".join(lines)
        return f"{prefix}{connector}Unknown"


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
        self._selected_point_arrays: set[str] | None = None
        self._selected_cell_arrays: set[str] | None = None
        self._selected_field_arrays: set[str] | None = None

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
            frame_meta = [struct.unpack("<QQ", meta_data[i * 16 : (i + 1) * 16]) for i in range(num_frames)]
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
        self._metadata = self._load_file_metadata()
        self._ds_metadata = self._load_root_dataset_meta()
        self.__ds_reader: _DataSetReader | None = None

    def __getitem__(self, idx: int) -> _DataSetReader:
        return self._ds_reader[idx]

    def __len__(self) -> int:
        return len(self._ds_reader)

    @property
    def _ds_reader(self) -> _DataSetReader:
        if self.__ds_reader is None:
            self.__ds_reader = self._load_ds_reader()

        return self.__ds_reader

    def _load_root_dataset_meta(self) -> None:
        """
        Identify all datasets.

        There may be more than one dataset if it's a multi-block dataset, but
        there's always a root dataset.
        """
        dmeta = [n for n in self._metadata.frame_names if n.endswith(DS_METADATA_KEY)]
        if not dmeta:  # pragma: no cover
            msg = "No dataset metadata found"
            raise RuntimeError(msg)
        return self._load_ds_meta(dmeta[0])

    def _load_ds_meta(self, key: str) -> DataSetMetadata | MultiBlockMetadata:
        index = self._metadata.frame_names.index(key)
        dctx = zstd.ZstdDecompressor()

        # read in only the segment
        segments = dctx.multi_decompress_to_buffer(
            [self._frames[index]],
            decompressed_sizes=self._decompressed_sizes[index * 8 : (index + 1) * 8],
            threads=0,  # tiny
        )

        name, arr = _reconstruct_array(segments[0])
        if name.endswith(MULTIBLOCK_METADATA_KEY):
            return MultiBlockMetadata.from_array(arr)
        if name.endswith(DS_METADATA_KEY):
            return DataSetMetadata.from_array(arr)

        msg = "Metadata key invalid."
        raise RuntimeError(msg)

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

    def _load_file_metadata(self) -> ZvtkFileMetadata:
        """Load the metadata from the zvtk file without full decompression."""
        dctx = zstd.ZstdDecompressor()

        # read in only the last segment
        segments = dctx.multi_decompress_to_buffer(
            [self._frames[-1]],
            decompressed_sizes=self._decompressed_sizes[-8:],
            threads=0,  # tiny
        )
        name, arr = _reconstruct_array(segments[0])
        if name != FILE_METADATA_KEY:  # pragma: no cover
            msg = "File metadata not found in zvtk file."
            raise RuntimeError(msg)

        metadata = ZvtkFileMetadata.from_json(arr.tobytes().decode("utf-8"))

        if metadata.file_version > FILE_VERSION:
            warnings.warn(
                f"The file version {metadata.file_version} of this zvtk file is "
                f"newer than the version supported by this library {FILE_VERSION}. "
                "This file may fail to read. Consider upgrading `zvtk`.",
                stacklevel=0,
            )

        return metadata

    def _read_ds(self, ds_id: str, n_threads: int | None = None) -> DataSet:
        """Read a single dataset."""
        # map frame indices to names using metadata
        frame_names = self._metadata.frame_names
        if frame_names is None:  # pragma: no cover
            msg = "Frame names not found in metadata."
            raise RuntimeError(msg)

        excluded = set()
        for name in self.available_point_arrays - self.selected_point_arrays:
            excluded.add(f"{ds_id}{name}{POINT_DATA_SUFFIX}")
        for name in self.available_cell_arrays - self.selected_cell_arrays:
            excluded.add(f"{ds_id}{name}{CELL_DATA_SUFFIX}")
        for name in self.available_field_arrays - self.selected_field_arrays:
            excluded.add(f"{ds_id}{name}{FIELD_DATA_SUFFIX}")

        selected_frames = []
        sizes = []
        selected_frame_names = set(frame_names) - excluded

        for ii, frame_name in enumerate(frame_names):
            if not frame_name.startswith(ds_id):
                continue
            if frame_name in selected_frame_names:
                selected_frames.append(self._frames[ii])
                # 8 bytes per frame
                sizes.append(self._decompressed_sizes[ii * 8 : (ii + 1) * 8])

        # Decompress with multi-threaded buffer API
        d_sizes_bytes = b"".join(sizes)
        ds_size = np.frombuffer(d_sizes_bytes, dtype=np.uint64).sum()
        n_threads = _set_n_threads(n_threads, ds_size)
        if selected_frames:
            dctx = zstd.ZstdDecompressor()
            segments_raw = dctx.multi_decompress_to_buffer(
                selected_frames,
                decompressed_sizes=d_sizes_bytes,
                threads=n_threads,
            )
            segments = dict(_reconstruct_array(segment) for segment in segments_raw)
        else:
            segments = {}

        return self._segments_to_ds(ds_id, segments)

    def _load_ds_reader(self) -> _DataSetReader:
        """Read metadata hierarchy from the zvtk file."""
        if not isinstance(self._ds_metadata, MultiBlockMetadata):
            return self._ds_metadata

        # find only metadata frames
        frame_names = self._metadata.frame_names
        selected_frames = []
        sizes = []
        for ii, name in enumerate(frame_names):
            if name.endswith((MULTIBLOCK_METADATA_KEY, DS_METADATA_KEY)):
                selected_frames.append(self._frames[ii])
                sizes.append(self._decompressed_sizes[ii * 8 : (ii + 1) * 8])

        d_sizes_bytes = b"".join(sizes)
        dctx = zstd.ZstdDecompressor()
        segments_raw = dctx.multi_decompress_to_buffer(
            selected_frames,
            decompressed_sizes=d_sizes_bytes,
            threads=0,
        )
        segments = dict(_reconstruct_array(segment) for segment in segments_raw)

        # decode metadata objects
        mblock_meta: dict[str, MultiBlockMetadata] = {}
        dataset_meta: dict[str, DataSetMetadata] = {}
        for key, segment in segments.items():
            if key.endswith(MULTIBLOCK_METADATA_KEY):
                meta = MultiBlockMetadata.from_array(segment)
                mblock_meta[meta.uid] = meta
            elif key.endswith(DS_METADATA_KEY):
                uid = key[:16]
                meta = DataSetMetadata.from_array(segment)
                dataset_meta[uid] = meta

        # assemble hierarchy tree by wiring children to their metadata
        for uid, m in mblock_meta.items():
            children_meta = {}
            for child_uid in m.children:
                if child_uid in mblock_meta:
                    children_meta[child_uid] = mblock_meta[child_uid]
                elif child_uid in dataset_meta:
                    children_meta[child_uid] = dataset_meta[child_uid]
                else:  # pragma: no cover
                    msg = f"Metadata child '{child_uid}' not found for multiblock '{uid}'"
                    raise RuntimeError(msg)
            m.children = children_meta

        root_uid = self._ds_metadata.uid

        if root_uid not in mblock_meta:  # pragma: no cover
            msg = "Top-level multiblock metadata not found."
            raise RuntimeError(msg)

        return _DataSetReader(mblock_meta[root_uid], self)

    def read(self, n_threads: int | None = None) -> DataSet:
        """Read in the dataset from the zvtk file."""
        if not isinstance(self._ds_metadata, MultiBlockMetadata):
            return self._read_ds(self._ds_metadata.uid, n_threads)

        # read everything
        n_threads = _set_n_threads(n_threads, self.nbytes)

        dctx = zstd.ZstdDecompressor()
        segments_raw = dctx.multi_decompress_to_buffer(
            self._frames,
            decompressed_sizes=self._decompressed_sizes,
            threads=n_threads,
        )
        segments = dict(_reconstruct_array(segment) for segment in segments_raw)

        mblock_meta = []
        dataset_map: dict[str, DataSet] = {}
        for key, segment in segments.items():
            if key.endswith(MULTIBLOCK_METADATA_KEY):
                mblock_meta.append(MultiBlockMetadata.from_array(segment))
            elif key.endswith(DS_METADATA_KEY):
                uid = key[:16]
                dataset_map[uid] = self._segments_to_ds(key[:16], segments)

        # Build empty MultiBlock objects for every multiblock metadata entry.
        multiblock_map: dict[str, MultiBlock] = {m.uid: MultiBlock() for m in mblock_meta}

        # Populate each MultiBlock using its children list. Children may be
        # datasets or other multiblocks (nested multiblocks).
        for m in mblock_meta:
            mb = multiblock_map[m.uid]
            for child_uid in m.children:
                if child_uid in multiblock_map:
                    mb.append(multiblock_map[child_uid])
                elif child_uid in dataset_map:
                    mb.append(dataset_map[child_uid])
                else:  # pragma: no cover
                    msg = f"Multiblock child '{child_uid}' not found for multiblock '{m.uid}'"
                    raise RuntimeError(msg)

        # Return the top-level multiblock identified by the root metadata uid.
        root_uid = self._ds_metadata.uid
        if root_uid not in multiblock_map:  # pragma: no cover
            msg = "Top-level multiblock metadata not found."
            raise RuntimeError(msg)

        return multiblock_map[root_uid]

    def _segments_to_ds(self, ds_id: str, segments: dict[str, Any]) -> DataSet:
        meta_arr = segments[f"{ds_id}{DS_METADATA_KEY}"]
        ds_metadata = DataSetMetadata.from_array(meta_arr)

        # convert this to match when Python 3.9 goes EOL
        ds_type = ds_metadata.ds_type
        if ds_type == "UnstructuredGrid":
            ds = _segments_to_ugrid(ds_id, segments)
        elif ds_type == "PolyData":
            ds = _segments_to_polydata(ds_id, segments)
        elif ds_type == "ImageData":
            ds = _metadata_to_imagedata(ds_metadata)
        elif ds_type == "PointSet":
            ds = _segments_to_pointset(ds_id, segments)
        elif ds_type == "RectilinearGrid":
            ds = _segments_to_rgrid(ds_id, segments)
        else:
            msg = f"zvtk does not support DataSet type `{ds_type}` for decompression"
            raise RuntimeError(msg)

        _add_data(ds_id, ds, segments)
        _apply_metadata(ds, ds_metadata)
        return ds

    @property
    def available_point_arrays(self) -> set[str]:
        """Return a set of all point array names available in the dataset."""
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return set()
        return set(self._ds_metadata.point_data_keys)

    @property
    def available_cell_arrays(self) -> set[str]:
        """Return a set of all cell array names available in the dataset."""
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return set()
        return set(self._ds_metadata.cell_data_keys)

    @property
    def available_field_arrays(self) -> set[str]:
        """Return a set of all field array names available in the dataset."""
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return set()
        return set(self._ds_metadata.field_data_keys)

    @property
    def selected_point_arrays(self) -> set[str]:
        """
        Return the set of currently selected point arrays to read.

        Defaults to all available arrays.
        """
        if self._selected_point_arrays is None:
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
            `available_point_arrays`. An empty set (``set()``) deselects all.

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
        if self._selected_cell_arrays is None:
            return self.available_cell_arrays.copy()
        return self._selected_cell_arrays

    @selected_cell_arrays.setter
    def selected_cell_arrays(self, value: set[str]) -> None:
        """
        Set the cell arrays to read from the file.

        Parameters
        ----------
        value : set[str]
            A set of cell array names to read. All names must exist in
            `available_cell_arrays`. An empty set (``set()``) deselects all.

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
        if self._selected_field_arrays is None:
            return self.available_field_arrays.copy()
        return self._selected_field_arrays

    @selected_field_arrays.setter
    def selected_field_arrays(self, value: set[str]) -> None:
        """
        Set the field arrays to read from the file.

        Parameters
        ----------
        value : set[str]
            A set of field array names to read. All names must exist in
            `available_field_arrays`. An empty set (``set()``) deselects all.

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

        ds_md = self._ds_metadata
        header = [
            f"zvtk.Decompressor ({hex(id(self))})",
            f"  File:               {self._filename}",
            f"  File Version:       {self._metadata.file_version}",
            f"  Compression:        {self._metadata.compression}",
            f"  Compression Level:  {self._metadata.compression_level}",
        ]

        if ds_md.ds_type == "MultiBlock":
            header.append(f"  Dataset Type:       {ds_md.ds_type}")
            header.append("  Hierarchy:")
            header.append(self._ds_reader._repr_recursive(prefix="    ", is_last=True))  # noqa: SLF001
        else:
            header.extend(
                [
                    f"  Dataset Type:       {ds_md.ds_type}",
                    f"  N Points:           {ds_md.n_points} ({ds_md.points_dtype})",
                    f"  N Cells:            {ds_md.n_cells}",
                ]
            )

            # data arrays
            header.extend(_format_dsa("Point", ds_md.point_data_keys))
            header.extend(_format_dsa("Cell", ds_md.cell_data_keys))
            if ds_md.field_data_keys:
                lines = ["  Field arrays"]
                for k, info in ds_md.field_data_keys.items():
                    shape = tuple(info.shape)
                    lines.append(f"      {k:<24} {info.dtype:<10} {shape}")
                header.extend(lines)

        return "\n".join(header)
