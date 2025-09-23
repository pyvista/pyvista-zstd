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
from typing import Literal
import warnings

import numpy as np
import pyvista as pv
from pyvista.core.composite import MultiBlock
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import StructuredGrid
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
from zstandard import BufferWithSegmentsCollection

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pyvista.core.dataset import DataSet

FILE_VERSION = 0
FILE_VERSION_KEY = "FILE_VERSION"
DS_TYPE_KEY = "ds_type"
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

Compression = Literal["zstandard", "lz4"]

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

UID_N_CHAR = 16
EMPTY_DS = "EMPTY_DS________"  # must be 16 char to align with UID

VTK_UNSIGNED_CHAR = 3
VTK_FLOAT = 10
VTK_DOUBLE = 11


@dataclass(slots=True, frozen=True)
class ArrayInfo:
    """Array metadata."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(slots=True, frozen=True)
class ZvtkFileMetadata:
    """Zvtk file metadata."""

    frame_names: list[str]
    compression_level: int
    compression: Compression = "zstandard"
    file_version: int = FILE_VERSION

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> ZvtkFileMetadata:
        """Create from JSON."""
        return cls(**json.loads(s))

    def to_array(self) -> NDArray[np.uint8]:
        """Output as a numpy uint8 array."""
        meta_bytes = self.to_json().encode("utf-8")
        return np.frombuffer(meta_bytes, dtype=np.uint8)


@dataclass
class MultiBlockMetadata:
    """MultiBlock metadata."""

    uid: str
    children: list[str]
    ds_type = "MultiBlock"
    children_keys: list[str]

    # optional and used for ds reader
    children_ds: dict[str, MultiBlockMetadata | DataSetMetadata | None] | None = None

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> MultiBlockMetadata:
        """Create from JSON."""
        return cls(**json.loads(s))

    @classmethod
    def from_array(cls, arr: NDArray[np.uint8]) -> MultiBlockMetadata:
        """Create from a numpy uint8 array."""
        raw_json = arr.tobytes().decode("utf-8")  # copy, but it's tiny
        return MultiBlockMetadata.from_json(raw_json)

    def to_array(self) -> NDArray[np.uint8]:
        """Output as a numpy uint8 array."""
        meta_bytes = self.to_json().encode("utf-8")
        return np.frombuffer(meta_bytes, dtype=np.uint8)


@dataclass(slots=True, frozen=True)
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
    def from_dataset(
        cls,
        ds: pv.DataSet,
        point_info: dict[str, ArrayInfo],
        cell_info: dict[str, ArrayInfo],
        field_info: dict[str, ArrayInfo],
    ) -> DataSetMetadata:
        """Create metadata from a dataset."""
        # Many pyvista calls require intermediate object assembly, side step or
        # do once when possible.

        # Get points
        vtk_dtype = ds.GetPoints().GetDataType()
        if vtk_dtype == VTK_FLOAT:
            points_dtype = np.float32
        elif vtk_dtype == VTK_DOUBLE:
            points_dtype = np.float64
        else:  # pragma: no cover
            msg = "Invalid points datatype. Should be float or double"
            raise RuntimeError(msg)

        pd = ds.point_data
        cd = ds.cell_data
        kwargs: dict[str, Any] = {
            "ds_type": type(ds).__name__,
            "uid": _make_ds_id(ds),
            "n_points": ds.n_points,
            "points_dtype": str(points_dtype),
            "n_cells": ds.n_cells,
            "celltypes_dtype": str(ds.celltypes.dtype) if hasattr(ds, "celltypes") else None,
            "point_data_keys": point_info,
            "cell_data_keys": cell_info,
            "field_data_keys": field_info,
            "point_data_active_scalars_name": pd.active_scalars_name,
            "point_data_active_vectors_name": pd.active_vectors_name,
            "point_data_active_texture_coordinates_name": pd.active_texture_coordinates_name,
            "point_data_active_normals_name": pd.active_normals_name,
            "cell_data_active_scalars_name": cd.active_scalars_name,
            "cell_data_active_vectors_name": cd.active_vectors_name,
            "cell_data_active_texture_coordinates_name": cd.active_texture_coordinates_name,
            "cell_data_active_normals_name": cd.active_normals_name,
        }

        if isinstance(ds, pv.ImageData):
            kwargs.update(
                dimensions=ds.dimensions,
                origin=ds.origin,
                spacing=ds.spacing,
                direction_matrix=ds.direction_matrix.tolist(),
                offset=ds.offset,
            )
        elif isinstance(ds, pv.StructuredGrid):
            kwargs["dimensions"] = ds.dimensions

        return cls(**kwargs)

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_array(cls, arr: NDArray[np.uint8]) -> DataSetMetadata:
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

    def to_array(self) -> NDArray[np.uint8]:
        """Output as a numpy uint8 array."""
        meta_bytes = self.to_json().encode("utf-8")
        return np.frombuffer(meta_bytes, dtype=np.uint8)


def _format_bytes(size: float) -> str:
    """Return a byte size in a human readable format."""
    kb = 1024
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < kb:
            return f"{size:.1f}{unit}"
        size = size / kb
    return f"{size:.1f}TB"


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


def _add_arrays_ugrid(
    ds: UnstructuredGrid, arrays: dict[str, NDArray[Any]], ds_id: str | None = None, *, force_int32: bool = True
) -> None:
    if ds_id is None:
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


def _add_arrays_esgrid(
    ds: ExplicitStructuredGrid, arrays: dict[str, NDArray[Any]], *, force_int32: bool = True
) -> None:
    ds_id = _make_ds_id(ds)
    ugrid = ds.cast_to_unstructured_grid()
    _add_arrays_ugrid(ugrid, arrays, ds_id, force_int32=force_int32)


def _add_arrays_sgrid(ds: StructuredGrid, arrays: dict[str, NDArray[Any]]) -> None:
    ds_id = _make_ds_id(ds)
    arrays[f"{ds_id}{POINTS_KEY}"] = ds.points


# eventually add: compression: Compression = "zstandard",
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
    * :class:`pyvista.StructuredGrid`
    * :class:`pyvista.RectilinearGrid`
    * :class:`pyvista.StructuredGrid`
    * :class:`pyvista.UnstructuredGrid`
    * :class:`pyvista.MultiBlock`
    * :class:`pyvista.ExplicitStructuredGrid`

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
        Show a progress bar while writing to disk.
    level : int, default: 3
        Compression level. Valid values are all negative integers through
        22. Lower values generally yield faster operations with lower
        compression ratios. Higher values are generally slower but compress
        better.
    n_threads : int, optional
        Number of threads to use when compressing. A value of ``-1`` uses all
        available cores and ``0`` disables multi-threading.

    """
    writer = Writer(ds, filename)

    # if compression == "zstandard":
    writer.write(
        progress_bar=progress_bar,
        force_int32=force_int32,
        level=level,
        n_threads=n_threads,
    )


def _make_ds_id(ds: DataSet) -> str:
    """Make a unique dataset ID using the memory address."""
    # padded for 32-bit
    return f"{id(ds):016x}"


def _pack_array_metadata(name: str, arr: np.ndarray) -> bytes:
    parts = [
        struct.pack("<I", len(name)),
        name.encode("utf-8"),
        struct.pack("<I", arr.ndim),
    ]
    parts.extend(struct.pack("<Q", dim) for dim in arr.shape)
    parts.append(arr.dtype.str.encode("utf-8").ljust(UID_N_CHAR, b" "))
    return b"".join(parts)


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

        # used to hold a reference to the dataset. This is necessary for
        # multiblocks to avoid having them collected and getting duplicate
        # memory addresses
        self._refs: list[DataSet | MultiBlock] = []

    def _add_ds_arrays(self, ds: DataSet, *, force_int32: bool) -> None:  # noqa: C901, PLR0912
        """Extract dataset data as arrays."""
        # Hold on to a reference of the dataset to avoid it being collected
        # while we generate all memory IDs
        self._refs.append(ds)
        ds_id = _make_ds_id(ds)

        if isinstance(ds, PolyData):
            _add_arrays_polydata(ds, self._arrays, force_int32=force_int32)
        elif isinstance(ds, UnstructuredGrid):
            _add_arrays_ugrid(ds, self._arrays, force_int32=force_int32)
        elif isinstance(ds, ExplicitStructuredGrid):
            _add_arrays_esgrid(ds, self._arrays, force_int32=force_int32)
        elif isinstance(ds, ImageData):
            pass
        elif isinstance(ds, StructuredGrid):
            _add_arrays_sgrid(ds, self._arrays)
        elif isinstance(ds, PointSet):
            _add_arrays_pointset(ds, self._arrays)
        elif isinstance(ds, RectilinearGrid):
            _add_arrays_rgrid(ds, self._arrays)
        elif isinstance(ds, MultiBlock):
            # placeholder, array insertion order matters
            self._arrays[f"{ds_id}{MULTIBLOCK_METADATA_KEY}"] = None

            child_ids = []
            for ds_child in ds:
                # special handling none edge case
                if ds_child is None:
                    child_ids.append(EMPTY_DS)
                else:
                    child_ids.append(_make_ds_id(ds_child))
                    self._add_ds_arrays(ds_child, force_int32=force_int32)

            # edge case where multiblock can contain a NoneType key
            children_keys = ["None" if key is None else key for key in ds.keys()]  # noqa: SIM118
            multi_meta = MultiBlockMetadata(
                uid=ds_id,
                children=child_ids,
                children_keys=children_keys,
            )
            self._arrays[f"{ds_id}{MULTIBLOCK_METADATA_KEY}"] = multi_meta.to_array()

            return
        else:  # pragma: no cover
            msg = f"Unsupported type {type(ds)}"
            raise TypeError(msg)

        point_info: dict[str, ArrayInfo] = {}
        for key, array in ds.point_data.items():
            self._arrays[f"{ds_id}{key}{POINT_DATA_SUFFIX}"] = array
            point_info[key] = ArrayInfo(shape=array.shape, dtype=str(array.dtype))

        cell_info: dict[str, ArrayInfo] = {}
        for key, array in ds.cell_data.items():
            self._arrays[f"{ds_id}{key}{CELL_DATA_SUFFIX}"] = array
            cell_info[key] = ArrayInfo(shape=array.shape, dtype=str(array.dtype))

        field_info: dict[str, ArrayInfo] = {}
        for key, array in ds.field_data.items():
            self._arrays[f"{ds_id}{key}{FIELD_DATA_SUFFIX}"] = array
            field_info[key] = ArrayInfo(shape=array.shape, dtype=str(array.dtype))

        # supply dataset metadata
        ds_meta = DataSetMetadata.from_dataset(ds, point_info, cell_info, field_info)
        self._arrays[f"{ds_id}{DS_METADATA_KEY}"] = ds_meta.to_array()

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
        self._arrays[FILE_METADATA_KEY] = file_meta.to_array()

        # data to compress must include array metadata and the array
        data: list[bytes] = []
        for name, arr in self._arrays.items():
            # must be a view of uint8 for no copy to bytes
            arr_bytes = arr.ravel().view(np.uint8).data
            data.extend([_pack_array_metadata(name, arr), arr_bytes])

        # Compress multiple pieces of data as a single function call to minimize overhead
        frame_meta: list[tuple[float, float]] = []  # (compressed_end, decompressed_size)
        offset = 0
        with self._filename.open("wb") as fout:
            cctx = zstd.ZstdCompressor(level=level, threads=n_threads)
            buff_seg = cctx.multi_compress_to_buffer(data, threads=n_threads)
            for ii, cdata in enumerate(tqdm(buff_seg, disable=not progress_bar, desc="Writing frames")):
                offset += fout.write(cdata)
                frame_meta.append((offset, len(data[ii])))

            # finally, write out compressed and decompressed size of each segment
            # 16 bytes per frame
            fout.writelines(struct.pack("<QQ", off, dsz) for off, dsz in frame_meta)
            fout.write(struct.pack("<Q", len(frame_meta)))  # total frames at very end

        # no need to hold onto any references as all IDs have been written
        self._refs = []


def _reconstruct_array(meta_segment: BufferSegment, arr_segment: BufferSegment) -> np.ndarray:
    """
    Reconstruct a NumPy array from a single decompressed Zstd frame.

    Frame layout:
    ``[name_len:uint32][name:bytes][ndim:uint32][shape:Q*ndim][dtype:16 bytes][array data]``.

    """
    meta_buf = memoryview(meta_segment)

    offset = 0
    name_len = struct.unpack_from("<I", meta_buf, offset)[0]
    offset += 4
    name = meta_buf[offset : offset + name_len].tobytes().decode("utf-8")
    offset += name_len

    ndim = struct.unpack_from("<I", meta_buf, offset)[0]
    offset += 4

    shape = tuple(struct.unpack_from(f"<{ndim}Q", meta_buf, offset))
    offset += 8 * ndim

    dtype_str = meta_buf[offset : offset + UID_N_CHAR].tobytes().strip().decode("utf-8")
    offset += UID_N_CHAR

    # finally construct the array using the array segment
    data_buf = memoryview(arr_segment)
    data = np.frombuffer(data_buf, dtype=np.dtype(dtype_str)).reshape(shape)
    return name, data


def _raw_segments_to_arrays(
    segments_raw: BufferWithSegmentsCollection,
) -> dict[str, NDArray[Any]]:
    segments = {}
    for ii in range(int(len(segments_raw) / 2)):
        name, arr = _reconstruct_array(segments_raw[ii * 2], segments_raw[ii * 2 + 1])
        segments[name] = arr
    return segments


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
            point_data.set_array(array, key[UID_N_CHAR : -len(POINT_DATA_SUFFIX)])
        if key.endswith(CELL_DATA_SUFFIX):
            cell_data.set_array(array, key[UID_N_CHAR : -len(CELL_DATA_SUFFIX)])
        if key.endswith(FIELD_DATA_SUFFIX):
            field_data.set_array(array, key[UID_N_CHAR : -len(FIELD_DATA_SUFFIX)])


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


def _segments_to_esgrid(ds_id: str, segments: dict[str, Any]) -> ExplicitStructuredGrid:
    return _segments_to_ugrid(ds_id, segments).cast_to_explicit_structured_grid()


def _segments_to_sgrid(ds_id: str, segments: dict[str, Any], metadata: DataSetMetadata) -> StructuredGrid:
    sgrid = StructuredGrid(segments[f"{ds_id}{POINTS_KEY}"])
    sgrid.dimensions = metadata.dimensions
    return sgrid


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
        msg = f"Invalid faces dtype {dtype}. Expected `np.int32` or `np.int64`."
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

    Returns
    -------
    pyvista.DataSet

    Examples
    --------
    >>> import zvtk
    >>> ds = zvtk.read("dataset.zvtk")

    """
    return Reader(filename).read(n_threads=n_threads)


class _DataSetReader:
    def __init__(
        self,
        metadata: MultiBlockMetadata | DataSetMetadata | None,
        parent: Reader,
    ) -> None:
        self._meta = metadata
        self._parent = parent
        self._children: list[_DataSetReader] = []

        if isinstance(metadata, MultiBlockMetadata):
            if metadata.children_ds is None:
                return
            for child in metadata.children_ds.values():
                self._children.append(_DataSetReader(child, parent))

    def __getitem__(self, idx: int) -> _DataSetReader:
        if not isinstance(self._meta, MultiBlockMetadata):
            msg = "Only MultiBlock nodes are indexable."
            raise TypeError(msg)
        return self._children[idx]

    def __len__(self) -> int:
        if not isinstance(self._meta, MultiBlockMetadata):
            msg = "Only MultiBlock nodes have a length."
            raise TypeError(msg)
        return len(self._children)

    @property
    def uid(self) -> str:
        if self._meta is None:
            return EMPTY_DS
        return self._meta.uid

    def read(self) -> DataSet | MultiBlock:
        if isinstance(self._meta, DataSetMetadata):
            return self._parent._read_ds(self.uid)  # noqa: SLF001
        if isinstance(self._meta, MultiBlockMetadata):
            mb = MultiBlock()
            for key, child in zip(self._meta.children_keys, self._children):
                mb[key] = child.read()
            return mb

        if self._meta is None:
            return None

        msg = "Unknown metadata type"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    def __repr__(self) -> str:
        return self._repr_recursive(prefix="", is_last=True)

    def _repr_recursive(self, prefix: str = "", *, is_last: bool = True) -> str:
        connector = "└─ " if is_last else "├─ "
        if isinstance(self._meta, DataSetMetadata):
            return f"{prefix}{connector}{self._meta.ds_type}"
        if isinstance(self._meta, MultiBlockMetadata):
            lines = [f"{prefix}{connector}MultiBlock(children={len(self._children)})"]
            for i, child in enumerate(self._children):
                last = i == len(self._children) - 1
                child_prefix = prefix + ("   " if is_last else "│  ")
                lines.append(child._repr_recursive(child_prefix, is_last=last))  # noqa: SLF001
            return "\n".join(lines)
        if self._meta is None:
            return f"{prefix}{connector}None"

        return f"{prefix}{connector}Unknown"  # pragma: no cover


class Reader:
    """
    Class to control zvtk file decompression.

    Use this class in lieu of :func:`zvtk.read` to fine-tune reading in
    compressed files. With this you can:

    * Inspect the dataset before reading it.
    * Control which arrays to read in.
    * For files containing a :class:`pyvista.MultiBlock`, select which blocks
      to read in.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to the file. Must end in ``.zvtk``.

    Examples
    --------
    First write out an example dataset.

    >>> import pyvista as pv
    >>> import zvtk
    >>> ds = pv.Sphere()
    >>> zvtk.write(ds, "sphere.zvtk")

    Create a reader.

    >>> reader = zvtk.Reader("sphere.zvtk")
    >>> reader
    zvtk.Decompressor (0x7f1ed1496c00)
      File:               sphere.zvtk
      File Version:       0
      Compression:        zstandard
      Compression Level:  3
      Dataset Type:       PolyData
      N Points:           842 (<class 'numpy.float32'>)
      N Cells:            1680
      Point arrays:
          Normals                  float32    (842, 3)

    Disable reading in point arrays and read the dataset.

    >>> reader.selected_point_arrays = set()
    >>> ds_in = reader.read()
    >>> ds_in
    PolyData (0x7f1ece066ce0)
      N Cells:    1680
      N Points:   842
      N Strips:   0
      X Bounds:   -4.993e-01, 4.993e-01
      Y Bounds:   -4.965e-01, 4.965e-01
      Z Bounds:   -5.000e-01, 5.000e-01
      N Arrays:   0

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

            f.seek(-(8 + num_frames * UID_N_CHAR), 2)
            meta_data = f.read(num_frames * UID_N_CHAR)
            frame_meta = [
                struct.unpack("<QQ", meta_data[i * UID_N_CHAR : (i + 1) * UID_N_CHAR]) for i in range(num_frames)
            ]
            frame_starts = [0] + [end for end, _ in frame_meta[:-1]]
            frame_ends = [end for end, _ in frame_meta]
            sizes = [dsz for _, dsz in frame_meta]
            self._decompressed_sizes = struct.pack(f"={len(sizes)}Q", *sizes)
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # store compressed frame sizes
        sizes = []
        for i in range(len(frame_ends)):
            if i == 0:
                sizes.append(frame_ends[0])
            else:
                sizes.append(frame_ends[i] - frame_ends[i - 1])
        self._compressed_sizes = np.array(sizes, dtype=np.uint64)

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
        """Return an indexed reader."""
        return self._ds_reader[idx]

    def __len__(self) -> int:
        """Return the number of items in the reader."""
        return len(self._ds_reader)

    @property
    def _ds_reader(self) -> _DataSetReader:
        if self.__ds_reader is None:
            self.__ds_reader = self._load_ds_reader()

        return self.__ds_reader

    def _load_root_dataset_meta(self) -> DataSetMetadata | MultiBlockMetadata:
        """
        Return the root dataset metadata.

        There may be more than one dataset if it's a multi-block dataset, but
        there's always a root dataset.
        """
        for frame_name in self._metadata.frame_names:
            if frame_name.endswith(DS_METADATA_KEY):
                return self._load_ds_meta(frame_name)

        msg = "No dataset metadata found"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    def _load_ds_meta(self, key: str) -> DataSetMetadata | MultiBlockMetadata:
        index = self._metadata.frame_names.index(key) * 2  # times two for metadata
        dctx = zstd.ZstdDecompressor()

        # read in only the segment
        segments = dctx.multi_decompress_to_buffer(
            [self._frames[index], self._frames[index + 1]],
            decompressed_sizes=self._decompressed_sizes[index * 8 : (index + 2) * 8],
            threads=0,  # tiny
        )

        name, arr = _reconstruct_array(*segments)
        if name.endswith(MULTIBLOCK_METADATA_KEY):
            return MultiBlockMetadata.from_array(arr)
        if name.endswith(DS_METADATA_KEY):
            return DataSetMetadata.from_array(arr)

        msg = "Metadata key invalid."  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

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
            [self._frames[-2], self._frames[-1]],
            decompressed_sizes=self._decompressed_sizes[-UID_N_CHAR:],
            threads=0,  # tiny
        )
        name, arr = _reconstruct_array(*segments)
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

    # @profile
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

        # downselect to the matching dataset id
        selected_frame_names = {f for f in selected_frame_names if f.startswith(ds_id)}

        n_frames = len(frame_names)
        if len(selected_frame_names) == n_frames:
            # Decompress with multi-threaded buffer API
            dctx = zstd.ZstdDecompressor()
            segments_raw = dctx.multi_decompress_to_buffer(
                self._frames,
                decompressed_sizes=self._decompressed_sizes,
                threads=_set_n_threads(n_threads, self.nbytes),
            )

        elif selected_frame_names:
            for ii, frame_name in enumerate(frame_names):
                if not frame_name.startswith(ds_id):
                    continue
                if frame_name in selected_frame_names:
                    idx = ii * 2  # double for metadata
                    selected_frames.extend([self._frames[idx], self._frames[idx + 1]])
                    # 8 bytes per frame
                    sizes.append(self._decompressed_sizes[idx * 8 : (idx + 2) * 8])

            # Decompress with multi-threaded buffer API
            d_sizes_bytes = b"".join(sizes)
            ds_size = np.frombuffer(d_sizes_bytes, dtype=np.uint64).sum()
            n_threads = _set_n_threads(n_threads, ds_size)
            dctx = zstd.ZstdDecompressor()
            segments_raw = dctx.multi_decompress_to_buffer(
                selected_frames,
                decompressed_sizes=d_sizes_bytes,
                threads=n_threads,
            )
        else:  # pragma: no cover
            msg = "No selected frames"
            raise RuntimeError(msg)

        segments = _raw_segments_to_arrays(segments_raw)  # pragma: no cover
        return self._segments_to_ds(ds_id, segments)  # pragma: no cover

    def _load_ds_reader(self) -> _DataSetReader:  # noqa: C901, PLR0912
        """Read metadata hierarchy from the zvtk file."""
        if not isinstance(self._ds_metadata, MultiBlockMetadata):
            msg = "Can only index a MultiBlock compressed zvtk file."
            raise TypeError(msg)

        # find only metadata frames
        frame_names = self._metadata.frame_names
        selected_frames = []
        sizes = []
        for ii, name in enumerate(frame_names):
            idx = ii * 2  # double for metadata
            if name.endswith((MULTIBLOCK_METADATA_KEY, DS_METADATA_KEY)):
                selected_frames.extend([self._frames[idx], self._frames[idx + 1]])
                sizes.append(self._decompressed_sizes[idx * 8 : (idx + 2) * 8])

        d_sizes_bytes = b"".join(sizes)
        dctx = zstd.ZstdDecompressor()
        segments_raw = dctx.multi_decompress_to_buffer(
            selected_frames,
            decompressed_sizes=d_sizes_bytes,
            threads=0,
        )
        segments = _raw_segments_to_arrays(segments_raw)

        # decode metadata objects
        mblock_meta: dict[str, MultiBlockMetadata] = {}
        dataset_meta: dict[str, DataSetMetadata] = {}
        for key, segment in segments.items():
            if key.endswith(MULTIBLOCK_METADATA_KEY):
                mb_meta = MultiBlockMetadata.from_array(segment)
                mblock_meta[mb_meta.uid] = mb_meta
            elif key.endswith(DS_METADATA_KEY):
                uid = key[:UID_N_CHAR]
                ds_meta = DataSetMetadata.from_array(segment)
                dataset_meta[uid] = ds_meta

        # assemble hierarchy tree by wiring children to their metadata
        for uid, m in mblock_meta.items():
            children_meta: dict[str, MultiBlockMetadata | DataSetMetadata | None] = {}
            for child_uid in m.children:
                if child_uid in mblock_meta:
                    children_meta[child_uid] = mblock_meta[child_uid]
                elif child_uid in dataset_meta:
                    children_meta[child_uid] = dataset_meta[child_uid]
                elif child_uid == EMPTY_DS:
                    children_meta[child_uid] = None
                else:  # pragma: no cover
                    msg = f"Metadata child '{child_uid}' not found for multiblock '{uid}'"
                    raise RuntimeError(msg)
            m.children_ds = children_meta

        root_uid = self._ds_metadata.uid

        if root_uid not in mblock_meta:  # pragma: no cover
            msg = "Top-level multiblock metadata not found."
            raise RuntimeError(msg)

        return _DataSetReader(mblock_meta[root_uid], self)

    def read(self, n_threads: int | None = None) -> DataSet:  # noqa: C901
        """
        Read in the dataset from the zvtk file.

        Parameters
        ----------
        n_threads : int, optional
            Number of threads to use when reading. A value of ``-1`` uses all
            available cores and ``0`` disables multi-threading.

        Examples
        --------
        >>> import pyvista as pv
        >>> import zvtk
        >>> ds = pv.Sphere()
        >>> zvtk.write(ds, "sphere.zvtk")
        >>> reader = zvtk.Reader("sphere.zvtk")
        >>> ds_in = reader.read()
        >>> ds_in
        PolyData (0x7f1eca564520)
          N Cells:    1680
          N Points:   842
          N Strips:   0
          X Bounds:   -4.993e-01, 4.993e-01
          Y Bounds:   -4.965e-01, 4.965e-01
          Z Bounds:   -5.000e-01, 5.000e-01
          N Arrays:   1

        """
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
        segments = _raw_segments_to_arrays(segments_raw)

        mblock_meta = []
        dataset_map: dict[str, DataSet] = {}
        for key, segment in segments.items():
            if key.endswith(MULTIBLOCK_METADATA_KEY):
                mblock_meta.append(MultiBlockMetadata.from_array(segment))
            elif key.endswith(DS_METADATA_KEY):
                uid = key[:UID_N_CHAR]
                dataset_map[uid] = self._segments_to_ds(key[:UID_N_CHAR], segments)

        # Build empty MultiBlock objects for every multiblock metadata entry.
        multiblock_map: dict[str, MultiBlock] = {m.uid: MultiBlock() for m in mblock_meta}

        # Populate each MultiBlock using its children list. Children may be
        # datasets or other multiblocks (nested multiblocks).
        for m in mblock_meta:
            mb = multiblock_map[m.uid]
            for child_key, child_uid in zip(m.children_keys, m.children):
                if child_uid in multiblock_map:
                    mb[child_key] = multiblock_map[child_uid]
                elif child_uid in dataset_map:
                    mb[child_key] = dataset_map[child_uid]
                elif child_uid == EMPTY_DS:
                    mb[child_key] = None
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
        elif ds_type == "StructuredGrid":
            ds = _segments_to_sgrid(ds_id, segments, ds_metadata)
        elif ds_type == "ExplicitStructuredGrid":
            ds = _segments_to_esgrid(ds_id, segments)
        else:  # pragma: no cover
            msg = f"zvtk does not support DataSet type `{ds_type}` for decompression"
            raise RuntimeError(msg)

        _add_data(ds_id, ds, segments)
        _apply_metadata(ds, ds_metadata)
        return ds

    @property
    def available_point_arrays(self) -> set[str]:
        """
        Return a set of all point array names available in the dataset.

        Returns
        -------
        set[str]
            Names of all point arrays available in the dataset.

        Examples
        --------
        First write out an example dataset.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> import zvtk
        >>> ds = pv.Sphere()
        >>> ds.point_data["pdata"] = np.arange(ds.n_points)
        >>> zvtk.write(ds, "sphere.zvtk")

        Create a reader and list available point arrays.

        >>> reader = zvtk.Reader("sphere.zvtk")
        >>> reader.available_point_arrays
        {"Normals", "pdata"}

        """
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return set()
        return set(self._ds_metadata.point_data_keys)

    @property
    def available_cell_arrays(self) -> set[str]:
        """
        Return a set of all cell array names available in the dataset.

        Returns
        -------
        set[str]
            Names of all cell arrays available in the dataset.

        Examples
        --------
        First write out an example dataset.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> import zvtk
        >>> ds = pv.Sphere()
        >>> ds.point_data["pdata"] = np.arange(ds.n_points)
        >>> zvtk.write(ds, "sphere.zvtk")

        Create a reader and list available point arrays.

        >>> reader = zvtk.Reader("sphere.zvtk")
        >>> reader.available_point_arrays
        {"Normals", "pdata"}

        """
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

        if isinstance(ds_md, MultiBlockMetadata):
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

    def show_frame_compression(self) -> str:  # noqa: C901, PLR0912
        """
        Return a table showing compression statistics for each frame in the dataset.

        For MultiBlock datasets, shows a hierarchical view with compression stats
        for each block. For regular datasets, shows stats for each array.

        Examples
        --------
        Download the aero bracket dataset.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> import zvtk
        >>> ds = examples.download_aero_bracket()
        >>> ds
        UnstructuredGrid (0x7fd751589360)
          N Cells:    117292
          N Points:   187037
          X Bounds:   -6.858e-03, 1.118e-01
          Y Bounds:   -1.237e-02, 6.634e-02
          Z Bounds:   -1.638e-02, 1.638e-02
          N Arrays:   3

        Compress it and then show the compressed frame sizes through the reader.

        >>> zvtk.write(ds, "bracket.zvtk")
        >>> reader = zvtk.Reader("bracket.zvtk")
        >>> print(reader.show_frame_compression())
        Dataset ID       Frame Type                      Compressed   Decompressed Ratio
        --------------------------------------------------------------------------------
        00007fd751589360 Points                          1.9MB        2.1MB        0.877
        00007fd751589360 Cell Types                      22.0B        114.5KB      0.000
        00007fd751589360 Offsets: cells                  330.5KB      458.2KB      0.721
        00007fd751589360 Connectivity: cells             2.2MB        4.5MB        0.499
        00007fd751589360 Point Data: displacement        2.0MB        2.1MB        0.935
        00007fd751589360 Point Data: total nonlinear st  4.0MB        4.3MB        0.938
        00007fd751589360 Point Data: von Mises stress    650.7KB      730.6KB      0.891
        --------------------------------------------------------------------------------
        TOTAL                                            11.1MB       14.3MB       0.775

        Note how the compression ratio can be marginally improved by increasing
        the compression level.

        >>> zvtk.write(ds, "bracket.zvtk", level=22)
        >>> reader = zvtk.Reader("bracket.zvtk")
        Dataset ID       Frame Type                      Compressed   Decompressed Ratio
        --------------------------------------------------------------------------------
        00007fd751589360 Points                          1.8MB        2.1MB        0.863
        00007fd751589360 Cell Types                      21.0B        114.5KB      0.000
        00007fd751589360 Offsets: cells                  56.1KB       458.2KB      0.123
        00007fd751589360 Connectivity: cells             1.6MB        4.5MB        0.358
        00007fd751589360 Point Data: displacement        2.0MB        2.1MB        0.937
        00007fd751589360 Point Data: total nonlinear st  4.0MB        4.3MB        0.940
        00007fd751589360 Point Data: von Mises stress    651.7KB      730.6KB      0.892
        --------------------------------------------------------------------------------
        TOTAL                                            10.2MB       14.3MB       0.711

        """
        lines: list[str] = []
        frame_names = self._metadata.frame_names

        # Frame sizes are [array header, array, ..., metadata frame]
        # skip headers and metadata frame
        d_sizes = self.decompressed_sizes[1:-1:2]
        c_sizes = self._compressed_sizes[1:-1:2]

        # Group frames by dataset ID for better organization
        frame_data = []
        for name, comp_size, decomp_size in zip(frame_names, c_sizes, d_sizes):
            # Extract dataset ID and frame type
            if len(name) >= UID_N_CHAR:
                ds_id = name[:UID_N_CHAR]
                suffix = name[UID_N_CHAR:]
            else:
                continue

            # always skip metadata
            if suffix.endswith("metadata"):
                continue

            # Determine frame type and human-readable name
            if suffix.endswith(POINT_DATA_SUFFIX):
                array_name = suffix[: -len(POINT_DATA_SUFFIX)]
                frame_type = f"Point Data: {array_name}"
            elif suffix.endswith(CELL_DATA_SUFFIX):
                array_name = suffix[: -len(CELL_DATA_SUFFIX)]
                frame_type = f"Cell Data: {array_name}"
            elif suffix.endswith(FIELD_DATA_SUFFIX):
                array_name = suffix[: -len(FIELD_DATA_SUFFIX)]
                frame_type = f"Field Data: {array_name}"
            elif suffix == POINTS_KEY:
                frame_type = "Points"
            elif suffix == CELL_TYPES_KEY:
                frame_type = "Cell Types"
            elif suffix.endswith(OFFSET_SUFFIX):
                array_name = suffix[: -len(OFFSET_SUFFIX)]
                frame_type = f"Offsets: {array_name}"
            elif suffix.endswith(CONNECTIVITY_SUFFIX):
                array_name = suffix[: -len(CONNECTIVITY_SUFFIX)]
                frame_type = f"Connectivity: {array_name}"
            elif suffix.endswith((RGRID_X_SUFFIX, RGRID_Y_SUFFIX, RGRID_Z_SUFFIX)):
                coord = suffix[-8]  # x, y, or z
                frame_type = f"RGrid {coord.upper()} Coords"
            else:
                frame_type = suffix

            ratio = comp_size / decomp_size if decomp_size > 0 else 0
            frame_data.append((ds_id, frame_type, comp_size, decomp_size, ratio))

        # Print header
        lines.append(f"{'Dataset ID':<16} {'Frame Type':<31} {'Compressed':<12} {'Decompressed':<12} {'Ratio':<5}")
        lines.append("-" * 80)

        # Group by dataset ID for MultiBlock organization
        # Single dataset - print all frames
        total_comp = 0
        total_decomp = 0
        for ds_id, frame_type, comp_size, decomp_size, ratio in frame_data:
            total_comp += comp_size
            total_decomp += decomp_size
            lines.append(
                f"{ds_id:<16} {frame_type[:30]:<31} {_format_bytes(comp_size):<12} "
                f"{_format_bytes(decomp_size):<12} {ratio:.3f}"
            )

        lines.append("-" * 80)
        overall_ratio = total_comp / total_decomp if total_decomp > 0 else 0
        lines.append(
            f"{'TOTAL':<16} {'':<31} {_format_bytes(total_comp):<12} "
            f"{_format_bytes(total_decomp):<12} {overall_ratio:.3f}"
        )

        return "\n".join(lines)
