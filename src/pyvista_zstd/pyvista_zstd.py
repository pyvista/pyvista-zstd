"""
Compress VTK objects using zstandard.

We're writing everything out using `zstandard frames
<https://python-zstandard.readthedocs.io/en/latest/concepts.html>`_.

"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
import warnings

import numpy as np
import pyvista as pv

from pyvista_zstd import _capi
from pyvista_zstd.thumbnail import ThumbnailInfo
from pyvista_zstd.thumbnail import _thumbnail_png

# Import VTK through PyVista rather than from ``vtkmodules`` directly, so the
# classes constructed here always come from the same VTK binding PyVista itself
# is built on. PyVista can be built against a binding other than the stock
# ``vtkmodules`` wheel -- ``cvista``, for instance -- and in that case a
# ``vtkmodules`` cell array is a different C++ type from the one a PyVista
# ``PolyData`` expects. ``SetPolys`` then rejects it with a bare
# ``TypeError: SetPolys argument 1:`` and every ``.pv`` file becomes unreadable.
#
# ``pyvista._vtk`` resolves to whichever binding is in use, so this is a no-op
# on a stock install and correct on the others.  That module was added in
# pyvista 0.48; before then the same names lived in ``pyvista.core._vtk_core``.
try:  # pyvista >= 0.48
    from pyvista._vtk import get_vtk_array_type
    from pyvista._vtk import numpy_to_vtk
    from pyvista._vtk import vtk_to_numpy
    from pyvista._vtk import vtkCellArray
    from pyvista._vtk import vtkDataArray
    from pyvista._vtk import vtkPoints
    from pyvista._vtk import vtkPointSet
    from pyvista._vtk import vtkTypeInt32Array
    from pyvista._vtk import vtkTypeInt64Array
except ModuleNotFoundError:  # pyvista < 0.48
    from pyvista.core._vtk_core import get_vtk_array_type
    from pyvista.core._vtk_core import numpy_to_vtk
    from pyvista.core._vtk_core import vtk_to_numpy
    from pyvista.core._vtk_core import vtkCellArray
    from pyvista.core._vtk_core import vtkDataArray
    from pyvista.core._vtk_core import vtkPoints
    from pyvista.core._vtk_core import vtkPointSet
    from pyvista.core._vtk_core import vtkTypeInt32Array
    from pyvista.core._vtk_core import vtkTypeInt64Array
from pyvista.core.composite import MultiBlock
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import StructuredGrid
from pyvista.core.pointset import UnstructuredGrid

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from pyvista.core.dataset import DataSet

    from pyvista_zstd._capi import CoreReader

# Highest on-disk format version this library can READ. Version 1 added the
# optional byte-shuffle pre-filter, version 2 fixed-width cell arrays, version 3
# MultiBlock field data, ExplicitStructuredGrid without celltypes, and the
# extent offset of a StructuredGrid or RectilinearGrid. The core enforces the
# ceiling; this constant only publishes it.
FILE_VERSION = 3
# Version stamped on files that use no optional feature. Such files stay
# byte-identical to the legacy format and remain readable by older releases.
FILE_VERSION_UNFILTERED = 0
FILE_VERSION_SHUFFLE = 1
FILE_VERSION_FIXED_WIDTH_CELLS = 2
# These features share version 3.
FILE_VERSION_MULTIBLOCK_FIELD_DATA = 3
FILE_VERSION_EXPLICIT_STRUCTURED_GRID = 3
FILE_VERSION_STRUCTURED_EXTENT_OFFSET = 3
FILE_VERSION_KEY = "FILE_VERSION"
DS_TYPE_KEY = "ds_type"
POINT_DATA_SUFFIX = "__point_data"
CELL_DATA_SUFFIX = "__cell_data"
# Cell arrays that rebuild an ExplicitStructuredGrid from a file with no stored dimensions.
ESGRID_BLOCK_KEYS = ("BLOCK_I", "BLOCK_J", "BLOCK_K")
FIELD_DATA_SUFFIX = "__field_data"
IMAGE_DATA_SUFFIX = "__image_data"
OFFSET_SUFFIX = "_offset"
CONNECTIVITY_SUFFIX = "_connectivity"
METADATA_KEY_COMPRESSION = "COMPRESSION"
METADATA_KEY_COMPRESSION_LVL = "COMPRESSION_LEVEL"
CELL_TYPES_KEY = "celltypes"
DS_METADATA_KEY = "__ds_metadata"
MULTIBLOCK_METADATA_KEY = "__multiblock__ds_metadata"
FILE_METADATA_KEY = "__pyvista_zstd_metadata"
LEGACY_FILE_METADATA_KEY = "__zvtk_metadata"
FILE_SUFFIX = ".pv"
LEGACY_FILE_SUFFIX = ".zvtk"
SUPPORTED_READ_SUFFIXES = (FILE_SUFFIX, LEGACY_FILE_SUFFIX)

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

# ---------------------------------------------------------------------------
# Byte-shuffle pre-filter (file_version >= 1)
# ---------------------------------------------------------------------------
# Splits an array into byte planes before compression, turning the repetitive
# sign/exponent planes of an IEEE-754 array into long runs. Opt-in; ``"auto"``
# keeps it only when a trial compression confirms it shrinks the data.
#
# The core applies the filter and records the id as an optional trailing byte
# on the array metadata frame. The id is declared here because it is part of
# the on-disk format; nothing in this module branches on it.
_FILTER_SHUFFLE = 1

ShuffleSpec = Literal["auto", True, False]

Association = Literal["point", "cell", "field"]
_ASSOCIATION_SUFFIXES: dict[str, str] = {
    "point": POINT_DATA_SUFFIX,
    "cell": CELL_DATA_SUFFIX,
    "field": FIELD_DATA_SUFFIX,
}


def _shuffle_mode(shuffle: ShuffleSpec) -> int:
    """Translate the API spelling to the core's policy enum."""
    return {
        False: _capi.SHUFFLE_NEVER,
        True: _capi.SHUFFLE_ALWAYS,
        "auto": _capi.SHUFFLE_AUTO,
    }[shuffle]


def _known_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Return ``raw`` without the keys that are not fields of dataclass ``cls``."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


@dataclass(slots=True, frozen=True)
class ArrayInfo:
    """
    Shape and dtype of one stored array, known without decompressing it.

    Attributes
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : str
        NumPy dtype name, such as ``"float32"``.

    """

    shape: tuple[int, ...]
    dtype: str


@dataclass(slots=True, frozen=True)
class ZstdFileMetadata:
    """
    File-level facts of a pyvista-zstd container.

    Returned by :attr:`pyvista_zstd.Reader.file_metadata`. The attributes are
    the supported interface; the JSON and array conversions serve the file
    format.

    Attributes
    ----------
    frame_names : list[str]
        Stored name of every array and dataset-metadata document, in file
        order; the file-metadata document itself is not listed. Each name
        starts with the 16-character identifier of its dataset, except
        ``"__thumbnail"``, which comes first in a file with a thumbnail and
        names no array.
    compression_level : int
        Zstandard level the file was written with.
    compression : str
        Compression algorithm, ``"zstandard"``.
    file_version : int
        On-disk format version.

    """

    frame_names: list[str]
    compression_level: int
    compression: Compression = "zstandard"
    # Defaults to the legacy version; the core writer promotes it when the file
    # uses an optional feature.
    file_version: int = FILE_VERSION_UNFILTERED

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> ZstdFileMetadata:
        """Create from JSON."""
        return cls(**_known_fields(cls, json.loads(s)))

    def to_array(self) -> NDArray[np.uint8]:
        """Output as a numpy uint8 array."""
        meta_bytes = self.to_json().encode("utf-8")
        return np.frombuffer(meta_bytes, dtype=np.uint8)


@dataclass
class MultiBlockMetadata:
    """
    Metadata of a stored :class:`pyvista.MultiBlock`.

    Returned by :attr:`pyvista_zstd.Reader.dataset_metadata` for a MultiBlock
    file, with :attr:`children_ds` filled in down the whole tree. The
    attributes are the supported interface; the JSON and array conversions
    serve the file format.

    Attributes
    ----------
    ds_type : str
        ``"MultiBlock"``.
    uid : str
        Identifier of this block within the file.
    children : list[str]
        Identifier of each child, in block order.
    children_keys : list[str]
        Name of each child, in block order.
    children_ds : dict[str, MultiBlockMetadata | DataSetMetadata | None] | None
        Metadata of each child keyed by identifier. ``None`` values are empty
        blocks.
    field_data_keys : dict[str, ArrayInfo]
        Shape and dtype of this MultiBlock's own field arrays.

    """

    uid: str
    children: list[str]
    ds_type = "MultiBlock"
    children_keys: list[str]

    # optional and used for ds reader
    children_ds: dict[str, MultiBlockMetadata | DataSetMetadata | None] | None = None

    # the MultiBlock's own field arrays, omitted from the JSON when empty
    field_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON."""
        raw = asdict(self)
        # Omitted when empty, since its presence stamps file version 3.
        if not raw["field_data_keys"]:
            del raw["field_data_keys"]
        return json.dumps(raw, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> MultiBlockMetadata:
        """Create from JSON."""
        raw = json.loads(s)
        raw["field_data_keys"] = {
            k: ArrayInfo(shape=tuple(v["shape"]), dtype=v["dtype"]) for k, v in raw.get("field_data_keys", {}).items()
        }
        return cls(**_known_fields(cls, raw))

    @classmethod
    def from_array(cls, arr: NDArray[np.uint8]) -> MultiBlockMetadata:
        """Create from a numpy uint8 array."""
        raw_json = arr.tobytes().decode("utf-8")  # copy, but it's tiny
        return MultiBlockMetadata.from_json(raw_json)

    def to_array(self) -> NDArray[np.uint8]:
        """Output as a numpy uint8 array."""
        meta_bytes = self.to_json().encode("utf-8")
        return np.frombuffer(meta_bytes, dtype=np.uint8)


# Files written before 0.6 store the points dtype as the repr of its NumPy type.
_LEGACY_POINTS_DTYPES = {"<class 'numpy.float32'>": "float32", "<class 'numpy.float64'>": "float64"}


@dataclass(slots=True, frozen=True)
class DataSetMetadata:
    """
    Metadata of one stored dataset, known without decompressing its arrays.

    Returned by :attr:`pyvista_zstd.Reader.dataset_metadata`. The attributes
    are the supported interface; the JSON and array conversions serve the
    file format.

    Attributes
    ----------
    ds_type : str
        Dataset class name, such as ``"UnstructuredGrid"``.
    uid : str
        Identifier of this dataset within the file.
    n_points, n_cells : int
        Number of points and cells.
    points_dtype, celltypes_dtype : str | None
        Dtype of the points and cell types, when the dataset stores them.
    point_data_keys, cell_data_keys, field_data_keys : dict[str, ArrayInfo]
        Shape and dtype of each point, cell and field array, by name.
    fixed_cell_sizes : dict[str, int]
        Common cell width of each fixed-width cell array.
    point_data_active_scalars_name, point_data_active_vectors_name : str | None
        Active point scalars and vectors.
    point_data_active_texture_coordinates_name, point_data_active_normals_name : str | None
        Active point texture coordinates and normals.
    cell_data_active_scalars_name, cell_data_active_vectors_name : str | None
        Active cell scalars and vectors.
    cell_data_active_texture_coordinates_name, cell_data_active_normals_name : str | None
        Active cell texture coordinates and normals.
    dimensions : tuple[int, int, int] | None
        Dimensions of a structured dataset.
    origin, spacing : tuple[float, float, float] | None
        Origin and spacing of an :class:`pyvista.ImageData`.
    direction_matrix : list[list[float]] | None
        Direction matrix of an :class:`pyvista.ImageData`.
    offset : tuple[int, int, int] | None
        Extent offset of a structured dataset.
    faces_connectivity_flags_array_name : str | None
        Connectivity flags array name of an :class:`pyvista.ExplicitStructuredGrid`.

    """

    ds_type: str
    uid: str
    n_points: int
    points_dtype: str | None
    n_cells: int
    celltypes_dtype: str | None
    point_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)
    cell_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)
    field_data_keys: dict[str, ArrayInfo] = field(default_factory=dict)
    fixed_cell_sizes: dict[str, int] = field(default_factory=dict)
    point_data_active_scalars_name: str | None = None
    point_data_active_vectors_name: str | None = None
    point_data_active_texture_coordinates_name: str | None = None
    point_data_active_normals_name: str | None = None
    cell_data_active_scalars_name: str | None = None
    cell_data_active_vectors_name: str | None = None
    cell_data_active_texture_coordinates_name: str | None = None
    cell_data_active_normals_name: str | None = None

    # Optional grid metadata; dimensions and offset describe every structured dataset
    dimensions: tuple[int, int, int] | None = None
    origin: tuple[float, float, float] | None = None
    spacing: tuple[float, float, float] | None = None
    direction_matrix: list[list[float]] | None = None
    offset: tuple[int, int, int] | None = None
    faces_connectivity_flags_array_name: str | None = None

    @classmethod
    def from_dataset(  # noqa: C901, PLR0912, PLR0913
        cls,
        ds: pv.DataSet,
        point_info: dict[str, ArrayInfo],
        cell_info: dict[str, ArrayInfo],
        field_info: dict[str, ArrayInfo],
        fixed_cell_sizes: dict[str, int],
        *,
        uid: str,
        point_active: tuple[str | None, str | None, str | None, str | None] | None = None,
        cell_active: tuple[str | None, str | None, str | None, str | None] | None = None,
        celltypes_dtype: str | None = None,
    ) -> DataSetMetadata:
        """
        Create metadata from a dataset.

        *point_active* and *cell_active* name the active scalars, vectors,
        normals and texture coordinates; left out, they are read from *ds*.
        """
        # Many pyvista calls require intermediate object assembly, side step or
        # do once when possible.

        # Get points dtype only for datasets that store explicit points.
        # ``vtkImageData`` has no ``GetPoints`` in older VTK and
        # ``vtkRectilinearGrid.GetPoints()`` requires an out-arg in older VTK,
        # so probe via ``vtkPointSet`` membership instead.
        if isinstance(ds, vtkPointSet) and ds.GetPoints() is not None:
            vtk_dtype = ds.GetPoints().GetDataType()
            if vtk_dtype == VTK_FLOAT:
                points_dtype: type[np.floating] | None = np.float32
            elif vtk_dtype == VTK_DOUBLE:
                points_dtype = np.float64
            else:  # pragma: no cover
                msg = "Invalid points datatype. Should be float or double"
                raise RuntimeError(msg)
        else:
            points_dtype = None

        if point_active is None:
            pd = ds.point_data
            point_active = (
                pd.active_scalars_name,
                pd.active_vectors_name,
                pd.active_normals_name,
                pd.active_texture_coordinates_name,
            )
        if cell_active is None:
            cd = ds.cell_data
            cell_active = (
                cd.active_scalars_name,
                cd.active_vectors_name,
                cd.active_normals_name,
                cd.active_texture_coordinates_name,
            )
        if celltypes_dtype is None and isinstance(ds, UnstructuredGrid):
            celltypes_dtype = str(ds.celltypes.dtype)
        kwargs: dict[str, Any] = {
            "ds_type": type(ds).__name__,
            "uid": uid,
            "n_points": ds.GetNumberOfPoints(),
            "points_dtype": np.dtype(points_dtype).name if points_dtype is not None else None,
            "n_cells": ds.GetNumberOfCells(),
            "celltypes_dtype": celltypes_dtype,
            "point_data_keys": point_info,
            "cell_data_keys": cell_info,
            "field_data_keys": field_info,
            "fixed_cell_sizes": fixed_cell_sizes,
            "point_data_active_scalars_name": point_active[_ACTIVE_SCALARS],
            "point_data_active_vectors_name": point_active[_ACTIVE_VECTORS],
            "point_data_active_texture_coordinates_name": point_active[_ACTIVE_TCOORDS],
            "point_data_active_normals_name": point_active[_ACTIVE_NORMALS],
            "cell_data_active_scalars_name": cell_active[_ACTIVE_SCALARS],
            "cell_data_active_vectors_name": cell_active[_ACTIVE_VECTORS],
            "cell_data_active_texture_coordinates_name": cell_active[_ACTIVE_TCOORDS],
            "cell_data_active_normals_name": cell_active[_ACTIVE_NORMALS],
        }

        # Only structured datasets have dimensions; older pyvista lacks offset on some, so read the extent.
        if hasattr(ds, "dimensions"):
            kwargs["dimensions"] = ds.dimensions
            kwargs["offset"] = tuple(ds.offset) if hasattr(ds, "offset") else ds.GetExtent()[::2]
        if isinstance(ds, pv.ImageData):
            kwargs.update(
                origin=ds.origin,
                spacing=ds.spacing,
                direction_matrix=ds.direction_matrix.tolist(),
            )
        elif isinstance(ds, (pv.StructuredGrid, ExplicitStructuredGrid)):
            if not ds.n_points:
                # Store either grid without points as empty; a default ExplicitStructuredGrid's extent is uninitialised.
                kwargs.update(dimensions=(0, 0, 0), offset=(0, 0, 0), n_cells=0)
            if isinstance(ds, ExplicitStructuredGrid):
                kwargs["faces_connectivity_flags_array_name"] = ds.GetFacesConnectivityFlagsArrayName()

        return cls(**kwargs)

    def to_json(self) -> str:
        """Convert to JSON, leaving out an unset ExplicitStructuredGrid flags array name."""
        fields = asdict(self)
        # pyvista-zstd 0.5.0 rejects metadata keys it does not know.
        if fields["faces_connectivity_flags_array_name"] is None:
            del fields["faces_connectivity_flags_array_name"]
        return json.dumps(fields, separators=(",", ":"))

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
            return {k: ArrayInfo(shape=tuple(v["shape"]), dtype=v["dtype"]) for k, v in m.items()}

        raw["point_data_keys"] = decode_mapping(raw.get("point_data_keys", {}))
        raw["cell_data_keys"] = decode_mapping(raw.get("cell_data_keys", {}))
        raw["field_data_keys"] = decode_mapping(raw.get("field_data_keys", {}))
        raw["fixed_cell_sizes"] = raw.get("fixed_cell_sizes", {})
        raw["points_dtype"] = _LEGACY_POINTS_DTYPES.get(raw.get("points_dtype"), raw.get("points_dtype"))
        for key in ("dimensions", "origin", "spacing", "offset"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**_known_fields(cls, raw))

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


def _narrowed_to_int32(values: NDArray[Any], *, upper_bound: int | None = None) -> NDArray[Any]:
    """
    Return *values* as int32 when every entry survives the cast, else unchanged.

    *upper_bound* must come from what the entries mean -- connectivity is
    bounded by the point count, an offsets array by the length of the
    connectivity it indexes -- never from the array's length. ``astype`` wraps
    an out-of-range id silently, so an under-estimate corrupts topology.

    Omit it when the meaning is not certain; the array is scanned instead.
    Erring high only declines to narrow, which is correct if larger.

    Ids are non-negative by construction, so only the upper end is checked.
    """
    if upper_bound is None:
        upper_bound = int(values.max()) if values.size else 0
    if upper_bound > np.iinfo(np.int32).max:
        return values
    return values.astype(np.int32, copy=False)


def _add_cell_array(  # noqa: PLR0913
    ds_id: str,
    arrays: dict[str, np.ndarray],
    name: str,
    cell_array: vtkCellArray,
    fixed_cell_sizes: dict[str, int],
    *,
    force_int32: bool = False,
    max_entry: int | None = None,
    skip_empty: bool = False,
) -> None:
    """
    Stage a cell array's connectivity, and its offsets when the cells vary in size.

    *skip_empty* leaves an empty array out of the file; only PolyData readers
    treat a missing cell frame as empty.

    *max_entry* is the largest value the connectivity can hold, where the caller
    knows it: entries are ids into some other array, so that array's extent
    bounds them without looking. Leave it None to have the values scanned. It
    only ever governs the int32 narrowing -- see :func:`_narrowed_to_int32`.
    """
    # A vtkCellArray is truthy even when empty, so the count is what decides.
    if cell_array is None or (skip_empty and cell_array.GetNumberOfCells() == 0):
        return

    connectivity = vtk_to_numpy(cell_array.GetConnectivityArray())
    cell_size = cell_array.IsHomogeneous()

    # compress to int32 whenever possible
    if force_int32:
        connectivity = _narrowed_to_int32(connectivity, upper_bound=max_entry)

    if cell_size > 0:
        fixed_cell_sizes[name] = cell_size
    else:
        offsets = vtk_to_numpy(cell_array.GetOffsetsArray())
        if force_int32:
            # An offset indexes into the connectivity, so its length is the bound.
            offsets = _narrowed_to_int32(offsets, upper_bound=connectivity.size)
        arrays[f"{ds_id}{name}{OFFSET_SUFFIX}"] = offsets
    arrays[f"{ds_id}{name}{CONNECTIVITY_SUFFIX}"] = connectivity


def _extract_cell_array(
    ds_id: str,
    name: str,
    segments: dict[str, Any],
    fixed_cell_sizes: dict[str, int],
) -> vtkCellArray | None:
    conn_key = f"{ds_id}{name}{CONNECTIVITY_SUFFIX}"
    if conn_key not in segments:
        return None

    offset_key = f"{ds_id}{name}{OFFSET_SUFFIX}"
    connectivity = _owned(segments, conn_key)
    if name in fixed_cell_sizes:
        if connectivity is not None:
            return _vtk_cells(None, connectivity, segments[conn_key].size, cell_size=fixed_cell_sizes[name])
        return _numpy_to_vtk_cells(None, segments[conn_key], cell_size=fixed_cell_sizes[name])
    if offset_key not in segments:
        msg = f"Cell array '{name}' has neither offsets nor a fixed cell size."
        raise ValueError(msg)
    offset = _owned(segments, offset_key)
    if connectivity is not None and offset is not None and offset.GetDataType() == connectivity.GetDataType():
        return _vtk_cells(offset, connectivity, segments[conn_key].size)
    return _numpy_to_vtk_cells(segments[offset_key], segments[conn_key])


def _vtk_cells(
    offset: vtkDataArray | None,
    connectivity: vtkDataArray,
    n_connectivity: int,
    *,
    cell_size: int | None = None,
) -> vtkCellArray:
    """Build a cell array around VTK-owned offsets and connectivity without copying."""
    carr = vtkCellArray()
    if cell_size is not None:
        if cell_size <= 0 or n_connectivity % cell_size:
            msg = f"Invalid fixed cell size {cell_size} for connectivity length {n_connectivity}."
            raise ValueError(msg)
        carr.SetData(cell_size, connectivity)
    else:
        carr.SetData(offset, connectivity)
    return carr


def _points_of(ds: DataSet) -> NDArray[Any]:
    """Return the dataset's points as the NumPy view of the VTK array, (0, 3) when it has none."""
    # Optimization: ``ds.points`` wraps a pyvista_ndarray per call; the raw view is enough to stage.
    vtk_points = ds.GetPoints()
    if vtk_points is None:
        return np.empty((0, 3), dtype=np.float64)
    return vtk_to_numpy(vtk_points.GetData())


def _add_arrays_pointset(ds_id: str, ds: PointSet, arrays: dict[str, NDArray[Any]]) -> None:
    arrays[f"{ds_id}{POINTS_KEY}"] = _points_of(ds)


def _add_arrays_rgrid(ds_id: str, ds: RectilinearGrid, arrays: dict[str, NDArray[Any]]) -> None:
    if ds.n_points:
        arrays[f"{ds_id}{RGRID_X_SUFFIX}"] = ds.x
        arrays[f"{ds_id}{RGRID_Y_SUFFIX}"] = ds.y
        arrays[f"{ds_id}{RGRID_Z_SUFFIX}"] = ds.z


def _add_arrays_polydata(
    ds_id: str,
    ds: PolyData,
    arrays: dict[str, NDArray[Any]],
    fixed_cell_sizes: dict[str, int],
    *,
    force_int32: bool = True,
) -> None:
    arrays[f"{ds_id}{POINTS_KEY}"] = _points_of(ds)
    # Every one of these holds point ids, so the point count bounds them all.
    max_point_id = ds.GetNumberOfPoints() - 1
    for name, cells in (
        (POLYS, ds.GetPolys()),
        (LINES, ds.GetLines()),
        (STRIPS, ds.GetStrips()),
        (VERTS, ds.GetVerts()),
    ):
        _add_cell_array(
            ds_id,
            arrays,
            name,
            cells,
            fixed_cell_sizes,
            force_int32=force_int32,
            max_entry=max_point_id,
            skip_empty=True,
        )


def _add_arrays_ugrid(
    ds_id: str,
    ds: UnstructuredGrid,
    arrays: dict[str, NDArray[Any]],
    fixed_cell_sizes: dict[str, int],
    *,
    force_int32: bool = True,
) -> None:
    arrays[f"{ds_id}{POINTS_KEY}"] = _points_of(ds)
    celltypes = ds.celltypes
    arrays[f"{ds_id}{CELL_TYPES_KEY}"] = celltypes

    max_point_id = ds.GetNumberOfPoints() - 1
    _add_cell_array(
        ds_id,
        arrays,
        CELLS,
        ds.GetCells(),
        fixed_cell_sizes,
        force_int32=force_int32,
        max_entry=max_point_id,
    )

    has_polyhedra = bool(np.any(celltypes == pv.CellType.POLYHEDRON))
    if has_polyhedra:
        if pv.vtk_version_info < (9, 4):
            msg = (
                "Polyhedron round-trip requires VTK >= 9.4. "
                f"Detected polyhedra in dataset but VTK {pv.vtk_version_info} is installed. "
                "Upgrade VTK to encode polyhedra."
            )
            raise NotImplementedError(msg)
        _add_cell_array(
            ds_id,
            arrays,
            POLYHEDRON,
            ds.GetPolyhedronFaces(),
            fixed_cell_sizes,
            force_int32=force_int32,
            max_entry=max_point_id,  # a face is a list of point ids
        )
        # Face locations index into the face stream, not the points, and that
        # extent is not to hand here -- so it is scanned, not guessed.
        _add_cell_array(
            ds_id,
            arrays,
            POLYHEDRON_LOCATION,
            ds.GetPolyhedronFaceLocations(),
            fixed_cell_sizes,
            force_int32=force_int32,
        )


def _add_arrays_esgrid(
    ds_id: str,
    ds: ExplicitStructuredGrid,
    arrays: dict[str, NDArray[Any]],
    fixed_cell_sizes: dict[str, int],
    *,
    force_int32: bool = True,
) -> None:
    arrays[f"{ds_id}{POINTS_KEY}"] = ds.points
    _add_cell_array(
        ds_id,
        arrays,
        CELLS,
        # A grid without points is stored empty, so drop the cells its extent allocated.
        ds.GetCells() if ds.n_points else vtkCellArray(),
        fixed_cell_sizes,
        force_int32=force_int32,
        max_entry=ds.n_points - 1,
    )


def _add_arrays_sgrid(ds_id: str, ds: StructuredGrid, arrays: dict[str, NDArray[Any]]) -> None:
    arrays[f"{ds_id}{POINTS_KEY}"] = _points_of(ds)


# eventually add: compression: Compression = "zstandard",
def write(  # noqa: PLR0913
    ds: DataSet,
    filename: Path | str,
    *,
    progress_bar: bool = False,
    force_int32: bool = True,
    level: int = 3,
    n_threads: int | None = None,
    shuffle: ShuffleSpec = False,
    thumbnail: bytes | bool | None = None,
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

    All file types should end in ``.pv``, borrowing both from the legacy
    VTK extension ``.vtk`` and the ``.zst`` file types.

    Parameters
    ----------
    ds : pyvista.DataSet
        Dataset to compress. All PyVista dataset types except for
        :class:`pyvista.MultiBlock` are supported.
    filename : pathlib.Path | str
        Path to the file.
    force_int32 : bool, default: True
        Write cell topology as int32 whenever possible. Only applies to
        :class:`pyvista.PolyData` and
        :class:`pyvista.UnstructuredGrid`. A mesh whose point ids do not fit
        in int32 keeps its wider topology; the request is a preference, not a
        promise to narrow.
    progress_bar : bool, default: False
        Show a progress bar while writing to disk.
    level : int, default: 3
        Compression level. Valid values are all negative integers through
        22. Lower values generally yield faster operations with lower
        compression ratios. Higher values are generally slower but compress
        better.
    n_threads : int, optional
        Number of threads to use when compressing. A value of ``-1`` uses all
        available cores and ``0`` disables multi-threading.
    shuffle : {"auto", True, False}, default: False
        Optionally apply a reversible byte-shuffle pre-filter before
        compression. The filter splits each array into byte planes, which can
        let zstd compress smooth floating-point data somewhat better. Disabled
        by default. ``"auto"`` shuffles a multibyte floating-point array only
        when a quick trial compression shows it shrinks the data (so it never
        enlarges a file); ``True`` shuffles every multibyte array. Files that
        use the filter are written at format version 1 or higher and can only
        be read by this release or newer; unfiltered files (the default) stay
        backward compatible unless they use a feature listed in the Notes.
    thumbnail : bytes | bool, optional
        Embed a PNG preview image that viewers can show without reading the
        dataset. Pass the PNG bytes, or ``True`` to render a 512 x 512 image
        off-screen, cropped to the dataset on a transparent background. Rendering
        needs a working VTK render window; on a machine without a display VTK
        may crash the interpreter, so pass PNG bytes there instead.
        Files with a thumbnail remain readable by earlier releases, which
        ignore it.

    Notes
    -----
    Cell arrays whose cells all contain the same number of points are stored
    without their redundant offsets array. Their common width is recorded in
    dataset metadata and VTK reconstructs the offsets when the file is read.
    Files using this encoding are written at format version 2 or higher.

    Files containing a :class:`pyvista.ExplicitStructuredGrid`,
    :class:`pyvista.MultiBlock` field data, or a :class:`pyvista.StructuredGrid`
    or :class:`pyvista.RectilinearGrid` whose extent does not start at 0 are
    written at format version 3.

    """
    writer = Writer(ds, filename)

    # if compression == "zstandard":
    writer.write(
        progress_bar=progress_bar,
        force_int32=force_int32,
        level=level,
        n_threads=n_threads,
        shuffle=shuffle,
        thumbnail=thumbnail,
    )


class Writer:
    """Class to write a pyvista-zstd file."""

    def __init__(self, ds: DataSet, filename: Path | str) -> None:
        """Initialize the writer."""
        self._filename = Path(filename)

        if self._filename.suffix not in SUPPORTED_READ_SUFFIXES:
            msg = f"Filename must end in '{FILE_SUFFIX}', not '{self._filename.suffix}'"
            raise ValueError(msg)

        self._arrays: dict[str, NDArray[Any]] = {}
        self._ds = pv.wrap(ds)
        self._uses_fixed_width_cells = False

        # Sequential UIDs keyed by object identity: a dataset shared between
        # blocks is stored once, and the payload carries no process state.
        self._ids: dict[int, str] = {}
        self._refs: list[DataSet | MultiBlock] = []

    def _make_ds_id(self, ds: DataSet | MultiBlock) -> str:
        key = id(ds)
        if key not in self._ids:
            self._refs.append(ds)
            self._ids[key] = f"{len(self._ids):0{UID_N_CHAR}x}"
        return self._ids[key]

    def _add_ds_arrays(self, ds: DataSet, *, force_int32: bool) -> None:  # noqa: C901, PLR0912
        """Extract dataset data as arrays."""
        ds_id = self._make_ds_id(ds)
        fixed_cell_sizes: dict[str, int] = {}

        if isinstance(ds, PolyData):
            _add_arrays_polydata(ds_id, ds, self._arrays, fixed_cell_sizes, force_int32=force_int32)
        elif isinstance(ds, UnstructuredGrid):
            _add_arrays_ugrid(ds_id, ds, self._arrays, fixed_cell_sizes, force_int32=force_int32)
        elif isinstance(ds, ExplicitStructuredGrid):
            _add_arrays_esgrid(ds_id, ds, self._arrays, fixed_cell_sizes, force_int32=force_int32)
        elif isinstance(ds, ImageData):
            pass
        elif isinstance(ds, StructuredGrid):
            _add_arrays_sgrid(ds_id, ds, self._arrays)
        elif isinstance(ds, PointSet):
            _add_arrays_pointset(ds_id, ds, self._arrays)
        elif isinstance(ds, RectilinearGrid):
            _add_arrays_rgrid(ds_id, ds, self._arrays)
        elif isinstance(ds, MultiBlock):
            # placeholder, array insertion order matters
            self._arrays[f"{ds_id}{MULTIBLOCK_METADATA_KEY}"] = None

            child_ids = []
            for ds_child in ds:
                # special handling none edge case
                if ds_child is None:
                    child_ids.append(EMPTY_DS)
                else:
                    child_ids.append(self._make_ds_id(ds_child))
                    self._add_ds_arrays(ds_child, force_int32=force_int32)

            # edge case where multiblock can contain a NoneType key
            children_keys = ["None" if key is None else key for key in ds.keys()]  # noqa: SIM118
            multi_meta = MultiBlockMetadata(
                uid=ds_id,
                children=child_ids,
                children_keys=children_keys,
                field_data_keys=self._add_field_arrays(ds_id, ds),
            )
            self._arrays[f"{ds_id}{MULTIBLOCK_METADATA_KEY}"] = multi_meta.to_array()

            return
        else:  # pragma: no cover
            msg = f"Unsupported type {type(ds)}"
            raise TypeError(msg)

        point_info, point_active = _stage_attributes(ds_id, ds, "point_data", POINT_DATA_SUFFIX, self._arrays)
        # A structured grid without points is stored with no cells, so it keeps no cell data either.
        stored_empty = isinstance(ds, (StructuredGrid, ExplicitStructuredGrid)) and not ds.GetNumberOfPoints()
        if stored_empty:
            cell_info: dict[str, ArrayInfo] = {}
            cell_active: tuple[str | None, str | None, str | None, str | None] = (None, None, None, None)
        else:
            cell_info, cell_active = _stage_attributes(ds_id, ds, "cell_data", CELL_DATA_SUFFIX, self._arrays)
        field_info = self._add_field_arrays(ds_id, ds)

        # supply dataset metadata
        ds_meta = DataSetMetadata.from_dataset(
            ds,
            point_info,
            cell_info,
            field_info,
            fixed_cell_sizes,
            uid=ds_id,
            point_active=point_active,
            cell_active=cell_active,
            celltypes_dtype=str(self._arrays[f"{ds_id}{CELL_TYPES_KEY}"].dtype)
            if f"{ds_id}{CELL_TYPES_KEY}" in self._arrays
            else None,
        )
        self._arrays[f"{ds_id}{DS_METADATA_KEY}"] = ds_meta.to_array()
        self._uses_fixed_width_cells = self._uses_fixed_width_cells or bool(fixed_cell_sizes)

    def _add_field_arrays(self, ds_id: str, ds: DataSet | MultiBlock) -> dict[str, ArrayInfo]:
        """Stage the field arrays of a dataset or MultiBlock and return their metadata."""
        field_info, _ = _stage_attributes(ds_id, ds, "field_data", FIELD_DATA_SUFFIX, self._arrays)
        return field_info

    def write(  # noqa: PLR0913
        self,
        *,
        progress_bar: bool = False,
        force_int32: bool = True,
        level: int = 3,
        n_threads: int | None = None,
        shuffle: ShuffleSpec = False,
        thumbnail: bytes | bool | None = None,
    ) -> None:
        """Write the dataset."""
        if self._filename.suffix == LEGACY_FILE_SUFFIX:
            # FutureWarning (not DeprecationWarning) because this is aimed at
            # end users writing data files, and Python's default warning
            # filters hide DeprecationWarning from non-__main__ code.
            warnings.warn(
                f"Writing '{self._filename}' as a legacy zvtk file. Support "
                f"for the '{LEGACY_FILE_SUFFIX}' format will be removed in a "
                f"future release; prefer the '{FILE_SUFFIX}' extension.",
                FutureWarning,
                stacklevel=2,
            )

        if progress_bar:
            warnings.warn(
                "`progress_bar` no longer has any effect: frames are written by the C++ core in a single call.",
                DeprecationWarning,
                stacklevel=2,
            )

        png = _thumbnail_png(self._ds, thumbnail=thumbnail)
        self._add_ds_arrays(self._ds, force_int32=force_int32)

        # The core owns the format decisions; this side stages arrays in
        # frame order.
        with _capi.CoreWriter() as writer:
            writer.set_level(level)
            writer.set_threads(_capi.THREADS_AUTO if n_threads is None else n_threads)
            writer.set_shuffle(_shuffle_mode(shuffle))
            writer.set_fixed_width_cells(enabled=self._uses_fixed_width_cells)
            if png is not None:
                writer.set_thumbnail(png)
            for name, arr in self._arrays.items():
                writer.add_array(name, arr)
            writer.write(self._filename)

        # no need to hold onto any references as all IDs have been written
        self._refs = []
        self._ids = {}


class _Segments(dict[str, Any]):
    """Decompressed frames by name, plus the VTK arrays that own the ones decompressed in place."""

    def __init__(self) -> None:
        super().__init__()
        self.vtk: dict[str, vtkDataArray] = {}

    def allocate(self, name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> NDArray[Any]:
        """Return the destination for frame *name*, VTK-owned where possible."""
        vtk_type = _vtk_type_for(name, shape, dtype)
        if vtk_type is None:
            return np.empty(shape, dtype=dtype)
        # Optimization: decompress straight into a VTK-owned array so attaching it later is a pointer
        # hand-off instead of numpy_to_vtk's wrapper-per-array cost.
        vtk_array = vtkDataArray.CreateDataArray(vtk_type)
        vtk_array.SetNumberOfComponents(shape[1] if len(shape) == 2 else 1)  # noqa: PLR2004
        vtk_array.SetNumberOfTuples(shape[0])
        self.vtk[name] = vtk_array
        if not vtk_array.GetNumberOfValues():
            return np.empty(shape, dtype=dtype)
        return vtk_to_numpy(vtk_array).reshape(shape)

    def release(self) -> None:
        """Forget the VTK wrappers once a dataset owns the arrays."""
        # Optimization: before VTK 9.8 (d1489916c5), dropping a wrapper that carries state scans every ghost.
        if pv.vtk_version_info < (9, 8):
            for vtk_array in self.vtk.values():
                vtk_array.__dict__.clear()
        self.vtk.clear()


def _vtk_type_for(name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> int | None:
    """Return the VTK type to decompress frame *name* into, or None to keep it in NumPy."""
    if dtype.kind not in "iuf" or not dtype.isnative or len(shape) not in (1, 2) or 0 in shape[1:]:
        return None
    if name.endswith((CONNECTIVITY_SUFFIX, OFFSET_SUFFIX)):
        # vtkCellArray stores only these two types without converting.
        if dtype == np.int32:
            return _VTK_INT32
        if dtype == np.int64:
            return _VTK_INT64
        return None
    try:
        return int(get_vtk_array_type(dtype))
    except TypeError:
        return None


_VTK_INT32 = vtkTypeInt32Array().GetDataType()
_VTK_INT64 = vtkTypeInt64Array().GetDataType()


def _owned(segments: dict[str, Any], key: str, vtk_type: int | None = None) -> vtkDataArray | None:
    """Return the VTK array that owns frame *key*, when it has one of *vtk_type*."""
    if not isinstance(segments, _Segments):
        return None
    vtk_array = segments.vtk.get(key)
    if vtk_array is None or (vtk_type is not None and vtk_array.GetDataType() != vtk_type):
        return None
    return vtk_array


def _set_points(ds: DataSet, ds_id: str, segments: dict[str, Any]) -> None:
    """Give *ds* its points, without a copy when VTK owns them."""
    key = f"{ds_id}{POINTS_KEY}"
    vtk_array = _owned(segments, key)
    points = segments[key]
    if vtk_array is None or points.ndim != 2 or points.shape[1] != 3:  # noqa: PLR2004
        ds.points = points
        return
    vtk_points = vtkPoints()
    vtk_points.SetData(vtk_array)
    ds.SetPoints(vtk_points)


_ATTRIBUTE_GETTERS = {"point_data": "GetPointData", "cell_data": "GetCellData", "field_data": "GetFieldData"}
# Keys of pyvista's per-association bool/complex name sets (FieldAssociation.name).
_ASSOCIATION_KEYS = {"point_data": "POINT", "cell_data": "CELL", "field_data": "NONE"}
# vtkDataSetAttributes attribute slots, in the order GetAttributeIndices fills.
_ACTIVE_SCALARS, _ACTIVE_VECTORS, _ACTIVE_NORMALS, _ACTIVE_TCOORDS = 0, 1, 2, 3


def _stage_attributes(
    ds_id: str,
    ds: DataSet | MultiBlock,
    attribute: str,
    suffix: str,
    arrays: dict[str, NDArray[Any]],
) -> tuple[dict[str, ArrayInfo], tuple[str | None, str | None, str | None, str | None]]:
    """
    Stage one attribute set's arrays and return their info plus the active names.

    Each VTK array is fetched once; arrays pyvista converts for the caller
    (bool, complex, string) go through its getter so they stage as the caller
    sees them.
    """
    # Optimization: walk the VTK arrays by index. ``ds.point_data.items()`` resolves every name through
    # ``GetAbstractArray(name)``, an O(n) scan each, so one call is quadratic in the array count.
    field_data = getattr(ds, _ATTRIBUTE_GETTERS[attribute])()
    association = _ASSOCIATION_KEYS[attribute]
    special: set[str] = set()
    for tracked in ("_association_bitarray_names", "_association_complex_names"):
        special |= getattr(ds, tracked, {}).get(association, set())

    names: list[str] = []
    info: dict[str, ArrayInfo] = {}
    for index in range(field_data.GetNumberOfArrays()):
        # Nameless arrays take the name pyvista gives them in ``keys()``.
        name = field_data.GetArrayName(index) or f"Unnamed_{index}"
        names.append(name)
        vtk_array = field_data.GetAbstractArray(index)
        if name in special or not isinstance(vtk_array, vtkDataArray):
            # pyvista < 0.50 can return a one-value string field, such as user_dict, as ``str``.
            array = np.atleast_1d(getattr(ds, attribute)[name])
        else:
            array = vtk_to_numpy(vtk_array)
        arrays[f"{ds_id}{name}{suffix}"] = array
        info[name] = ArrayInfo(shape=array.shape, dtype=str(array.dtype))

    if attribute == "field_data":
        return info, (None, None, None, None)
    slots = [0] * field_data.NUM_ATTRIBUTES
    field_data.GetAttributeIndices(slots)
    active = tuple(names[slots[i]] if slots[i] >= 0 else None for i in range(_ACTIVE_TCOORDS + 1))
    return info, active  # type: ignore[return-value]


_DATA_SUFFIXES = (
    (POINT_DATA_SUFFIX, "point_data", "GetPointData"),
    (CELL_DATA_SUFFIX, "cell_data", "GetCellData"),
    (FIELD_DATA_SUFFIX, "field_data", "GetFieldData"),
)


def _set_attribute_array(data: pv.DataSetAttributes, array: NDArray[Any], name: str, *, field: bool) -> None:
    """Add *array* to *data* through pyvista, keeping a field array's empty tuples."""
    if field and array.ndim > 1 and array.shape[0] == 0 and math.prod(array.shape[1:]) > 0:
        # VTK supports empty tuples; PyVista only admits their scalar view.
        # Let PyVista retain its bool/complex conventions before restoring
        # the component count on the same zero-tuple VTK array.
        data.set_array(array.reshape(-1), name)
        components = math.prod(array.shape[1:]) * (2 if np.iscomplexobj(array) else 1)
        data.VTKObject.GetAbstractArray(name).SetNumberOfComponents(components)
    else:
        data.set_array(array, name)


def _check_tuple_count(ds: DataSet, vtk_getter: str, vtk_array: vtkDataArray, name: str) -> None:
    """Raise ValueError unless a point or cell array has one tuple per point or cell."""
    if vtk_getter == "GetPointData":
        expected, kind = ds.GetNumberOfPoints(), "point"
    elif vtk_getter == "GetCellData":
        expected, kind = ds.GetNumberOfCells(), "cell"
    else:
        return
    if vtk_array.GetNumberOfTuples() != expected:
        msg = (
            f"{kind.capitalize()} array {name!r} has {vtk_array.GetNumberOfTuples()} tuples but the dataset "
            f"has {expected} {kind}s; the file is corrupt or was written from an inconsistent dataset."
        )
        raise ValueError(msg)


def _add_data(ds_id: str, ds: DataSet, segment_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach the arrays of *ds_id* to *ds* and return the VTK attribute sets fetched on the way."""
    # Optimization: attribute sets are fetched once per dataset (on VTK 9.7 every
    # dropped wrapper costs a scan of all live ones) and owned arrays go in through
    # raw ``AddArray``, which skips pyvista's wrapping; their length is checked here instead.
    attributes: dict[str, Any] = {}
    for key, array in segment_dict.items():
        if not key.startswith(ds_id):
            continue
        for suffix, pyvista_name, vtk_getter in _DATA_SUFFIXES:
            if not key.endswith(suffix):
                continue
            name = key[UID_N_CHAR : -len(suffix)]
            vtk_array = _owned(segment_dict, key)
            if vtk_array is None:
                _set_attribute_array(getattr(ds, pyvista_name), array, name, field=suffix == FIELD_DATA_SUFFIX)
            else:
                _check_tuple_count(ds, vtk_getter, vtk_array, name)
                target = attributes.get(vtk_getter)
                if target is None:
                    target = attributes[vtk_getter] = getattr(ds, vtk_getter)()
                vtk_array.SetName(name)
                target.AddArray(vtk_array)
    return attributes


def _segments_to_ugrid(
    ds_id: str,
    segments: dict[str, Any],
    metadata: DataSetMetadata,
) -> UnstructuredGrid:
    # A grid without cells has no cells frame; SetCells needs an array regardless.
    cells = _extract_cell_array(ds_id, CELLS, segments, metadata.fixed_cell_sizes) or vtkCellArray()

    celltypes_key = f"{ds_id}{CELL_TYPES_KEY}"
    celltypes_vtk = _owned(segments, celltypes_key, VTK_UNSIGNED_CHAR)
    if celltypes_vtk is None:
        celltypes_vtk = numpy_to_vtk(segments[celltypes_key], deep=False, array_type=VTK_UNSIGNED_CHAR)

    ugrid = UnstructuredGrid()
    _set_points(ugrid, ds_id, segments)

    poly = _extract_cell_array(ds_id, POLYHEDRON, segments, metadata.fixed_cell_sizes)
    poly_loc = _extract_cell_array(ds_id, POLYHEDRON_LOCATION, segments, metadata.fixed_cell_sizes)

    if poly and poly_loc:
        if pv.vtk_version_info < (9, 4):
            msg = (
                "Polyhedron decode requires VTK >= 9.4. "
                f"File contains polyhedra but VTK {pv.vtk_version_info} is installed. "
                "Upgrade VTK to load this file."
            )
            raise NotImplementedError(msg)
        ugrid.SetPolyhedralCells(
            celltypes_vtk,
            cells,
            poly_loc,
            poly,
        )
    else:
        ugrid.SetCells(celltypes_vtk, cells)

    return ugrid


def _segments_to_esgrid(
    ds_id: str,
    segments: dict[str, Any],
    metadata: DataSetMetadata,
) -> ExplicitStructuredGrid:
    """Rebuild an explicit structured grid from its dimensions, points and cells."""
    if metadata.dimensions is None:
        return _segments_to_esgrid_by_blocking(ds_id, segments, metadata)

    points = segments[f"{ds_id}{POINTS_KEY}"]
    cells = _extract_cell_array(ds_id, CELLS, segments, metadata.fixed_cell_sizes)
    n_cells = 0 if cells is None else cells.GetNumberOfCells()
    dimensions = metadata.dimensions
    # Explicit points are arbitrary; the dimensions describe the cell lattice.
    # Check before SetExtent allocates a cell for every i/j/k position.
    valid_dimensions = dimensions == (0, 0, 0) or (
        len(dimensions) == 3 and all(dim > 0 for dim in dimensions)  # noqa: PLR2004
    )
    if not valid_dimensions or math.prod(max(dim - 1, 0) for dim in dimensions) != n_cells:
        msg = (
            f"ExplicitStructuredGrid dimensions {tuple(metadata.dimensions)} do not match its "
            f"{n_cells} cells; the file is corrupt or was written from an inconsistent grid."
        )
        raise ValueError(msg)

    esgrid = ExplicitStructuredGrid()
    # Set before the cells, which vtkExplicitStructuredGrid.SetExtent replaces.
    esgrid.SetExtent(*_extent(metadata.dimensions, metadata.offset))
    esgrid.points = points
    if cells is not None:
        # VTK misreads ESG cells narrower than vtkIdType: https://gitlab.kitware.com/vtk/vtk/-/work_items/20126
        if cells.IsHomogeneous() > 0 and hasattr(cells, "ConvertToFixedSizeDefaultStorage"):
            cells.ConvertToFixedSizeDefaultStorage()
        else:
            cells.ConvertToDefaultStorage()
        esgrid.SetCells(cells)
    if metadata.faces_connectivity_flags_array_name is not None:
        esgrid.SetFacesConnectivityFlagsArrayName(metadata.faces_connectivity_flags_array_name)
    return esgrid


def _segments_to_esgrid_by_blocking(
    ds_id: str,
    segments: dict[str, Any],
    metadata: DataSetMetadata,
) -> ExplicitStructuredGrid:
    """Rebuild an explicit structured grid with no stored dimensions from its blocking cell arrays."""
    ugrid = _segments_to_ugrid(ds_id, segments, metadata)

    missing = []
    for name in ESGRID_BLOCK_KEYS:
        key = f"{ds_id}{name}{CELL_DATA_SUFFIX}"
        if key in segments:
            ugrid.cell_data.set_array(np.asarray(segments[key]), name)
        else:
            missing.append(name)
    if missing:
        absent = [name for name in missing if name not in metadata.cell_data_keys]
        if absent:
            advice = f"The file does not store {absent}; rewrite it from the source grid with a current pyvista-zstd."
        else:
            advice = f"Select {missing} in selected_cell_arrays to read this file."
        msg = (
            f"Cannot rebuild an ExplicitStructuredGrid without {missing}, which define its "
            f"i/j/k blocking in a file with no stored dimensions. {advice}"
        )
        raise ValueError(msg)

    return ugrid.cast_to_explicit_structured_grid()


def _extent(dimensions: Sequence[int], offset: Sequence[int] | None) -> tuple[int, ...]:
    """Return the extent of a grid with these dimensions starting at offset, which defaults to 0."""
    starts = (0, 0, 0) if offset is None else offset
    return tuple(bound for start, dim in zip(starts, dimensions, strict=True) for bound in (start, start + dim - 1))


def _apply_offset(ds: StructuredGrid | RectilinearGrid, metadata: DataSetMetadata) -> None:
    """Shift a structured grid's extent to start at its stored offset."""
    if metadata.offset is not None and any(metadata.offset):
        ds.SetExtent(*_extent(ds.dimensions, metadata.offset))


def _segments_to_sgrid(ds_id: str, segments: dict[str, Any], metadata: DataSetMetadata) -> StructuredGrid:
    sgrid = StructuredGrid(segments[f"{ds_id}{POINTS_KEY}"])
    sgrid.dimensions = metadata.dimensions
    _apply_offset(sgrid, metadata)
    return sgrid


def _numpy_to_vtk_cells(
    offset: NDArray[np.int32] | NDArray[np.int64] | None,
    connectivity: NDArray[np.int32] | NDArray[np.int64],
    *,
    cell_size: int | None = None,
) -> vtkCellArray:
    # Build directly via VTK to preserve int32/int64 dtype on the connectivity
    # array. ``pv.CellArray.from_arrays`` always casts to ``pv.ID_TYPE``
    # (typically int64) which would defeat ``force_int32``.
    dtype = connectivity.dtype
    if dtype == np.int32:
        vtk_dtype = vtkTypeInt32Array().GetDataType()
    elif dtype == np.int64:
        vtk_dtype = vtkTypeInt64Array().GetDataType()
    else:  # pragma: no cover
        msg = f"Invalid faces dtype {dtype}. Expected `np.int32` or `np.int64`."
        raise ValueError(msg)
    connectivity_vtk = numpy_to_vtk(connectivity, deep=False, array_type=vtk_dtype)
    carr = vtkCellArray()
    vtk_arrays = [connectivity_vtk]
    if cell_size is not None:
        if cell_size <= 0 or connectivity.size % cell_size:
            msg = f"Invalid fixed cell size {cell_size} for connectivity length {connectivity.size}."
            raise ValueError(msg)
        carr.SetData(cell_size, connectivity_vtk)
    else:
        if offset is None:  # pragma: no cover
            msg = "Offsets are required when no fixed cell size is provided."
            raise ValueError(msg)
        offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)
        carr.SetData(offset_vtk, connectivity_vtk)
        vtk_arrays.append(offset_vtk)
    # ``numpy_to_vtk(deep=False)`` stores its NumPy owner on the Python VTK
    # array wrapper. Keep those wrappers with the cell array so shuffled
    # topology buffers remain alive after the temporary segment mapping is
    # released. VTK preserves the cell-array wrapper while a dataset owns it.
    carr.pyvista_zstd_array_references = vtk_arrays
    return carr


def _segments_to_polydata(
    ds_id: str,
    segments: dict[str, Any],
    metadata: DataSetMetadata,
) -> PolyData:
    pdata = PolyData()
    _set_points(pdata, ds_id, segments)

    pdata.SetPolys(_extract_cell_array(ds_id, POLYS, segments, metadata.fixed_cell_sizes))
    pdata.SetLines(_extract_cell_array(ds_id, LINES, segments, metadata.fixed_cell_sizes))
    pdata.SetStrips(_extract_cell_array(ds_id, STRIPS, segments, metadata.fixed_cell_sizes))
    pdata.SetVerts(_extract_cell_array(ds_id, VERTS, segments, metadata.fixed_cell_sizes))

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


def _segments_to_rgrid(ds_id: str, segments: dict[str, Any], metadata: DataSetMetadata) -> RectilinearGrid:
    if f"{ds_id}{RGRID_X_SUFFIX}" in segments:
        rgrid = RectilinearGrid(
            segments.get(f"{ds_id}{RGRID_X_SUFFIX}"),
            segments.get(f"{ds_id}{RGRID_Y_SUFFIX}"),
            segments.get(f"{ds_id}{RGRID_Z_SUFFIX}"),
        )
    else:
        rgrid = RectilinearGrid()

    _apply_offset(rgrid, metadata)
    return rgrid


def _apply_metadata(ds: DataSet, metadata: DataSetMetadata, attributes: dict[str, Any]) -> None:
    """Apply metadata to a dataset, through the attribute sets in *attributes* where already fetched."""
    for getter, scalars, vectors, tcoords, normals in (
        (
            "GetPointData",
            metadata.point_data_active_scalars_name,
            metadata.point_data_active_vectors_name,
            metadata.point_data_active_texture_coordinates_name,
            metadata.point_data_active_normals_name,
        ),
        (
            "GetCellData",
            metadata.cell_data_active_scalars_name,
            metadata.cell_data_active_vectors_name,
            metadata.cell_data_active_texture_coordinates_name,
            metadata.cell_data_active_normals_name,
        ),
    ):
        if scalars is None and vectors is None and tcoords is None and normals is None:
            continue
        field_data = attributes.get(getter)
        if field_data is None:
            field_data = getattr(ds, getter)()
        # Optimization: raw ``SetActive*`` calls; pyvista's setters re-fetch and re-wrap the array each time.
        # Only a vtkDataArray can be active scalars; a string array is left as is.
        if scalars is not None and isinstance(field_data.GetAbstractArray(scalars), vtkDataArray):
            field_data.SetActiveScalars(scalars)
        if vectors is not None and field_data.HasArray(vectors):
            field_data.SetActiveVectors(vectors)
        if tcoords is not None and field_data.HasArray(tcoords):
            field_data.SetActiveTCoords(tcoords)
        if normals is not None and field_data.HasArray(normals):
            field_data.SetActiveNormals(normals)


try:
    from pyvista import LocalFileRequiredError
    from pyvista import has_scheme
except ImportError:
    # Requires pyvista >= 0.48
    LocalFileRequiredError = None  # type: ignore[assignment, misc]
    has_scheme = None  # type: ignore[assignment]


def _warn_backend_deprecated(backend: object) -> None:
    """Warn that ``backend=`` is deprecated, and ignore it."""
    if backend is None:
        return
    msg = (
        "The 'backend' argument is deprecated and no longer has any effect. "
        "The C++ core is the only implementation, so there is nothing to "
        "select between. Remove the argument."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def read(
    filename: Path | str,
    n_threads: int | None = None,
    *,
    backend: str | None = None,
) -> DataSet:
    """
    Decompress a ``pyvista-zstd`` file.

    This is a convenience function that uses :class:`Reader`. Use that class to
    finely tune reading in a file.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to the file.
    n_threads : None | int, optional
        Workers to spread the frames over. If omitted, the count is chosen
        from the total size being decompressed. ``-1`` uses all available
        cores and ``0`` disables multi-threading. Frames are independent, so
        this changes how long the read takes and nothing about its result.
    backend : str, optional
        Deprecated and ignored; passing it raises a :class:`DeprecationWarning`.

    Returns
    -------
    pyvista.DataSet

    Raises
    ------
    pyvista.LocalFileRequiredError
        If *filename* is a remote URI. When called via
        :func:`pyvista.read`, this triggers an automatic download
        and retry with a local path.

    Examples
    --------
    >>> import pyvista_zstd
    >>> ds = pyvista_zstd.read("dataset.pv")

    """
    if has_scheme is not None and has_scheme(str(filename)):
        raise LocalFileRequiredError
    _warn_backend_deprecated(backend)
    with Reader(filename) as reader:
        return reader.read(n_threads=n_threads)


def read_buffer(data: bytes | bytearray | memoryview | NDArray[Any], n_threads: int | None = None) -> DataSet:
    """
    Decompress a ``pyvista-zstd`` container already held in memory.

    :func:`read` for bytes rather than a path -- an archive member, a response
    body, or a build with no filesystem to open a path on. The bytes are
    borrowed, not copied, and must not be modified while the read is running:
    a write into them is not detected, so the dataset comes back with undefined
    contents and no error to say so. A resize is detected, and raises
    :class:`BufferError` rather than corrupting anything.

    This is a convenience function that uses :class:`Reader`. Use that class to
    finely tune reading.

    Parameters
    ----------
    data : bytes | bytearray | memoryview | numpy.ndarray
        A whole ``.pv`` or ``.zvtk`` container, contiguous.
    n_threads : None | int, optional
        Workers to spread the frames over, exactly as in :func:`read`.

    Returns
    -------
    pyvista.DataSet

    Examples
    --------
    >>> from pathlib import Path
    >>> import pyvista_zstd
    >>> ds = pyvista_zstd.read_buffer(Path("dataset.pv").read_bytes())

    """
    # Closed on the way out, so the caller's bytes stop being borrowed as soon
    # as the arrays are built rather than whenever the reader is collected.
    with Reader(buffer=data) as reader:
        return reader.read(n_threads=n_threads)


def _check_association(association: str) -> None:
    """Raise ValueError unless *association* is one of the valid associations."""
    if association not in _ASSOCIATION_SUFFIXES:
        msg = f"association must be one of {tuple(_ASSOCIATION_SUFFIXES)}, not {association!r}"
        raise ValueError(msg)


def _association_arrays(meta: DataSetMetadata | MultiBlockMetadata, association: str) -> dict[str, ArrayInfo]:
    """Return the arrays of *meta* that belong to *association*."""
    _check_association(association)
    if isinstance(meta, MultiBlockMetadata) and association != "field":
        return {}
    return getattr(meta, f"{association}_data_keys")


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
            # Walk ``children``, not ``children_ds``. The mapping is keyed by
            # UID, and a MultiBlock may hold the same dataset in two slots (or
            # two empty ones), so iterating the mapping loses the repeat and
            # leaves fewer readers than the block has keys.
            for child_uid in metadata.children:
                self._children.append(_DataSetReader(metadata.children_ds[child_uid], parent))

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

    @property
    def dataset_metadata(self) -> DataSetMetadata | MultiBlockMetadata | None:
        """Return this block's metadata, or ``None`` for an empty block; treat it as read-only."""
        return self._meta

    def read_array(self, name: str, *, association: Association) -> NDArray[Any]:
        """Decompress one named array of this block; see :meth:`Reader.read_array`."""
        _check_association(association)
        if self._meta is None or (isinstance(self._meta, MultiBlockMetadata) and association != "field"):
            msg = "Only dataset blocks have point and cell arrays; index a MultiBlock node further."
            raise TypeError(msg)
        return self._parent._read_array(self._meta, name, association)  # noqa: SLF001

    def read(self, n_threads: int | None = None) -> DataSet | MultiBlock:
        return self._read_shared({}, n_threads)

    def _read_shared(self, built: dict[str, DataSet], n_threads: int | None) -> DataSet | MultiBlock:
        """
        Read this node, reusing datasets already built in this call.

        A dataset stored once and referenced from two blocks comes back as one
        object in both, which is how it was written and what callers compare
        with ``is``. Decompressing it twice would also cost twice.
        """
        if isinstance(self._meta, DataSetMetadata):
            uid = self.uid
            if uid not in built:
                built[uid] = self._parent._read_ds(uid, n_threads)  # noqa: SLF001
            return built[uid]
        if isinstance(self._meta, MultiBlockMetadata):
            mb = MultiBlock()
            # Optimization: ``mb[key] = ...`` searches every existing name, so track them here.
            # A repeated name replaces its first block, as ``mb[key]`` would.
            index_of: dict[str, int] = {}
            for key, child in zip(self._meta.children_keys, self._children, strict=True):
                block = child._read_shared(built, n_threads)  # noqa: SLF001
                index = index_of.get(key)
                if index is None:
                    index_of[key] = mb.n_blocks
                    mb.append(block, key)
                else:
                    mb[index] = block
                    mb.set_block_name(index, key)
            self._parent._read_multiblock_field_data(mb, self._meta, n_threads)  # noqa: SLF001
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
            fields = ""
            if self._meta.field_data_keys:
                fields = f", field arrays: {', '.join(self._meta.field_data_keys)}"
            lines = [f"{prefix}{connector}MultiBlock(children={len(self._children)}{fields})"]
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
    Class to control pyvista-zstd file decompression.

    Use this class in lieu of :func:`pyvista_zstd.read` to fine-tune reading in
    compressed files. With this you can:

    * Inspect the dataset before reading it.
    * Control which arrays to read in.
    * For files containing a :class:`pyvista.MultiBlock`, select which blocks
      to read in.

    A container already resident is read by passing ``buffer`` instead of
    ``filename`` -- an archive member, a response body, or a build with no
    filesystem to open a path on. Those bytes are borrowed rather than copied
    and are held for the reader's life, so they must not be modified meanwhile.
    Writing into them is not detected: the arrays read afterwards hold
    undefined contents, and no error is raised to say so. Resizing them is
    detected -- the borrow pins the buffer, so a resize raises
    :class:`BufferError` rather than corrupting anything.

    A reader holds the file mapped, or the caller's buffer alive, until
    :meth:`close` -- which ``with`` calls on the way out:

    .. code-block:: python

        with pyvista_zstd.Reader(buffer=raw) as reader:
            ds = reader.read()

    Parameters
    ----------
    filename : pathlib.Path | str, optional
        Path to the file. Must end in ``.pv``.
    buffer : bytes | bytearray | memoryview | numpy.ndarray, optional
        A whole container, contiguous. Exactly one of *filename* and *buffer*
        is given; a buffer has no suffix to check, so its bytes decide whether
        it is a container.
    check_extension : bool, default: True
        Set False for an application-specific filename extension. Container
        contents and version are still validated normally.
    backend : str, optional
        Deprecated and ignored; passing it raises a :class:`DeprecationWarning`.

    Examples
    --------
    First write out an example dataset.

    >>> import pyvista as pv
    >>> import pyvista_zstd
    >>> ds = pv.Sphere()
    >>> pyvista_zstd.write(ds, "sphere.pv")

    Create a reader.

    >>> reader = pyvista_zstd.Reader("sphere.pv")
    >>> reader
    pyvista_zstd.Reader (0x7f1ed1496c00)
      File:               sphere.pv
      File Version:       0
      Compression:        zstandard
      Compression Level:  3
      Dataset Type:       PolyData
      N Points:           842 (float32)
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

    Read the same container out of memory instead of off disk.

    >>> from pathlib import Path
    >>> raw = Path("sphere.pv").read_bytes()
    >>> ds_in = pyvista_zstd.Reader(buffer=raw).read()

    """

    def __init__(
        self,
        filename: Path | str | None = None,
        *,
        buffer: bytes | bytearray | memoryview | NDArray[Any] | None = None,
        backend: str | None = None,
        check_extension: bool = True,
    ) -> None:
        """
        Initialize the decompressor.

        ``backend`` is deprecated and ignored.
        ``check_extension=False`` permits application-specific filenames;
        container contents and version are still validated normally.
        """
        _warn_backend_deprecated(backend)
        if (filename is None) == (buffer is None):
            msg = "Reader takes exactly one of `filename` and `buffer`"
            raise TypeError(msg)

        self._filename = None if filename is None else Path(filename)
        self._buffer = buffer
        self._selected_point_arrays: set[str] | None = None
        self._selected_cell_arrays: set[str] | None = None
        self._selected_field_arrays: set[str] | None = None
        self._core: CoreReader | None = None
        self._closed = False

        # Only a path carries a suffix to judge; a buffer is judged by its bytes.
        if check_extension and self._filename is not None and self._filename.suffix not in SUPPORTED_READ_SUFFIXES:
            msg = f"Filename must end in one of {SUPPORTED_READ_SUFFIXES}, not '{self._filename.suffix}'"
            raise ValueError(msg)

        try:
            core = self._core_reader()
        except _capi.ContainerFormatError as err:
            # Callers catch on this wording, so keep it.
            msg = f"'{self._source}' did not parse as a pyvista-zstd container. File may be corrupted."
            raise RuntimeError(msg) from err
        self._frame_decompressed, self._compressed_sizes = core.frame_sizes()
        self._metadata_documents = dict(core.metadata_documents())
        self._thumbnail = core.thumbnail()

        self._metadata = self._file_metadata_from_core()
        self._ds_metadata = self._root_ds_meta_from_core()

        self.__ds_reader: _DataSetReader | None = None
        self._frames_by_ds: dict[str, list[str]] | None = None
        self._ds_meta_by_uid: dict[str, DataSetMetadata] = {}
        if isinstance(self._ds_metadata, DataSetMetadata):
            self._ds_meta_by_uid[self._ds_metadata.uid] = self._ds_metadata

    # PYI034 wants `Self`, which is 3.11+; this package supports 3.10.
    def __enter__(self) -> Reader:  # noqa: PYI034
        """Return self; the reader is already open."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the container, as :meth:`close` does."""
        self.close()

    def close(self) -> None:
        """
        Release the container. Safe to call more than once.

        Unmaps the file, or drops the reference to the buffer this reader was
        given -- which, until now, has kept those bytes alive on the caller's
        behalf. Reading anything afterwards raises :class:`ValueError`; the
        metadata taken when the container was opened stays readable, because it
        was copied out then and no longer touches the container.
        """
        if self._core is not None:
            self._core.close()
            self._core = None
        self._buffer = None
        self._closed = True

    @property
    def _source(self) -> str:
        """Name this container came under, for messages and for the repr."""
        return "<memory>" if self._filename is None else str(self._filename)

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

    @staticmethod
    def _ds_meta_from_json(frame_name: str, raw: str) -> DataSetMetadata | MultiBlockMetadata:
        """
        Decode one dataset-metadata document.

        The class comes from the frame's name, not from the contents: a lone
        document may be either kind and the two overlap in what they carry.
        """
        arr = np.frombuffer(raw.encode("utf-8"), dtype=np.uint8)
        if frame_name.endswith(MULTIBLOCK_METADATA_KEY):
            return MultiBlockMetadata.from_array(arr)
        if frame_name.endswith(DS_METADATA_KEY):
            return DataSetMetadata.from_array(arr)

        msg = "Metadata key invalid."  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    @property
    def thumbnail(self) -> bytes | None:
        """
        Return the embedded PNG thumbnail, or None when the file has none.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_zstd
        >>> pyvista_zstd.write(pv.Sphere(), "sphere.pv", thumbnail=True)  # doctest: +SKIP
        >>> png = pyvista_zstd.Reader("sphere.pv").thumbnail  # doctest: +SKIP

        """
        return None if self._thumbnail is None else self._thumbnail[0]

    @property
    def thumbnail_info(self) -> ThumbnailInfo | None:
        """
        Return the thumbnail's pixel size and staleness, or None when the file has none.

        A thumbnail is stale when the container's data was changed after it was
        written, for example by :func:`pyvista_zstd.append_arrays`.
        """
        if self._thumbnail is None:
            return None
        _, width, height, stale = self._thumbnail
        return ThumbnailInfo(width=width, height=height, stale=stale)

    @property
    def decompressed_sizes(self) -> NDArray[np.uint64]:
        """
        Return decompressed frame sizes.

        This an array containing 64-bit unsigned integers containing the
        decompressed sizes in bytes of each frame.
        """
        return self._frame_decompressed

    @property
    def nbytes(self) -> int:
        """Return the size of the decompressed dataset."""
        return int(self.decompressed_sizes.sum())

    @property
    def compressed_nbytes(self) -> int:
        """Return the size in bytes of the compressed frames, metadata frames and any thumbnail included."""
        return int(self._compressed_sizes.sum())

    @property
    def file_metadata(self) -> ZstdFileMetadata:
        """
        Return the file-level facts: format version, compression and level.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_zstd
        >>> pyvista_zstd.write(pv.Sphere(), "sphere.pv", level=5)
        >>> reader = pyvista_zstd.Reader("sphere.pv")
        >>> reader.file_metadata.compression_level
        5

        """
        return self._metadata

    @property
    def dataset_metadata(self) -> DataSetMetadata | MultiBlockMetadata:
        """
        Return the dataset's metadata without decompressing any array.

        A MultiBlock file returns a :class:`MultiBlockMetadata` whose
        ``children_ds`` holds the metadata of every block, nested. The objects
        are the reader's own, so treat them as read-only.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_zstd
        >>> pyvista_zstd.write(pv.Sphere(), "sphere.pv")
        >>> meta = pyvista_zstd.Reader("sphere.pv").dataset_metadata
        >>> meta.ds_type, meta.n_points, meta.n_cells
        ('PolyData', 842, 1680)
        >>> meta.point_data_keys["Normals"]
        ArrayInfo(shape=(842, 3), dtype='float32')

        """
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return cast("MultiBlockMetadata", self._ds_reader.dataset_metadata)
        return self._ds_metadata

    def read_array(self, name: str, *, association: Association) -> NDArray[Any]:
        """
        Decompress one named array, leaving the rest of the file untouched.

        A block of a MultiBlock file is read the same way after indexing to it,
        as in ``reader[0].read_array(name, association="point")``.
        A MultiBlock node's own field arrays use ``association="field"``.

        Parameters
        ----------
        name : str
            Array name.
        association : {"point", "cell", "field"}
            Data the array belongs to.

        Returns
        -------
        numpy.ndarray
            The stored array.

        Raises
        ------
        KeyError
            The dataset has no such array.
        TypeError
            The file is a MultiBlock and the association is point or cell;
            index it to a dataset block first.
        ValueError
            *association* is not one of the three above.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_zstd
        >>> pyvista_zstd.write(pv.Sphere(), "sphere.pv")
        >>> reader = pyvista_zstd.Reader("sphere.pv")
        >>> reader.read_array("Normals", association="point").shape
        (842, 3)

        """
        _check_association(association)
        if isinstance(self._ds_metadata, MultiBlockMetadata) and association != "field":
            msg = "A MultiBlock file has no point or cell arrays; read a block's arrays with reader[i].read_array(...)."
            raise TypeError(msg)
        return self._read_array(self._ds_metadata, name, association)

    def _read_array(self, meta: DataSetMetadata | MultiBlockMetadata, name: str, association: str) -> NDArray[Any]:
        """Decompress the named array of the dataset *meta* describes."""
        arrays = _association_arrays(meta, association)
        if name not in arrays:
            msg = f"{association} array {name!r} not found; available: {sorted(arrays)}"
            elsewhere = [other for other in _ASSOCIATION_SUFFIXES if name in _association_arrays(meta, other)]
            if elsewhere:
                msg += f"; {name!r} is a {' and '.join(elsewhere)} array"
            raise KeyError(msg)
        core = self._core_reader()
        index = core.find(f"{meta.uid}{name}{_ASSOCIATION_SUFFIXES[association]}")
        if index is None:  # pragma: no cover - metadata and index disagree
            msg = f"{association} array {name!r} is listed in the metadata but not stored"
            raise RuntimeError(msg)
        return core.read_at(index)

    def _file_metadata_from_core(self) -> ZstdFileMetadata:
        """
        Return the file metadata, and warn if the container is a legacy one.

        The core accepts both spellings of the frame and reports which one the
        file used, so the deprecation notice is raised from the name rather
        than by parsing the trailer a second time to find out.

        Nothing here decides whether the container is readable: a version this
        build cannot decode was refused when the core opened the file, so by the
        time this runs the document is one this build understands.
        """
        raw = self._metadata_documents.get(FILE_METADATA_KEY)
        if raw is None:
            raw = self._metadata_documents.get(LEGACY_FILE_METADATA_KEY)
            if raw is None:  # pragma: no cover
                msg = "File metadata not found in pyvista-zstd file."
                raise RuntimeError(msg)
            # FutureWarning (not DeprecationWarning) because this is aimed at
            # end users re-saving their data files, and Python's default
            # warning filters hide DeprecationWarning from non-__main__ code.
            warnings.warn(
                f"'{self._source}' is a legacy zvtk file. Support for the "
                "'.zvtk' format will be removed in a future release; re-save "
                "it with `pyvista_zstd.write(pyvista_zstd.read(path), new_path)`.",
                FutureWarning,
                stacklevel=3,
            )

        return ZstdFileMetadata.from_json(raw)

    def _root_ds_meta_from_core(self) -> DataSetMetadata | MultiBlockMetadata:
        """
        Return the root dataset's metadata.

        The root is the first dataset-metadata frame in the file, which is the
        order ``frame_names`` records; the core's documents are keyed by name
        rather than ordered by role, so the order comes from there.
        """
        for frame_name in self._metadata.frame_names or ():
            if frame_name.endswith(DS_METADATA_KEY):
                raw = self._metadata_documents.get(frame_name)
                if raw is None:  # pragma: no cover - named but not carried
                    break
                return self._ds_meta_from_json(frame_name, raw)

        msg = "No dataset metadata found"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    def _read_ds_segments_cpp(self, ds_id: str, keep: set[str], n_threads: int | None) -> _Segments:
        """
        Decompress a dataset's arrays through the C++ core.

        The core lifts the dataset-metadata frame out into JSON and does not
        report it among the arrays, so it is taken from the parked documents
        by name -- a MultiBlock carries one per block.
        """
        meta_key = f"{ds_id}{DS_METADATA_KEY}"
        reader = self._core_reader()
        threads = _capi.THREADS_AUTO if n_threads is None else n_threads
        segments = _Segments()
        segments.update(reader.read_arrays(keep=keep - {meta_key}, n_threads=threads, allocate=segments.allocate))
        return segments

    def _dataset_metadata(self, ds_id: str) -> DataSetMetadata:
        """Return the parsed metadata of dataset *ds_id*, parsing its document once."""
        metadata = self._ds_meta_by_uid.get(ds_id)
        if metadata is None:
            metadata = DataSetMetadata.from_json(self._metadata_documents[f"{ds_id}{DS_METADATA_KEY}"])
            self._ds_meta_by_uid[ds_id] = metadata
        return metadata

    def _core_reader(self) -> CoreReader:
        """
        Return the C++ reader for this container, opened once and kept.

        Opening one takes the container's bytes -- mapping the file, or
        borrowing the buffer the ``buffer`` argument was given -- and parses
        the trailer and every array header. This object did all of that already
        at construction, so building a
        fresh C++ reader per dataset paid for a second copy of it on every
        read -- and a MultiBlock paid once per dataset.

        Holding it does not weaken any guarantee that was being offered: the
        frame index and the mapping are both taken in __init__ and kept, so a
        reader has always been a snapshot of the file as it was when opened.
        It is released by :meth:`close`, or when this object is collected.
        """
        if self._closed:
            msg = "operation on a closed Reader"
            raise ValueError(msg)
        reader = self._core
        if reader is None:
            reader = _capi.CoreReader(self._filename) if self._buffer is None else _capi.CoreReader(buffer=self._buffer)
            self._core = reader
        return reader

    def _selected_frame_names(self, ds_id: str) -> set[str]:
        """
        Return the frame names to decompress for one dataset.

        Applies the array selection, then narrows to frames belonging to
        *ds_id* -- a MultiBlock file holds several datasets' frames side by
        side and only this one's are wanted.
        """
        excluded = set()
        for name in self.available_point_arrays - self.selected_point_arrays:
            excluded.add(f"{ds_id}{name}{POINT_DATA_SUFFIX}")
        for name in self.available_cell_arrays - self.selected_cell_arrays:
            excluded.add(f"{ds_id}{name}{CELL_DATA_SUFFIX}")
        for name in self.available_field_arrays - self.selected_field_arrays:
            excluded.add(f"{ds_id}{name}{FIELD_DATA_SUFFIX}")

        return {f for f in self._frame_names_by_ds().get(ds_id, ()) if f not in excluded}

    def _frame_names_by_ds(self) -> dict[str, list[str]]:
        """Return the file's frame names grouped by their dataset UID prefix."""
        if self._frames_by_ds is None:
            groups: dict[str, list[str]] = {}
            for name in self._metadata.frame_names or ():
                groups.setdefault(name[:UID_N_CHAR], []).append(name)
            self._frames_by_ds = groups
        return self._frames_by_ds

    def _read_multiblock_field_data(self, mb: MultiBlock, meta: MultiBlockMetadata, n_threads: int | None) -> None:
        """Add a MultiBlock's own field arrays from the file."""
        if not meta.field_data_keys:
            return
        frames = {f"{meta.uid}{name}{FIELD_DATA_SUFFIX}": name for name in meta.field_data_keys}
        threads = _capi.THREADS_AUTO if n_threads is None else n_threads
        segments = self._core_reader().read_arrays(keep=set(frames), n_threads=threads)
        for frame, name in frames.items():
            _set_attribute_array(mb.field_data, segments[frame], name, field=True)

    # @profile
    def _read_ds(self, ds_id: str, n_threads: int | None = None) -> DataSet:
        """Read a single dataset."""
        # map frame indices to names using metadata
        frame_names = self._metadata.frame_names
        if frame_names is None:  # pragma: no cover
            msg = "Frame names not found in metadata."
            raise RuntimeError(msg)

        selected_frame_names = self._selected_frame_names(ds_id)

        if not selected_frame_names:  # pragma: no cover
            msg = "No selected frames"
            raise RuntimeError(msg)

        # Frame-addressed, so downselecting skips the decompression too, not
        # just the copy into the dataset.
        segments = self._read_ds_segments_cpp(ds_id, selected_frame_names, n_threads)
        ds = self._segments_to_ds(ds_id, segments, self._dataset_metadata(ds_id))
        segments.release()
        return ds

    def _load_ds_reader(self) -> _DataSetReader:  # noqa: C901
        """Read metadata hierarchy from the pyvista-zstd file."""
        if not isinstance(self._ds_metadata, MultiBlockMetadata):
            msg = "Can only index a MultiBlock compressed pyvista-zstd file."
            raise TypeError(msg)

        # decode metadata objects -- the core decompressed every one of these
        # at open, so the hierarchy is assembled from documents it already holds
        mblock_meta: dict[str, MultiBlockMetadata] = {}
        dataset_meta: dict[str, DataSetMetadata] = {}
        for key, raw in self._metadata_documents.items():
            if key.endswith(MULTIBLOCK_METADATA_KEY):
                mb_meta = MultiBlockMetadata.from_array(np.frombuffer(raw.encode("utf-8"), dtype=np.uint8))
                mblock_meta[mb_meta.uid] = mb_meta
            elif key.endswith(DS_METADATA_KEY):
                dataset_meta[key[:UID_N_CHAR]] = self._dataset_metadata(key[:UID_N_CHAR])

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

    def read(self, n_threads: int | None = None) -> DataSet:
        """
        Read in the dataset from the pyvista-zstd file.

        Parameters
        ----------
        n_threads : int, optional
            Workers to spread the frames over. A value of ``-1`` uses all
            available cores and ``0`` disables multi-threading. Frames are
            independent, so this changes how long the read takes and nothing
            about its result.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_zstd
        >>> ds = pv.Sphere()
        >>> pyvista_zstd.write(ds, "sphere.pv")
        >>> reader = pyvista_zstd.Reader("sphere.pv")
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

        # Same entry point as an indexed read, so the two cannot drift apart.
        return self._ds_reader.read(n_threads)

    def _segments_to_ds(self, ds_id: str, segments: dict[str, Any], ds_metadata: DataSetMetadata) -> DataSet:
        """Build the dataset *ds_id* from its decompressed frames."""
        # convert this to match when Python 3.9 goes EOL
        ds_type = ds_metadata.ds_type
        if ds_type == "UnstructuredGrid":
            ds = _segments_to_ugrid(ds_id, segments, ds_metadata)
        elif ds_type == "PolyData":
            ds = _segments_to_polydata(ds_id, segments, ds_metadata)
        elif ds_type == "ImageData":
            ds = _metadata_to_imagedata(ds_metadata)
        elif ds_type == "PointSet":
            ds = _segments_to_pointset(ds_id, segments)
        elif ds_type == "RectilinearGrid":
            ds = _segments_to_rgrid(ds_id, segments, ds_metadata)
        elif ds_type == "StructuredGrid":
            ds = _segments_to_sgrid(ds_id, segments, ds_metadata)
        elif ds_type == "ExplicitStructuredGrid":
            ds = _segments_to_esgrid(ds_id, segments, ds_metadata)
        else:  # pragma: no cover
            msg = f"pyvista-zstd does not support DataSet type `{ds_type}` for decompression"
            raise RuntimeError(msg)

        _apply_metadata(ds, ds_metadata, _add_data(ds_id, ds, segments))
        return ds

    @property
    def available_point_arrays(self) -> set[str]:
        """
        Return a set of all point array names available in the dataset.

        Empty for a MultiBlock file, which reads every array.

        Returns
        -------
        set[str]
            Names of all point arrays available in the dataset.

        Examples
        --------
        First write out an example dataset.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> import pyvista_zstd
        >>> ds = pv.Sphere()
        >>> ds.point_data["pdata"] = np.arange(ds.n_points)
        >>> pyvista_zstd.write(ds, "sphere.pv")

        Create a reader and list available point arrays.

        >>> reader = pyvista_zstd.Reader("sphere.pv")
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

        Empty for a MultiBlock file, which reads every array.

        Returns
        -------
        set[str]
            Names of all cell arrays available in the dataset.

        Examples
        --------
        First write out an example dataset.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> import pyvista_zstd
        >>> ds = pv.Sphere()
        >>> ds.point_data["pdata"] = np.arange(ds.n_points)
        >>> pyvista_zstd.write(ds, "sphere.pv")

        Create a reader and list available point arrays.

        >>> reader = pyvista_zstd.Reader("sphere.pv")
        >>> reader.available_point_arrays
        {"Normals", "pdata"}

        """
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return set()
        return set(self._ds_metadata.cell_data_keys)

    @property
    def available_field_arrays(self) -> set[str]:
        """
        Return a set of all field array names available in the dataset.

        Empty for a MultiBlock file, which reads every array.
        """
        if isinstance(self._ds_metadata, MultiBlockMetadata):
            return set()
        return set(self._ds_metadata.field_data_keys)

    @property
    def selected_point_arrays(self) -> set[str]:
        """
        Return the set of currently selected point arrays to read.

        Defaults to all available arrays. Applies to a single-dataset file only.
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

        Defaults to all available arrays. Applies to a single-dataset file only.
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

        Defaults to all available arrays. Applies to a single-dataset file only.
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
            f"pyvista_zstd.Reader ({hex(id(self))})",
            f"  File:               {self._source}",
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
        >>> import pyvista_zstd
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

        >>> pyvista_zstd.write(ds, "bracket.pv")
        >>> reader = pyvista_zstd.Reader("bracket.pv")
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

        >>> pyvista_zstd.write(ds, "bracket.pv", level=22)
        >>> reader = pyvista_zstd.Reader("bracket.pv")
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
        for name, comp_size, decomp_size in zip(frame_names, c_sizes, d_sizes, strict=True):
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
