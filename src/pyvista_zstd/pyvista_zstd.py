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
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
import warnings

import numpy as np
import pyvista as pv

from pyvista_zstd import _capi

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
    from pyvista._vtk import numpy_to_vtk
    from pyvista._vtk import vtk_to_numpy
    from pyvista._vtk import vtkCellArray
    from pyvista._vtk import vtkPointSet
    from pyvista._vtk import vtkTypeInt32Array
    from pyvista._vtk import vtkTypeInt64Array
except ModuleNotFoundError:  # pyvista < 0.48
    from pyvista.core._vtk_core import numpy_to_vtk
    from pyvista.core._vtk_core import vtk_to_numpy
    from pyvista.core._vtk_core import vtkCellArray
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
    from numpy.typing import NDArray
    from pyvista.core.dataset import DataSet

    from pyvista_zstd._capi import CoreReader

# Highest on-disk format version this library can READ. Version 1 added the
# optional byte-shuffle pre-filter, version 2 fixed-width cell arrays. The
# core enforces the ceiling; this constant only publishes it.
FILE_VERSION = 2
# Version stamped on files that use neither byte filters nor fixed-width cell
# arrays. Such files stay byte-identical to the legacy format and remain
# readable by older releases.
FILE_VERSION_UNFILTERED = 0
FILE_VERSION_SHUFFLE = 1
FILE_VERSION_FIXED_WIDTH_CELLS = 2
FILE_VERSION_KEY = "FILE_VERSION"
DS_TYPE_KEY = "ds_type"
POINT_DATA_SUFFIX = "__point_data"
CELL_DATA_SUFFIX = "__cell_data"
# The cell arrays an ExplicitStructuredGrid needs to be cast back into one.
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


def _shuffle_mode(shuffle: ShuffleSpec) -> int:
    """Translate the API spelling to the core's policy enum."""
    return {
        False: _capi.SHUFFLE_NEVER,
        True: _capi.SHUFFLE_ALWAYS,
        "auto": _capi.SHUFFLE_AUTO,
    }[shuffle]


@dataclass(slots=True, frozen=True)
class ArrayInfo:
    """Array metadata."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(slots=True, frozen=True)
class ZstdFileMetadata:
    """pyvista-zstd file metadata."""

    frame_names: list[str]
    compression_level: int
    compression: Compression = "zstandard"
    # Defaults to the legacy version; writers promote this when they use an
    # optional encoding such as byte shuffle or fixed-width cell topology.
    file_version: int = FILE_VERSION_UNFILTERED

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> ZstdFileMetadata:
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
    fixed_cell_sizes: dict[str, int] = field(default_factory=dict)
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
        fixed_cell_sizes: dict[str, int],
    ) -> DataSetMetadata:
        """Create metadata from a dataset."""
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

        pd = ds.point_data
        cd = ds.cell_data
        kwargs: dict[str, Any] = {
            "ds_type": type(ds).__name__,
            "uid": _make_ds_id(ds),
            "n_points": ds.n_points,
            "points_dtype": str(points_dtype) if points_dtype is not None else None,
            "n_cells": ds.n_cells,
            "celltypes_dtype": str(ds.celltypes.dtype) if hasattr(ds, "celltypes") else None,
            "point_data_keys": point_info,
            "cell_data_keys": cell_info,
            "field_data_keys": field_info,
            "fixed_cell_sizes": fixed_cell_sizes,
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
        raw["fixed_cell_sizes"] = raw.get("fixed_cell_sizes", {})
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
) -> None:
    """
    Stage a cell array's connectivity, and its offsets when the cells vary in size.

    *max_entry* is the largest value the connectivity can hold, where the caller
    knows it: entries are ids into some other array, so that array's extent
    bounds them without looking. Leave it None to have the values scanned. It
    only ever governs the int32 narrowing -- see :func:`_narrowed_to_int32`.
    """
    if not cell_array:
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
    if name in fixed_cell_sizes:
        return _numpy_to_vtk_cells(None, segments[conn_key], cell_size=fixed_cell_sizes[name])
    if offset_key not in segments:
        msg = f"Cell array '{name}' has neither offsets nor a fixed cell size."
        raise ValueError(msg)
    return _numpy_to_vtk_cells(segments[offset_key], segments[conn_key])


def _add_arrays_pointset(ds: PointSet, arrays: dict[str, NDArray[Any]]) -> None:
    arrays[f"{_make_ds_id(ds)}{POINTS_KEY}"] = ds.points


def _add_arrays_rgrid(ds: RectilinearGrid, arrays: dict[str, NDArray[Any]]) -> None:
    if ds.n_points:
        ds_id = _make_ds_id(ds)
        arrays[f"{ds_id}{RGRID_X_SUFFIX}"] = ds.x
        arrays[f"{ds_id}{RGRID_Y_SUFFIX}"] = ds.y
        arrays[f"{ds_id}{RGRID_Z_SUFFIX}"] = ds.z


def _add_arrays_polydata(
    ds: PolyData,
    arrays: dict[str, NDArray[Any]],
    fixed_cell_sizes: dict[str, int],
    *,
    force_int32: bool = True,
) -> None:
    ds_id = _make_ds_id(ds)
    arrays[f"{ds_id}{POINTS_KEY}"] = ds.points
    # Every one of these holds point ids, so the point count bounds them all.
    max_point_id = ds.n_points - 1
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
        )


def _add_arrays_ugrid(
    ds: UnstructuredGrid,
    arrays: dict[str, NDArray[Any]],
    fixed_cell_sizes: dict[str, int],
    ds_id: str | None = None,
    *,
    force_int32: bool = True,
) -> None:
    if ds_id is None:
        ds_id = _make_ds_id(ds)
    arrays[f"{ds_id}{POINTS_KEY}"] = ds.points
    arrays[f"{ds_id}{CELL_TYPES_KEY}"] = ds.celltypes

    max_point_id = ds.n_points - 1
    _add_cell_array(
        ds_id,
        arrays,
        CELLS,
        ds.GetCells(),
        fixed_cell_sizes,
        force_int32=force_int32,
        max_entry=max_point_id,
    )

    has_polyhedra = bool(np.any(ds.celltypes == pv.CellType.POLYHEDRON))
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
    ds: ExplicitStructuredGrid,
    arrays: dict[str, NDArray[Any]],
    fixed_cell_sizes: dict[str, int],
    *,
    force_int32: bool = True,
) -> None:
    ds_id = _make_ds_id(ds)
    ugrid = ds.cast_to_unstructured_grid()
    _add_arrays_ugrid(ugrid, arrays, fixed_cell_sizes, ds_id, force_int32=force_int32)


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
    shuffle: ShuffleSpec = False,
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
        use the filter are written at format version 1 and can only be read by
        this release or newer; unfiltered files (the default) stay backward
        compatible unless they use fixed-width cell arrays.

    Notes
    -----
    Cell arrays whose cells all contain the same number of points are stored
    without their redundant offsets array. Their common width is recorded in
    dataset metadata and VTK reconstructs the offsets when the file is read.
    Files using this encoding are written at format version 2.

    """
    writer = Writer(ds, filename)

    # if compression == "zstandard":
    writer.write(
        progress_bar=progress_bar,
        force_int32=force_int32,
        level=level,
        n_threads=n_threads,
        shuffle=shuffle,
    )


def _make_ds_id(ds: DataSet) -> str:
    """Make a unique dataset ID using the memory address."""
    # padded for 32-bit
    return f"{id(ds):016x}"


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
        fixed_cell_sizes: dict[str, int] = {}

        if isinstance(ds, PolyData):
            _add_arrays_polydata(ds, self._arrays, fixed_cell_sizes, force_int32=force_int32)
        elif isinstance(ds, UnstructuredGrid):
            _add_arrays_ugrid(ds, self._arrays, fixed_cell_sizes, force_int32=force_int32)
        elif isinstance(ds, ExplicitStructuredGrid):
            _add_arrays_esgrid(ds, self._arrays, fixed_cell_sizes, force_int32=force_int32)
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
        ds_meta = DataSetMetadata.from_dataset(ds, point_info, cell_info, field_info, fixed_cell_sizes)
        self._arrays[f"{ds_id}{DS_METADATA_KEY}"] = ds_meta.to_array()
        self._uses_fixed_width_cells = self._uses_fixed_width_cells or bool(fixed_cell_sizes)

    def write(
        self,
        *,
        progress_bar: bool = False,
        force_int32: bool = True,
        level: int = 3,
        n_threads: int | None = None,
        shuffle: ShuffleSpec = False,
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

        self._add_ds_arrays(self._ds, force_int32=force_int32)

        # The core owns the format decisions; this side stages arrays in
        # frame order.
        with _capi.CoreWriter() as writer:
            writer.set_level(level)
            writer.set_threads(_capi.THREADS_AUTO if n_threads is None else n_threads)
            writer.set_shuffle(_shuffle_mode(shuffle))
            writer.set_fixed_width_cells(enabled=self._uses_fixed_width_cells)
            for name, arr in self._arrays.items():
                writer.add_array(name, arr)
            writer.write(self._filename)

        # no need to hold onto any references as all IDs have been written
        self._refs = []


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


def _segments_to_ugrid(
    ds_id: str,
    segments: dict[str, Any],
    metadata: DataSetMetadata,
) -> UnstructuredGrid:
    cells = _extract_cell_array(ds_id, CELLS, segments, metadata.fixed_cell_sizes)

    celltypes = segments[f"{ds_id}{CELL_TYPES_KEY}"]
    celltypes_vtk = numpy_to_vtk(celltypes, deep=False, array_type=VTK_UNSIGNED_CHAR)

    ugrid = UnstructuredGrid()
    ugrid.points = segments[f"{ds_id}{POINTS_KEY}"]

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
    """
    Rebuild an explicit structured grid from its frames.

    The cast is driven by the ``BLOCK_I``/``BLOCK_J``/``BLOCK_K`` cell arrays,
    so they have to be on the grid before it happens. The caller attaches cell
    data only after this returns, which is why they are pulled from the
    segments here rather than left to the general path -- without this the
    cast sees a grid with no cell data at all and refuses every file.
    """
    ugrid = _segments_to_ugrid(ds_id, segments, metadata)

    missing = []
    for name in ESGRID_BLOCK_KEYS:
        key = f"{ds_id}{name}{CELL_DATA_SUFFIX}"
        if key in segments:
            ugrid.cell_data.set_array(np.asarray(segments[key]), name)
        else:
            missing.append(name)
    if missing:
        msg = (
            f"Cannot rebuild an ExplicitStructuredGrid without {missing}. "
            "These cell arrays define the i/j/k blocking and are required by "
            "the cast; do not exclude them via selected_cell_arrays."
        )
        raise ValueError(msg)

    return ugrid.cast_to_explicit_structured_grid()


def _segments_to_sgrid(ds_id: str, segments: dict[str, Any], metadata: DataSetMetadata) -> StructuredGrid:
    sgrid = StructuredGrid(segments[f"{ds_id}{POINTS_KEY}"])
    sgrid.dimensions = metadata.dimensions
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
    pdata.points = segments[f"{ds_id}{POINTS_KEY}"]

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
            for key, child in zip(self._meta.children_keys, self._children, strict=True):
                mb[key] = child._read_shared(built, n_threads)  # noqa: SLF001
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
    ) -> None:
        """
        Initialize the decompressor.

        ``backend`` is deprecated and ignored.
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
        if self._filename is not None and self._filename.suffix not in SUPPORTED_READ_SUFFIXES:
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

        self._metadata = self._file_metadata_from_core()
        self._ds_metadata = self._root_ds_meta_from_core()

        self.__ds_reader: _DataSetReader | None = None

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

    def _read_ds_segments_cpp(self, ds_id: str, keep: set[str], n_threads: int | None) -> dict[str, NDArray[Any]]:
        """
        Decompress a dataset's arrays through the C++ core.

        The core lifts the dataset-metadata frame out into JSON and does not
        report it among the arrays, so it is taken from the parked documents
        by name -- a MultiBlock carries one per block.
        """
        meta_key = f"{ds_id}{DS_METADATA_KEY}"
        reader = self._core_reader()
        threads = _capi.THREADS_AUTO if n_threads is None else n_threads
        segments = reader.read_arrays(keep=keep - {meta_key}, n_threads=threads)
        segments[meta_key] = np.frombuffer(self._metadata_documents[meta_key].encode("utf-8"), dtype=np.uint8)
        return segments

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

        names = set(self._metadata.frame_names) - excluded
        return {f for f in names if f.startswith(ds_id)}

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
        return self._segments_to_ds(ds_id, segments)

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
                uid = key[:UID_N_CHAR]
                dataset_meta[uid] = DataSetMetadata.from_array(np.frombuffer(raw.encode("utf-8"), dtype=np.uint8))

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

    def _segments_to_ds(self, ds_id: str, segments: dict[str, Any]) -> DataSet:
        meta_arr = segments[f"{ds_id}{DS_METADATA_KEY}"]
        ds_metadata = DataSetMetadata.from_array(meta_arr)

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
            ds = _segments_to_rgrid(ds_id, segments)
        elif ds_type == "StructuredGrid":
            ds = _segments_to_sgrid(ds_id, segments, ds_metadata)
        elif ds_type == "ExplicitStructuredGrid":
            ds = _segments_to_esgrid(ds_id, segments, ds_metadata)
        else:  # pragma: no cover
            msg = f"pyvista-zstd does not support DataSet type `{ds_type}` for decompression"
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
