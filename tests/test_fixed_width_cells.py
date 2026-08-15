"""Tests for fixed-width cell arrays (format version 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd.append import append_arrays
from pyvista_zstd.pyvista_zstd import CELLS
from pyvista_zstd.pyvista_zstd import CONNECTIVITY_SUFFIX
from pyvista_zstd.pyvista_zstd import FILE_VERSION_FIXED_WIDTH_CELLS
from pyvista_zstd.pyvista_zstd import FILE_VERSION_UNFILTERED
from pyvista_zstd.pyvista_zstd import OFFSET_SUFFIX
from pyvista_zstd.pyvista_zstd import POLYS
from pyvista_zstd.pyvista_zstd import _numpy_to_vtk_cells

if TYPE_CHECKING:
    from pathlib import Path

TRIANGLE_SIZE = 3
HEXAHEDRON_SIZE = 8


def _frame_name(reader: pz.Reader, name: str, suffix: str) -> str:
    """Return a topology frame name for the root dataset."""
    return f"{reader._ds_metadata.uid}{name}{suffix}"  # noqa: SLF001


def test_homogeneous_polydata_omits_offsets(polydata: pv.PolyData, tmp_path: Path) -> None:
    """Triangle-only PolyData stores its width in metadata, not an offsets frame."""
    path = tmp_path / "triangles.pv"
    pz.write(polydata, path)

    reader = pz.Reader(path)
    frame_names = reader._metadata.frame_names  # noqa: SLF001
    assert _frame_name(reader, POLYS, OFFSET_SUFFIX) not in frame_names
    assert _frame_name(reader, POLYS, CONNECTIVITY_SUFFIX) in frame_names
    assert reader._ds_metadata.fixed_cell_sizes[POLYS] == TRIANGLE_SIZE  # noqa: SLF001
    assert reader._metadata.file_version == FILE_VERSION_FIXED_WIDTH_CELLS  # noqa: SLF001

    result = reader.read()
    assert result == polydata


def test_homogeneous_ugrid_omits_offsets(ugrid: pv.UnstructuredGrid, tmp_path: Path) -> None:
    """The fixed-width encoding applies to homogeneous non-triangle cells too."""
    path = tmp_path / "hexahedra.pv"
    source_dtype = ugrid.cell_connectivity.dtype
    source_connectivity = ugrid.cell_connectivity.copy()
    pz.write(ugrid, path, force_int32=False)

    reader = pz.Reader(path)
    frame_names = reader._metadata.frame_names  # noqa: SLF001
    assert _frame_name(reader, CELLS, OFFSET_SUFFIX) not in frame_names
    assert reader._ds_metadata.fixed_cell_sizes[CELLS] == HEXAHEDRON_SIZE  # noqa: SLF001
    result = reader.read()
    assert result == ugrid
    # The connectivity width is a property of the VTK build, not something this
    # library chooses, so compare against what the source was given rather than
    # asserting int64 outright. Values are compared too, so a correctly typed
    # but corrupted array still fails.
    assert result.cell_connectivity.dtype == source_dtype
    assert np.array_equal(result.cell_connectivity, source_connectivity)


def test_equal_width_different_cell_types_use_fixed_width(tmp_path: Path) -> None:
    """Cell type does not matter when every connectivity tuple has equal width."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    cells = np.array([3, 0, 1, 2, 3, 3, 4, 5])
    cell_types = np.array([pv.CellType.TRIANGLE, pv.CellType.POLY_LINE], dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    path = tmp_path / "different-types.pv"

    pz.write(grid, path)
    reader = pz.Reader(path)

    assert reader._ds_metadata.fixed_cell_sizes[CELLS] == TRIANGLE_SIZE  # noqa: SLF001
    assert np.array_equal(reader.read().celltypes, cell_types)
    assert reader.read() == grid


def test_mixed_width_cells_keep_offsets(tmp_path: Path) -> None:
    """A mixed triangle and quad mesh retains the legacy offsets representation."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([3, 0, 1, 2, 4, 0, 2, 4, 3])
    mesh = pv.PolyData(points, faces)
    path = tmp_path / "mixed.pv"

    pz.write(mesh, path, shuffle=False)
    reader = pz.Reader(path)

    assert POLYS not in reader._ds_metadata.fixed_cell_sizes  # noqa: SLF001
    assert _frame_name(reader, POLYS, OFFSET_SUFFIX) in reader._metadata.frame_names  # noqa: SLF001
    assert reader._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    assert reader.read() == mesh


def test_fixed_width_and_shuffle_use_version_2(polydata: pv.PolyData, tmp_path: Path) -> None:
    """Format version 2 covers fixed-width topology combined with byte shuffle."""
    path = tmp_path / "fixed-shuffled.pv"
    pz.write(polydata, path, shuffle=True)
    reader = pz.Reader(path)
    assert reader._metadata.file_version == FILE_VERSION_FIXED_WIDTH_CELLS  # noqa: SLF001
    assert reader.read() == polydata


def test_append_preserves_fixed_width_metadata(polydata: pv.PolyData, tmp_path: Path) -> None:
    """Appending field data keeps the fixed-width topology metadata intact."""
    path = tmp_path / "append.pv"
    pz.write(polydata, path)
    append_arrays(path, {"run_id": np.array([7], dtype=np.int32)})

    reader = pz.Reader(path)
    assert reader._ds_metadata.fixed_cell_sizes[POLYS] == TRIANGLE_SIZE  # noqa: SLF001
    assert _frame_name(reader, POLYS, OFFSET_SUFFIX) not in reader._metadata.frame_names  # noqa: SLF001
    assert np.array_equal(reader.read().field_data["run_id"], np.array([7], dtype=np.int32))
    assert reader._metadata.file_version == FILE_VERSION_FIXED_WIDTH_CELLS  # noqa: SLF001


def test_fixed_width_rejects_inconsistent_connectivity() -> None:
    """Malformed fixed-width metadata fails instead of silently dropping values."""
    connectivity = np.arange(5, dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid fixed cell size"):
        _numpy_to_vtk_cells(None, connectivity, cell_size=3)
