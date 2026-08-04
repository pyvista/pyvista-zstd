"""Tests for lossless mesh-specific filters (format version 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
from vtkmodules.util.numpy_support import vtk_to_numpy

import pyvista_zstd as pz
from pyvista_zstd import pyvista_zstd as _pz_mod
from pyvista_zstd.pyvista_zstd import _FILTER_COMPONENT_SHUFFLE
from pyvista_zstd.pyvista_zstd import _FILTER_NONE
from pyvista_zstd.pyvista_zstd import _FILTER_TRIANGLE_DELTA_SHUFFLE
from pyvista_zstd.pyvista_zstd import CELLS
from pyvista_zstd.pyvista_zstd import CONNECTIVITY_SUFFIX
from pyvista_zstd.pyvista_zstd import FILE_VERSION_FIXED_WIDTH_CELLS
from pyvista_zstd.pyvista_zstd import FILE_VERSION_MESH_FILTERS
from pyvista_zstd.pyvista_zstd import FILE_VERSION_UNFILTERED
from pyvista_zstd.pyvista_zstd import POINTS_KEY
from pyvista_zstd.pyvista_zstd import POLYS
from pyvista_zstd.pyvista_zstd import Writer
from pyvista_zstd.pyvista_zstd import _component_shuffle_bytes
from pyvista_zstd.pyvista_zstd import _special_filter_beneficial
from pyvista_zstd.pyvista_zstd import _triangle_delta_shuffle_bytes
from pyvista_zstd.pyvista_zstd import _uncomponent_shuffle_bytes
from pyvista_zstd.pyvista_zstd import _untriangle_delta_shuffle_bytes

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_component_shuffle_roundtrip_is_bit_exact(dtype: type[np.floating]) -> None:
    """Component-planar byte shuffle preserves every floating-point bit."""
    bits = np.arange(3072 * np.dtype(dtype).itemsize, dtype=np.uint8)
    points = bits.view(dtype).reshape(-1, 3)
    encoded = _component_shuffle_bytes(points)
    recovered = _uncomponent_shuffle_bytes(encoded, points.dtype, points.shape)
    assert recovered.tobytes() == points.tobytes()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_triangle_delta_shuffle_roundtrip_is_bit_exact(dtype: type[np.integer]) -> None:
    """Same-dtype modular deltas preserve signed and unsigned connectivity."""
    info = np.iinfo(dtype)
    triangles = np.array(
        [[0, 1, 2], [9, 4, 7], [info.max, 0, info.min], [3, 2, 1]],
        dtype=dtype,
    ).reshape(-1)
    encoded = _triangle_delta_shuffle_bytes(triangles)
    recovered = _untriangle_delta_shuffle_bytes(encoded, triangles.dtype, triangles.shape)
    assert recovered.tobytes() == triangles.tobytes()


def _writer_filters(mesh: pv.DataSet, path: Path, *, automatic: bool = False) -> tuple[Writer, dict[str, int]]:
    writer = Writer(mesh, path)
    writer._add_ds_arrays(mesh, force_int32=True)  # noqa: SLF001
    filters = {
        name: writer._resolve_filter(  # noqa: SLF001
            name,
            arr,
            shuffle=False,
            mesh_filters="auto" if automatic else True,
            level=3,
        )
        for name, arr in writer._arrays.items()  # noqa: SLF001
    }
    return writer, filters


def test_triangle_polydata_uses_both_mesh_filters(polydata: pv.PolyData, tmp_path: Path) -> None:
    """Point coordinates and triangle connectivity select their dedicated filters."""
    writer, filters = _writer_filters(polydata, tmp_path / "unused.pv", automatic=True)
    ds_id = next(iter(writer._point_names))[:16]  # noqa: SLF001
    assert filters[f"{ds_id}{POINTS_KEY}"] == _FILTER_COMPONENT_SHUFFLE
    assert filters[f"{ds_id}{POLYS}{CONNECTIVITY_SUFFIX}"] == _FILTER_TRIANGLE_DELTA_SHUFFLE


def test_default_triangle_roundtrip_uses_version_3_and_is_smaller(polydata: pv.PolyData, tmp_path: Path) -> None:
    """Automatic mesh filters round-trip and beat fixed-width-only storage."""
    optimized = tmp_path / "optimized.pv"
    baseline = tmp_path / "baseline.pv"
    pz.write(polydata, optimized)
    pz.write(polydata, baseline, mesh_filters=False)

    result = pz.read(optimized)
    assert result == polydata
    assert result.points.tobytes() == polydata.points.tobytes()
    result_connectivity = vtk_to_numpy(result.GetPolys().GetConnectivityArray())
    source_connectivity = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray()).astype(np.int32)
    assert result_connectivity.tobytes() == source_connectivity.tobytes()
    assert pz.Reader(optimized)._metadata.file_version == FILE_VERSION_MESH_FILTERS  # noqa: SLF001
    assert pz.Reader(baseline)._metadata.file_version == FILE_VERSION_FIXED_WIDTH_CELLS  # noqa: SLF001
    assert optimized.stat().st_size < baseline.stat().st_size


def test_mesh_filters_false_preserves_version_2(polydata: pv.PolyData, tmp_path: Path) -> None:
    """Callers can retain the fixed-width-only format when compatibility requires it."""
    path = tmp_path / "version2.pv"
    pz.write(polydata, path, mesh_filters=False)
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_FIXED_WIDTH_CELLS  # noqa: SLF001
    assert pz.read(path) == polydata


def test_auto_skips_mesh_filters_for_incompressible_samples(tmp_path: Path) -> None:
    """Automatic selection falls back when transformed bytes are not smaller."""
    rng = np.random.default_rng(4)
    points = rng.integers(0, 256, size=(20000, 12), dtype=np.uint8).view(np.float32)
    triangles = rng.integers(0, 2**32, size=60000, dtype=np.uint32)
    assert not _special_filter_beneficial(points, _FILTER_COMPONENT_SHUFFLE, _FILTER_NONE, 3)
    assert not _special_filter_beneficial(triangles, _FILTER_TRIANGLE_DELTA_SHUFFLE, _FILTER_NONE, 3)

    path = tmp_path / "incompressible.pv"
    pointset = pv.PointSet(points)
    pz.write(pointset, path)
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    assert pz.read(path).points.tobytes() == pointset.points.tobytes()


def test_all_triangle_ugrid_uses_triangle_filter(tmp_path: Path) -> None:
    """The connectivity transform also applies to all-triangle unstructured grids."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    cells = np.array([3, 0, 1, 2, 3, 1, 3, 2])
    types = np.full(2, pv.CellType.TRIANGLE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, points)
    writer, filters = _writer_filters(grid, tmp_path / "unused.pv")
    ds_id = next(iter(writer._point_names))[:16]  # noqa: SLF001
    assert filters[f"{ds_id}{CELLS}{CONNECTIVITY_SUFFIX}"] == _FILTER_TRIANGLE_DELTA_SHUFFLE

    path = tmp_path / "triangles.pv"
    pz.write(grid, path, mesh_filters=True)
    result = pz.read(path)
    assert result == grid
    assert vtk_to_numpy(result.GetCells().GetConnectivityArray()).dtype == np.int32


def test_equal_width_mixed_types_do_not_use_triangle_filter(tmp_path: Path) -> None:
    """Triplet width alone does not imply that every unstructured cell is a triangle."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
        dtype=np.float32,
    )
    cells = np.array([3, 0, 1, 2, 3, 3, 4, 5])
    types = np.array([pv.CellType.TRIANGLE, pv.CellType.POLY_LINE], dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, points)
    writer, filters = _writer_filters(grid, tmp_path / "unused.pv")
    ds_id = next(iter(writer._point_names))[:16]  # noqa: SLF001
    assert filters[f"{ds_id}{CELLS}{CONNECTIVITY_SUFFIX}"] == _FILTER_NONE


@pytest.mark.parametrize(
    ("dtype", "shape", "message"),
    [(np.float32, (6,), "Invalid triangle-delta"), (np.int32, (5,), "Invalid triangle-delta")],
)
def test_triangle_delta_decode_rejects_invalid_metadata(dtype: type, shape: tuple[int, ...], message: str) -> None:
    """Malformed triangle metadata fails instead of returning corrupt connectivity."""
    buf = np.zeros(24, dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        _untriangle_delta_shuffle_bytes(buf, np.dtype(dtype), shape)


def test_component_decode_rejects_invalid_metadata() -> None:
    """A component filter cannot be applied to a scalar-shaped frame."""
    with pytest.raises(ValueError, match="Invalid component-shuffle shape"):
        _uncomponent_shuffle_bytes(b"", np.dtype(np.float32), (3,))


def test_future_mesh_filter_version_is_rejected(
    polydata: pv.PolyData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older reader rejects a file stamped with a newer mesh-filter format."""
    path = tmp_path / "future.pv"
    monkeypatch.setattr(_pz_mod, "FILE_VERSION_MESH_FILTERS", pz.FILE_VERSION + 1)
    pz.write(polydata, path, mesh_filters=True)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="newer than the version supported"):
        pz.read(path)
