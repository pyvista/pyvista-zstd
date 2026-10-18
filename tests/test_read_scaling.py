"""Reading a MultiBlock must cost O(frames), not O(datasets x frames)."""

from __future__ import annotations

import cProfile
import pstats
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd
from pyvista_zstd import _capi
from pyvista_zstd.pyvista_zstd import UID_N_CHAR

if TYPE_CHECKING:
    from pathlib import Path


LINEAR_BOUND = 5.0


def _multiblock(n_blocks: int) -> pv.MultiBlock:
    """Return *n_blocks* small spheres, each with a point and a cell array."""
    mb = pv.MultiBlock()
    for i in range(n_blocks):
        sphere = pv.Sphere(theta_resolution=6, phi_resolution=6, center=(i, 0, 0))
        sphere.point_data["d"] = np.arange(sphere.n_points, dtype=float)
        sphere.cell_data["c"] = np.arange(sphere.n_cells)
        mb.append(sphere, f"block-{i}")
    return mb


def _calls_to_read(path: Path) -> int:
    """Count every Python-level call made while reading *path*."""
    profiler = cProfile.Profile()
    profiler.enable()
    pyvista_zstd.read(path)
    profiler.disable()
    return pstats.Stats(profiler).total_calls


def test_read_call_count_scales_linearly(tmp_path: Path) -> None:
    """Quadrupling the block count must not more than quadruple the work (with slack)."""
    small, large = 20, 80
    paths = {}
    for n in (small, large):
        paths[n] = tmp_path / f"{n}.pv"
        pyvista_zstd.write(_multiblock(n), paths[n])

    ratio = _calls_to_read(paths[large]) / _calls_to_read(paths[small])
    # Linear scaling sits near 4x; a lookup that walked every frame per dataset measured 8x.
    assert ratio < LINEAR_BOUND, f"read work grew {ratio:.1f}x for {large / small:.0f}x the blocks"


def test_array_catalog_decoded_once_per_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The core's array headers are decoded once, not once per dataset."""
    n_blocks = 10
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(_multiblock(n_blocks), path)

    original = _capi.CoreReader._array_catalog  # noqa: SLF001
    cold_builds = 0

    def counting(self: _capi.CoreReader) -> object:
        nonlocal cold_builds
        if self._catalog is None:
            cold_builds += 1
        return original(self)

    monkeypatch.setattr(_capi.CoreReader, "_array_catalog", counting)
    mb = pyvista_zstd.read(path)
    assert mb.n_blocks == n_blocks
    assert cold_builds == 1


def test_selected_frame_names_only_own_frames(tmp_path: Path) -> None:
    """A dataset's selection holds its own frames only, found without scanning the rest."""
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(_multiblock(5), path)
    with pyvista_zstd.Reader(path) as reader:
        groups = reader._frame_names_by_ds()  # noqa: SLF001
        uid = reader[2].uid
        selected = reader._selected_frame_names(uid)  # noqa: SLF001
        assert selected == set(groups[uid])
        assert all(name.startswith(uid) for name in selected)
        assert sum(len(names) for names in groups.values()) == len(reader._metadata.frame_names)  # noqa: SLF001


@pytest.mark.parametrize("n_threads", [None, 0])
def test_multiblock_round_trip_matches(tmp_path: Path, n_threads: int | None) -> None:
    """Blocks come back equal, with their names and active arrays, through the VTK-owned path."""
    mb = _multiblock(4)
    mb[1].point_data.active_scalars_name = "d"
    mb[2].cell_data.active_scalars_name = "c"
    mb.append(None, "empty")
    mb.append(mb[0], "again")
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(mb, path)

    out = pyvista_zstd.read(path, n_threads=n_threads) if n_threads is not None else pyvista_zstd.read(path)
    assert out.keys() == mb.keys()
    assert out[4] is None
    for i in range(4):
        assert out[i] == mb[i]
        assert out[i].point_data.active_scalars_name == mb[i].point_data.active_scalars_name
        assert out[i].cell_data.active_scalars_name == mb[i].cell_data.active_scalars_name
    assert out["again"] is out[0]


def test_repeated_block_name_keeps_name(tmp_path: Path) -> None:
    """A name used twice replaces the first block and keeps the name, as ``mb[name] = ds`` does."""
    mb = _multiblock(2)
    mb.append(pv.Cube(), "block-0")
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(mb, path)

    out = pyvista_zstd.read(path)
    assert out.keys() == ["block-0", "block-1"]
    assert out[0] == pv.Cube()


def test_unsupported_vtk_dtypes_fall_back_to_numpy(tmp_path: Path) -> None:
    """Arrays VTK has no native type for still round-trip through NumPy."""
    sphere = pv.Sphere(theta_resolution=6, phi_resolution=6)
    sphere.point_data["flag"] = np.zeros(sphere.n_points, dtype=bool)
    sphere.point_data["big_endian"] = np.arange(sphere.n_points, dtype=">f8")
    sphere.cell_data["c"] = np.arange(sphere.n_cells, dtype=np.uint16)
    sphere.field_data["txt"] = ["a", "b"]
    sphere.field_data["bits"] = np.array([True, False, True])
    sphere.field_data["cplx"] = np.array([1 + 2j, 3 - 4j])
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(sphere, path)

    out = pyvista_zstd.read(path)
    assert out == sphere
    assert out.point_data["flag"].dtype == bool
    assert out.cell_data["c"].dtype == np.uint16
    assert list(out.field_data["txt"]) == ["a", "b"]
    assert out.field_data["bits"].dtype == bool
    assert out.field_data["cplx"].dtype == np.complex128


def test_read_arrays_rejects_bad_allocation(tmp_path: Path) -> None:
    """An allocator that hands back the wrong buffer is refused, not written into."""
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(pv.Sphere(), path)
    with _capi.CoreReader(path) as reader:
        names = reader.names()

        def wrong(_name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
            return np.empty((shape[0] + 1, *shape[1:]), dtype=dtype)

        with pytest.raises(ValueError, match="unusable destination"):
            reader.read_arrays(keep={names[0]}, allocate=wrong)


@pytest.mark.skipif(pv.vtk_version_info >= (9, 8), reason="VTK 9.8 drops stateful wrappers for free")
def test_segments_release_empties_vtk_wrappers() -> None:
    """Allocated VTK arrays are handed back with no Python-side state, so dropping them is free."""
    from pyvista_zstd.pyvista_zstd import _Segments  # noqa: PLC0415

    segments = _Segments()
    view = segments.allocate("0000000000000000points", (4, 3), np.dtype(np.float64))
    view[:] = 1.0
    (vtk_array,) = segments.vtk.values()
    vtk_array.cached = "state"  # what pyvista's mixins leave on a wrapper
    assert vtk_array.__dict__

    segments.release()
    assert not vtk_array.__dict__
    assert not segments.vtk
    assert np.asarray(vtk_array).shape == (4, 3)
    assert (view == 1.0).all()


def test_empty_cell_frames_are_skipped_only_for_polydata(tmp_path: Path) -> None:
    """A cell-less UnstructuredGrid keeps its cell frames, which older readers require."""
    pyvista_zstd.write(pv.UnstructuredGrid(), tmp_path / "ugrid.pv")
    pyvista_zstd.write(pv.Sphere(), tmp_path / "sphere.pv")
    with _capi.CoreReader(tmp_path / "ugrid.pv") as reader:
        names = {name[UID_N_CHAR:] for name in reader.names()}
    assert {"cells_offset", "cells_connectivity"} <= names
    with _capi.CoreReader(tmp_path / "sphere.pv") as reader:
        names = {name[UID_N_CHAR:] for name in reader.names()}
    assert "polys_connectivity" in names
    assert not {f"{kind}_connectivity" for kind in ("lines", "verts", "strips")} & names


def test_unnamed_vtk_arrays_keep_pyvista_names(tmp_path: Path) -> None:
    """An array added to the VTK attributes without a name round-trips under pyvista's name for it."""
    try:  # pyvista >= 0.48
        from pyvista._vtk import vtkFloatArray  # noqa: PLC0415
    except ModuleNotFoundError:  # pyvista < 0.48
        from pyvista.core._vtk_core import vtkFloatArray  # noqa: PLC0415

    sphere = pv.Sphere(theta_resolution=6, phi_resolution=6)
    nameless = vtkFloatArray()
    nameless.SetNumberOfTuples(sphere.n_points)
    nameless.Fill(1.0)
    sphere.GetPointData().AddArray(nameless)
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(sphere, path)
    out = pyvista_zstd.read(path)
    assert out.point_data.keys() == sphere.point_data.keys()
    assert "None" not in out.point_data
