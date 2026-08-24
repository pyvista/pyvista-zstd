"""
Hold the reader to exact reconstruction of what the writer was given.

Requires the core, and does not skip without it. ``PVZSTD_LIBRARY`` chooses
which build answers; failing that, the one installed beside the package. The
comparison is exact: a round trip through the format is lossless by
construction, so any difference in a value is a defect rather than a
tolerance question.

The oracle is the source dataset the writer was handed, so the writer is held
to account too. ``test_cpp_core_is_actually_used`` is the control: a test that
never reached the core would pass for the wrong reason.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import pyvista as pv
import vtk

import pyvista_zstd as pz
from pyvista_zstd import _capi

# No availability gate. The core is the only implementation, so a machine that
# cannot load it cannot read these files at all -- skipping here would report
# success for a package that does not work.

HEX = 8
TET = 4


def _sphere() -> pv.DataSet:
    rng = np.random.default_rng(21)
    ds = pv.Sphere(theta_resolution=10, phi_resolution=10)
    ds.point_data["scal_f64"] = rng.random(ds.n_points)
    ds.point_data["vec_f32"] = rng.random((ds.n_points, 3)).astype(np.float32)
    ds.cell_data["ids_i64"] = np.arange(ds.n_cells, dtype=np.int64)
    ds.cell_data["flag_u8"] = np.ones(ds.n_cells, dtype=np.uint8)
    ds.field_data["note_i32"] = np.arange(5, dtype=np.int32)
    return ds


def _unstructured() -> pv.DataSet:
    ds = pv.ImageData(dimensions=(9, 9, 9)).cast_to_unstructured_grid()
    ds.point_data["t_f64"] = np.linspace(0.0, 1.0, ds.n_points)
    return ds


def _image() -> pv.DataSet:
    ds = pv.ImageData(dimensions=(5, 6, 7), spacing=(0.5, 0.25, 2.0))
    ds.point_data["t_f64"] = np.linspace(-1.0, 1.0, ds.n_points)
    ds.cell_data["c_i32"] = np.arange(ds.n_cells, dtype=np.int32)
    return ds


def _structured() -> pv.DataSet:
    x, y, z = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 2, 4), np.linspace(0, 3, 3), indexing="ij")
    ds = pv.StructuredGrid(x, y, z)
    ds.point_data["s_f64"] = np.linspace(0.0, 1.0, ds.n_points)
    return ds


def _mixed() -> pv.DataSet:
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [2, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
        ],
        dtype=float,
    )
    cells = np.hstack([[HEX, 0, 1, 2, 3, 4, 5, 6, 7], [TET, 1, 8, 9, 10]])
    ctypes_ = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_TETRA], dtype=np.uint8)
    ds = pv.UnstructuredGrid(cells, ctypes_, pts)
    ds.point_data["p_f64"] = np.arange(ds.n_points, dtype=np.float64)
    return ds


DATASETS = {
    "polydata": _sphere,
    "unstructured": _unstructured,
    "image": _image,
    "structured": _structured,
    "mixed": _mixed,
}


def _assert_same_dataset(a: pv.DataSet, b: pv.DataSet) -> int:
    """Compare two datasets exactly. Returns how many arrays were compared."""
    assert type(a) is type(b)
    assert a.n_points == b.n_points
    assert a.n_cells == b.n_cells

    compared = 0
    if hasattr(a, "points") and a.n_points:
        assert np.array_equal(a.points, b.points)
        compared += 1
    if hasattr(a, "cells") and a.n_cells:
        assert np.array_equal(a.cells, b.cells)
        assert np.array_equal(a.celltypes, b.celltypes)
        compared += 1

    for attr in ("point_data", "cell_data", "field_data"):
        left, right = getattr(a, attr), getattr(b, attr)
        assert set(left.keys()) == set(right.keys()), attr
        for key in left:
            assert left[key].dtype == right[key].dtype, (attr, key)
            assert np.array_equal(left[key], right[key]), (attr, key)
            compared += 1

    assert compared > 0, "no arrays were compared -- the test proved nothing"
    return compared


@pytest.mark.parametrize("label", sorted(DATASETS))
@pytest.mark.parametrize("shuffle", [False, True, "auto"])
def test_roundtrip_reconstructs_source(tmp_path, label, shuffle) -> None:
    """What comes back out is exactly what went in."""
    source = DATASETS[label]()
    path = tmp_path / f"{label}.pv"
    pz.write(source, path, shuffle=shuffle, progress_bar=False)

    assert _assert_same_dataset(source, pz.Reader(path).read()) > 0


def test_cpp_core_is_actually_used(tmp_path, monkeypatch) -> None:
    """
    Control: crippling the C++ path must break the C++ core.

    Without this the round-trip tests above could be satisfied by something
    other than the core and pass for the wrong reason.
    """
    path = tmp_path / "control.pv"
    pz.write(_sphere(), path, shuffle=True, progress_bar=False)

    def _sabotage(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        msg = "C++ read path deliberately broken"
        raise RuntimeError(msg)

    # read_arrays, not read_at: the batch entry point is what the backend calls.
    # This control caught its own staleness when the reader moved to batching.
    monkeypatch.setattr(_capi.CoreReader, "read_arrays", _sabotage)

    with pytest.raises(RuntimeError, match="deliberately broken"):
        pz.Reader(path).read()


def test_shuffle_filter_is_exercised_in_cpp(tmp_path) -> None:
    """
    Control: the C++ unshuffle branch must actually run.

    ``shuffle`` defaults to off, so a parity test over default files never
    enters the filter branch at all.
    """
    path = tmp_path / "filtered.pv"
    pz.write(_sphere(), path, shuffle=True, progress_bar=False)

    with _capi.CoreReader(path) as reader:
        filters = [reader._info(i).filter_id for i in range(len(reader))]  # noqa: SLF001

    assert any(f == 1 for f in filters), "no array carried the shuffle filter"


def _call_discarding_result(fn, path) -> None:
    """
    Invoke ``fn`` for its warning alone.

    ``read_array`` is handed a name the fixture does not carry and raises; the
    warning is emitted before the lookup fails, and it is what is under test.
    """
    with contextlib.suppress(KeyError):
        fn(path)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda p: pz.read(p, backend="python"), id="read"),
        pytest.param(lambda p: pz.Reader(p, backend="cpp"), id="Reader"),
        pytest.param(lambda p: pz.read_array(p, "nope", backend="auto"), id="read_array"),
        pytest.param(lambda p: pz.AppendReader(p, backend="auto"), id="AppendReader"),
    ],
)
def test_backend_argument_is_deprecated(tmp_path, call) -> None:
    """
    Every public entry point that took ``backend`` now warns and ignores it.

    The value passed is deliberately not one the reader could honour, so the
    read must succeed on whatever the reader chose.
    """
    path = tmp_path / "deprecated.pv"
    pz.write(_sphere(), path, progress_bar=False)
    with pytest.warns(DeprecationWarning, match="no longer has any effect"):
        _call_discarding_result(call, path)


def test_array_downselection_keeps_only_what_was_asked_for(tmp_path) -> None:
    """Downselecting drops the rest and leaves the kept array untouched."""
    source = _sphere()
    path = tmp_path / "subset.pv"
    pz.write(source, path, progress_bar=False)

    reader = pz.Reader(path)
    reader.selected_point_arrays = {"scal_f64"}
    reader.selected_cell_arrays = set()
    got = reader.read()

    assert set(got.point_data.keys()) == {"scal_f64"}
    assert not set(got.cell_data.keys())
    # The survivor is unchanged; downselection must not perturb what it keeps.
    assert np.array_equal(got.point_data["scal_f64"], source.point_data["scal_f64"])


def test_cpp_reader_survives_close_and_reports_it(tmp_path) -> None:
    """Using a closed reader raises rather than reading freed memory."""
    path = tmp_path / "closed.pv"
    pz.write(_sphere(), path, progress_bar=False)

    reader = _capi.CoreReader(path)
    reader.close()
    reader.close()  # idempotent
    with pytest.raises(ValueError, match="closed CoreReader"):
        reader.names()


def test_missing_file_reports_io_not_crash(tmp_path) -> None:
    """A missing file is a clean error from the C++ core."""
    with pytest.raises(_capi.PvzstdError, match="I/O error"):
        _capi.CoreReader(tmp_path / "nope.pv")


def _esgrid() -> pv.DataSet:
    """
    Build an explicit structured grid carrying the arrays its own cast needs.

    ``BLOCK_I``/``BLOCK_J``/``BLOCK_K`` are what turn an unstructured grid
    back into this type.
    """
    ni, nj, nk = 2, 3, 4
    xc, yc, zc = (np.arange(n + 1, dtype=float) for n in (ni, nj, nk))
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
    grid = pv.StructuredGrid(xx, yy, zz).cast_to_unstructured_grid()

    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    grid.cell_data["BLOCK_I"] = i.ravel(order="F").astype(np.int32)
    grid.cell_data["BLOCK_J"] = j.ravel(order="F").astype(np.int32)
    grid.cell_data["BLOCK_K"] = k.ravel(order="F").astype(np.int32)
    grid.cell_data["payload"] = np.arange(grid.n_cells, dtype=np.float64)
    return grid.cast_to_explicit_structured_grid()


def test_explicit_structured_grid_round_trips(tmp_path) -> None:
    """
    An ExplicitStructuredGrid survives the container.

    The cast is driven by the blocking arrays, so they must be attached to the
    rebuilt unstructured grid before it runs.
    """
    ds = _esgrid()
    path = tmp_path / "esgrid.pv"
    pz.write(ds, path, progress_bar=False)

    got = pz.Reader(path).read()

    assert type(got) is type(ds), "round-trip changed the dataset type"
    assert _assert_same_dataset(ds, got) > 0
    for key in ("BLOCK_I", "BLOCK_J", "BLOCK_K", "payload"):
        assert np.array_equal(got.cell_data[key], ds.cell_data[key]), key
