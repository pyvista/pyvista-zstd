"""
Hold the native reader backend to agreement with the pure-Python one.

Skipped unless ``PVZSTD_LIBRARY`` points at a built shared library, or one is
installed beside the package. The comparison is exact: the native path is a
different implementation of the same format, not an approximation of it, so
any difference in a value is a defect rather than a tolerance question.

The tests also assert that the native path was actually *taken*. ``backend
="auto"`` falls back silently by design, which is right for users and
dangerous for a test suite -- a green run proves nothing if every case
quietly ran the Python implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
import vtk

import pyvista_zstd as pz
from pyvista_zstd import _capi

pytestmark = pytest.mark.skipif(
    not _capi.available(),
    reason="set PVZSTD_LIBRARY to a built libpvzstd to run native backend parity",
)

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
def test_native_matches_python(tmp_path, label, shuffle) -> None:
    """Both backends reconstruct exactly the same dataset."""
    path = tmp_path / f"{label}.pv"
    pz.write(DATASETS[label](), path, shuffle=shuffle, progress_bar=False)

    from_python = pz.read(path, backend="python")
    from_native = pz.read(path, backend="native")
    assert _assert_same_dataset(from_python, from_native) > 0


def test_native_backend_is_actually_used(tmp_path, monkeypatch) -> None:
    """
    Control: crippling the native path must break the native backend.

    Without this the parity tests above could all be running the pure-Python
    implementation twice and passing for the wrong reason.
    """
    path = tmp_path / "control.pv"
    pz.write(_sphere(), path, shuffle=True, progress_bar=False)

    def _sabotage(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
        msg = "native read path deliberately broken"
        raise RuntimeError(msg)

    # read_arrays, not read_at: the batch entry point is what the backend
    # actually calls. This control caught its own staleness when the reader
    # moved to batched decompression and the single-array patch stopped
    # reddening -- which is the whole reason to keep a control that must fail.
    monkeypatch.setattr(_capi.NativeReader, "read_arrays", _sabotage)

    with pytest.raises(RuntimeError, match="deliberately broken"):
        pz.read(path, backend="native")

    # The pure-Python backend must be untouched by the sabotage, which is what
    # proves the two paths are genuinely separate.
    assert pz.read(path, backend="python").n_points > 0


def test_shuffle_filter_is_exercised_natively(tmp_path) -> None:
    """
    Control: the native unshuffle branch must actually run.

    ``shuffle`` defaults to off, so a parity test over default files never
    enters the filter branch at all.
    """
    path = tmp_path / "filtered.pv"
    pz.write(_sphere(), path, shuffle=True, progress_bar=False)

    with _capi.NativeReader(path) as reader:
        filters = [reader._info(i).filter_id for i in range(len(reader))]  # noqa: SLF001

    assert any(f == 1 for f in filters), "no array carried the shuffle filter"


def test_unknown_backend_is_refused(tmp_path) -> None:
    """A misspelt backend fails loudly rather than falling back."""
    path = tmp_path / "x.pv"
    pz.write(_sphere(), path, progress_bar=False)
    with pytest.raises(ValueError, match="backend must be one of"):
        pz.read(path, backend="c++")


def test_array_downselection_matches(tmp_path) -> None:
    """Selecting a subset of arrays gives the same result on both backends."""
    path = tmp_path / "subset.pv"
    pz.write(_sphere(), path, progress_bar=False)

    def _read(backend: str) -> pv.DataSet:
        reader = pz.Reader(path, backend=backend)
        reader.selected_point_arrays = {"scal_f64"}
        reader.selected_cell_arrays = set()
        return reader.read()

    native, python = _read("native"), _read("python")
    assert set(native.point_data.keys()) == {"scal_f64"}
    _assert_same_dataset(python, native)


def test_native_reader_survives_close_and_reports_it(tmp_path) -> None:
    """Using a closed reader raises rather than reading freed memory."""
    path = tmp_path / "closed.pv"
    pz.write(_sphere(), path, progress_bar=False)

    reader = _capi.NativeReader(path)
    reader.close()
    reader.close()  # idempotent
    with pytest.raises(ValueError, match="closed NativeReader"):
        reader.names()


def test_missing_file_reports_io_not_crash(tmp_path) -> None:
    """A missing file is a clean error from the native core."""
    with pytest.raises(_capi.PvzstdError, match="I/O error"):
        _capi.NativeReader(tmp_path / "nope.pv")
