"""
Hold the C++ writer to byte-for-byte agreement with the reference writer.

Skipped unless ``PVZSTD_REWRITE`` points at a built ``pvzstd_rewrite`` binary. That
tool reads a container with the C++ reader and writes it back with the C++
writer, so one run exercises both halves and the comparison is against real
reference bytes rather than against our own idea of what they should be.

Byte-identity is a deliberately harsher gate than round-trip fidelity.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import pyvista as pv
import ref_reader
import vtk

import pyvista_zstd as pz

REWRITE = os.environ.get("PVZSTD_REWRITE")

pytestmark = pytest.mark.skipif(
    not REWRITE or not Path(REWRITE).exists(),
    reason="set PVZSTD_REWRITE to a built cpp/ pvzstd_rewrite binary to run writer parity",
)

UID_N_CHAR = 16
SHUFFLE_CODE = {False: 0, True: 1, "auto": 2}
# PVZSTD_THREADS_AUTO: exercises our replication of the reference's worker-count
# rule rather than side-stepping it by passing the resolved number.
THREADS_AUTO = "-2"

HEX = 8
TET = 4


def _sphere() -> pv.DataSet:
    rng = np.random.default_rng(11)
    ds = pv.Sphere(theta_resolution=12, phi_resolution=12)
    ds.point_data["scal_f32"] = np.arange(ds.n_points, dtype=np.float32)
    ds.point_data["vec_f64"] = rng.random((ds.n_points, 3))
    ds.cell_data["ids_i64"] = np.arange(ds.n_cells, dtype=np.int64)
    return ds


def _unstructured() -> pv.DataSet:
    rng = np.random.default_rng(12)
    ds = pv.ImageData(dimensions=(16, 16, 16)).cast_to_unstructured_grid()
    ds.point_data["smooth_f64"] = np.linspace(0.0, 1.0, ds.n_points)
    ds.point_data["noisy_f32"] = rng.random(ds.n_points).astype(np.float32)
    ds.point_data["flag_u8"] = np.ones(ds.n_points, dtype=np.uint8)
    return ds


def _image() -> pv.DataSet:
    ds = pv.ImageData(dimensions=(6, 7, 8))
    ds.point_data["t_f64"] = np.linspace(-1.0, 1.0, ds.n_points)
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
    ctypes = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_TETRA], dtype=np.uint8)
    return pv.UnstructuredGrid(cells, ctypes, pts)


def _bulk() -> pv.DataSet:
    """
    Big enough that the worker-count rule actually resolves to a worker.

    Two thresholds must be cleared at once: 2 MiB, above which the reference
    asks for a worker at all, and 1 MiB per frame, above which zstd's threaded
    and unthreaded output stop being byte-identical. Every other fixture here
    is well under a megabyte, so both sides compress single-threaded.
    """
    rng = np.random.default_rng(21)
    ds = pv.ImageData(dimensions=(64, 64, 64))
    ds.point_data["bulk_f64"] = rng.random(ds.n_points)
    return ds


DATASETS = {
    "polydata": _sphere,
    "unstructured": _unstructured,
    "image": _image,
    "mixed": _mixed,
    "bulk": _bulk,
}


def _rewrite(src: Path, dst: Path, *, shuffle: bool | str) -> None:
    container = ref_reader.read(src)
    uid = container.file_metadata["frame_names"][0][:UID_N_CHAR]
    fixed = "1" if container.ds_metadata.get("fixed_cell_sizes") else "0"
    level = str(container.file_metadata["compression_level"])
    subprocess.run(  # noqa: S603
        [
            REWRITE,
            str(src),
            str(dst),
            level,
            THREADS_AUTO,
            str(SHUFFLE_CODE[shuffle]),
            fixed,
            uid,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("label", sorted(DATASETS))
@pytest.mark.parametrize("shuffle", [False, True, "auto"])
def test_cpp_writer_is_byte_identical(tmp_path, label, shuffle) -> None:
    """The C++ writer reproduces the reference writer's file exactly."""
    src = tmp_path / f"{label}.pv"
    dst = tmp_path / f"{label}.rewritten.pv"
    pz.write(DATASETS[label](), src, shuffle=shuffle, progress_bar=False)
    _rewrite(src, dst, shuffle=shuffle)

    original = src.read_bytes()
    rewritten = dst.read_bytes()
    if original != rewritten:  # pragma: no cover - failure path
        first = next(
            (i for i, (a, b) in enumerate(zip(original, rewritten, strict=False)) if a != b),
            min(len(original), len(rewritten)),
        )
        pytest.fail(
            f"{label} shuffle={shuffle}: {len(original)} vs {len(rewritten)} bytes, first difference at byte {first}"
        )


def test_empty_arrays_still_record_their_filter(tmp_path) -> None:
    """
    Regression: an empty multibyte array is *recorded* as filtered.

    Shuffling nothing is a no-op, so only the one header byte reveals this.
    """
    src = tmp_path / "polydata.pv"
    pz.write(_sphere(), src, shuffle=True, progress_bar=False)
    container = ref_reader.read(src)

    empty_multibyte = [
        h
        for h in container.headers
        if h.name in container.arrays
        and container.arrays[h.name].size == 0
        and container.arrays[h.name].dtype.itemsize > 1
    ]
    assert empty_multibyte, "fixture no longer contains an empty multibyte array"
    assert all(h.filter_id == ref_reader.FILTER_SHUFFLE for h in empty_multibyte)
