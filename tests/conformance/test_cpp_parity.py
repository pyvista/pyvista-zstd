"""
Hold the C++ reader to the same oracle as the format spec.

Skipped unless ``PVZSTD_DUMP`` points at a built ``pvzstd_dump`` binary, so the
Python suite still runs on a checkout with no C++ toolchain. CI sets it after
building ``cpp/``.

The comparison is bit-exact: both sides emit a checksum over the *decoded*
payload, so a reader that gets shapes and dtypes right but mis-decodes bytes
still fails.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import pyvista as pv
import ref_reader

import pyvista_zstd as pz

DUMP = os.environ.get("PVZSTD_DUMP")

pytestmark = pytest.mark.skipif(
    not DUMP or not Path(DUMP).exists(),
    reason="set PVZSTD_DUMP to a built cpp/ pvzstd_dump binary to run C++ parity",
)

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = 0xFFFFFFFFFFFFFFFF


def _fnv1a(data: bytes) -> int:
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def _oracle_dump(path: Path) -> list[str]:
    container = ref_reader.read(path)
    lines = [
        f"arrays {len(container.arrays)}",
        f"ds_metadata {1 if container.ds_metadata is not None else 0}",
        f"file_metadata {1 if container.file_metadata is not None else 0}",
    ]
    for header in container.headers:
        arr = container.arrays.get(header.name)
        if arr is None:
            continue
        shape = ",".join(str(s) for s in header.shape)
        lines.append(
            f"array {header.name} {header.dtype.str} filter={header.filter_id} "
            f"nbytes={arr.nbytes} shape=[{shape}] fnv={_fnv1a(arr.tobytes()):016x}"
        )
    return lines


def _cpp_dump(path: Path) -> list[str]:
    proc = subprocess.run([DUMP, str(path)], capture_output=True, text=True, check=True)  # noqa: S603
    return proc.stdout.splitlines()


def _sphere() -> pv.DataSet:
    rng = np.random.default_rng(7)
    ds = pv.Sphere(theta_resolution=12, phi_resolution=12)
    ds.point_data["scal_f32"] = np.arange(ds.n_points, dtype=np.float32)
    ds.point_data["vec_f64"] = rng.random((ds.n_points, 3))
    ds.cell_data["ids_i64"] = np.arange(ds.n_cells, dtype=np.int64)
    return ds


def _unstructured() -> pv.DataSet:
    rng = np.random.default_rng(8)
    ds = pv.ImageData(dimensions=(16, 16, 16)).cast_to_unstructured_grid()
    ds.point_data["smooth_f64"] = np.linspace(0.0, 1.0, ds.n_points)
    ds.point_data["noisy_f32"] = rng.random(ds.n_points).astype(np.float32)
    ds.point_data["ramp_i32"] = np.arange(ds.n_points, dtype=np.int32)
    ds.point_data["flag_u8"] = np.ones(ds.n_points, dtype=np.uint8)
    return ds


def _image() -> pv.DataSet:
    ds = pv.ImageData(dimensions=(6, 7, 8))
    ds.point_data["t_f64"] = np.linspace(-1.0, 1.0, ds.n_points)
    return ds


def _structured() -> pv.DataSet:
    x, y, z = np.meshgrid(np.arange(4.0), np.arange(5.0), np.arange(6.0))
    return pv.StructuredGrid(x, y, z)


DATASETS = {
    "polydata": _sphere,
    "unstructured": _unstructured,
    "image": _image,
    "structured": _structured,
}


@pytest.mark.parametrize("label", sorted(DATASETS))
@pytest.mark.parametrize("shuffle", [False, True, "auto"])
def test_cpp_reader_matches_oracle(tmp_path, label, shuffle) -> None:
    """Every array decodes identically in C++ and in the spec-only reader."""
    path = tmp_path / f"{label}.pv"
    pz.write(DATASETS[label](), path, shuffle=shuffle, progress_bar=False)
    assert _cpp_dump(path) == _oracle_dump(path)


def test_cpp_reader_rejects_a_truncated_file(tmp_path) -> None:
    """A damaged container must fail loudly rather than return partial data."""
    path = tmp_path / "whole.pv"
    pz.write(_sphere(), path, progress_bar=False)
    truncated = tmp_path / "truncated.pv"
    truncated.write_bytes(path.read_bytes()[:-32])

    proc = subprocess.run([DUMP, str(truncated)], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode != 0, "truncated container was accepted"
