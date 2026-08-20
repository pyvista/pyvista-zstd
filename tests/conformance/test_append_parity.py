"""
Hold the C++ append to byte-for-byte agreement with the reference append.

Skipped unless ``PVZ_APPEND`` points at a built ``pvz_append`` binary.

Appending is a harder parity problem than writing, because most of the output
is not produced at all -- it is copied verbatim from the source file -- and the
part that *is* produced has to slot into a document another library wrote. Two
things here would pass a round-trip test and fail this one: compressing the new
frames with a worker pool (the reference append uses none, unlike the writer),
and regenerating the dataset metadata instead of splicing into it.

The gate is deliberately paired with a negative control. A byte-comparison that
cannot be made to fail is not evidence, and this one is easy to get wrong in a
direction that looks green: the appended arrays are small, so a mistake in the
metadata could be swamped by identical payload bytes if the comparison were
scoped too narrowly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import pyvista as pv
import ref_reader

import pyvista_zstd as pz

APPEND = os.environ.get("PVZ_APPEND")

pytestmark = pytest.mark.skipif(
    not APPEND or not Path(APPEND).exists(),
    reason="set PVZ_APPEND to a built cpp/ pvz_append binary to run append parity",
)

SHUFFLE_CODE = {False: 0, True: 1, "auto": 2}
# PVZ_LEVEL_FROM_FILE: makes the tool read the level out of the container, the
# way ``level=None`` does on the reference side, rather than being handed the
# resolved number.
LEVEL_FROM_FILE = "-1000"
# One filtered, one not: the two outcomes "auto" must be able to reach.
DISTINCT_FILTER_DECISIONS = 2


def _sphere() -> pv.DataSet:
    ds = pv.Sphere(theta_resolution=12, phi_resolution=12)
    ds.point_data["scal_f32"] = np.arange(ds.n_points, dtype=np.float32)
    return ds


def _unstructured() -> pv.DataSet:
    rng = np.random.default_rng(3)
    ds = pv.ImageData(dimensions=(12, 12, 12)).cast_to_unstructured_grid()
    ds.point_data["smooth_f64"] = np.linspace(0.0, 1.0, ds.n_points)
    ds.point_data["noisy_f32"] = rng.random(ds.n_points).astype(np.float32)
    return ds


def _image() -> pv.DataSet:
    ds = pv.ImageData(dimensions=(6, 7, 8))
    ds.point_data["t_f64"] = np.linspace(-1.0, 1.0, ds.n_points)
    return ds


DATASETS = {"polydata": _sphere, "unstructured": _unstructured, "image": _image}


def _payloads() -> dict[str, np.ndarray]:
    """Arrays chosen to exercise every branch of the filter decision."""
    rng = np.random.default_rng(17)
    return {
        # Two multibyte floats that "auto" decides *differently*: measured, the
        # probe declines the smooth ramp and accepts the noise. Whichever way
        # round it goes, the point is that they disagree -- a cheaper heuristic
        # that keyed on dtype alone would give both the same answer and still
        # look green on a single-array fixture.
        "step_1_disp": np.linspace(-2.0, 5.0, 300).reshape(100, 3),
        "step_1_noise": rng.random(97).astype(np.float32),
        # integer: never a candidate under "auto", always one under True
        "step_1_ids": np.arange(64, dtype=np.int64),
        # itemsize 1: excluded by the dtype gate under every mode
        "step_1_flags": np.ones(41, dtype=np.uint8),
        # empty multibyte: shuffling nothing is a no-op, but the header byte
        # that records it is not
        "step_1_empty": np.zeros(0, dtype=np.float64),
        # Over 1 MiB, and that size is the whole point. Measured: zstd emits
        # identical bytes with threads=0 and threads=1 for anything smaller, so
        # a fixture of small arrays cannot tell the reference append's
        # single-threaded framing from a worker pool. Without this array the
        # suite stayed green when the worker count was deliberately broken.
        "step_1_bulk": rng.random(1 << 18),
    }


def _write_spec(tmp_path: Path, arrays: dict[str, np.ndarray], *, dtype_names=None) -> Path:
    """Write the tab-separated spec plus one raw file per array."""
    spec_lines = []
    for i, (name, arr) in enumerate(arrays.items()):
        contiguous = np.ascontiguousarray(arr)
        raw = tmp_path / f"raw_{i}.bin"
        raw.write_bytes(contiguous.tobytes())
        dtype_name = str(contiguous.dtype) if dtype_names is None else dtype_names[name]
        shape_csv = ",".join(str(d) for d in contiguous.shape)
        spec_lines.append(
            "\t".join([name, contiguous.dtype.str, dtype_name, shape_csv, str(raw)])
        )
    spec = tmp_path / "spec.tsv"
    spec.write_text("\n".join(spec_lines) + "\n")
    return spec


def _cpp_append(container: Path, spec: Path, *, shuffle) -> None:
    subprocess.run(  # noqa: S603
        [APPEND, str(container), LEVEL_FROM_FILE, str(SHUFFLE_CODE[shuffle]), str(spec)],
        capture_output=True,
        text=True,
        check=True,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("label", sorted(DATASETS))
@pytest.mark.parametrize("shuffle", [False, True, "auto"])
def test_cpp_append_is_byte_identical(tmp_path, label, shuffle) -> None:
    """The C++ append reproduces the reference append's file exactly."""
    seed = tmp_path / f"{label}.pv"
    pz.write(DATASETS[label](), seed, progress_bar=False)

    reference = tmp_path / "reference.pv"
    native = tmp_path / "native.pv"
    shutil.copyfile(seed, reference)
    shutil.copyfile(seed, native)

    arrays = _payloads()
    pz.append_arrays(reference, arrays, shuffle=shuffle)
    _cpp_append(native, _write_spec(tmp_path, arrays), shuffle=shuffle)

    expected = reference.read_bytes()
    actual = native.read_bytes()
    if expected != actual:  # pragma: no cover - failure path
        first = next(
            (i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
            min(len(expected), len(actual)),
        )
        pytest.fail(
            f"{label} shuffle={shuffle}: {len(expected)} vs {len(actual)} bytes, "
            f"first difference at byte {first}"
        )


def test_the_parity_gate_can_fail(tmp_path) -> None:
    """
    Negative control: a wrong dtype spelling must redden the comparison above.

    The format records a dtype *string* ("<f8") in the frame header and a dtype
    *name* ("float64") in the dataset metadata, for the same array. Getting the
    second one wrong changes only a few bytes deep inside a compressed metadata
    frame, and nothing about reading the file back would reveal it. If this test
    ever passes, the byte comparison has stopped being sensitive to the
    metadata and every green run above is worth nothing.
    """
    seed = tmp_path / "seed.pv"
    pz.write(_sphere(), seed, progress_bar=False)

    reference = tmp_path / "reference.pv"
    native = tmp_path / "native.pv"
    shutil.copyfile(seed, reference)
    shutil.copyfile(seed, native)

    arrays = {"step_1_disp": np.linspace(0.0, 1.0, 90).reshape(30, 3)}
    pz.append_arrays(reference, arrays)
    # "f8" is a real numpy dtype spelling -- just not the one str() produces.
    wrong = _write_spec(tmp_path, arrays, dtype_names={"step_1_disp": "f8"})
    _cpp_append(native, wrong, shuffle=False)

    assert _digest(reference) != _digest(native), (
        "the wrong dtype name produced an identical file; the parity gate is not "
        "comparing the dataset metadata"
    )


def test_auto_shuffle_actually_probes(tmp_path) -> None:
    """
    Under "auto", two multibyte floats get *different* answers.

    Without this, the parity runs above could all be green while the C++ side
    decided from the dtype alone: every array in the fixture is float or
    non-float, so a dtype-only rule agrees with the probe on most of them. The
    disagreement is the only thing that distinguishes the two rules, so it is
    what gets asserted.
    """
    seed = tmp_path / "seed.pv"
    pz.write(_sphere(), seed, progress_bar=False)
    native = tmp_path / "native.pv"
    shutil.copyfile(seed, native)

    rng = np.random.default_rng(17)
    arrays = {
        "step_1_disp": np.linspace(-2.0, 5.0, 300).reshape(100, 3),
        "step_1_noise": rng.random(97).astype(np.float32),
    }
    _cpp_append(native, _write_spec(tmp_path, arrays), shuffle="auto")

    filters = {
        h.name: h.filter_id for h in ref_reader.read(native).headers if "__field_data" in h.name
    }
    assert len(filters) == len(arrays)
    assert len(set(filters.values())) == DISTINCT_FILTER_DECISIONS, (
        f"both appended floats got the same filter decision ({filters}); "
        "'auto' is not running the trial compression"
    )


def test_kept_frames_are_copied_not_recompressed(tmp_path) -> None:
    """
    Everything before the first regenerated frame is unchanged, byte for byte.

    This is the property that makes appending cheap, and it is not implied by
    byte-identity with the reference: an implementation that decompressed and
    recompressed every frame could still match, as long as it matched exactly.
    Asserting the prefix separately keeps that distinction measurable.
    """
    seed = tmp_path / "seed.pv"
    pz.write(_unstructured(), seed, progress_bar=False)
    native = tmp_path / "native.pv"
    shutil.copyfile(seed, native)

    before = seed.read_bytes()
    arrays = {"step_1_ids": np.arange(32, dtype=np.int64)}
    _cpp_append(native, _write_spec(tmp_path, arrays), shuffle=False)
    after = native.read_bytes()

    shared = next(
        (i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b),
        min(len(before), len(after)),
    )
    # The dataset-metadata frame is regenerated, and it is the last array
    # before the file-metadata frame, so a large majority of the body has to be
    # a verbatim prefix. Half is a floor, not a target.
    assert shared > len(before) // 2, (
        f"only {shared} of {len(before)} bytes survived unchanged; kept frames are "
        "being rewritten rather than copied"
    )


def test_append_refuses_a_name_that_already_exists(tmp_path) -> None:
    """A second append under the same name is refused, not silently applied."""
    seed = tmp_path / "seed.pv"
    pz.write(_sphere(), seed, progress_bar=False)
    arrays = {"step_1_ids": np.arange(16, dtype=np.int64)}
    spec = _write_spec(tmp_path, arrays)

    _cpp_append(seed, spec, shuffle=False)
    committed = _digest(seed)

    result = subprocess.run(  # noqa: S603
        [APPEND, str(seed), LEVEL_FROM_FILE, "0", str(spec)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "appending a duplicate name should fail"
    # The refusal must also leave the container exactly as it was; a partial
    # write that happened to fail late would be worse than the overwrite.
    assert _digest(seed) == committed
