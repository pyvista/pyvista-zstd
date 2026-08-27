"""
Append parity between the Python surface and the pvzstd_append tool.

Both arms reach pvzstd_append_arrays, so agreement pins the ctypes binding
against a second caller of the same ABI, not the format itself.
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

APPEND = os.environ.get("PVZSTD_APPEND")

pytestmark = pytest.mark.skipif(
    not APPEND or not Path(APPEND).exists(),
    reason="set PVZSTD_APPEND to a built cpp/ pvzstd_append binary to run append parity",
)

SHUFFLE_CODE = {False: 0, True: 1, "auto": 2}
# PVZSTD_LEVEL_FROM_FILE: makes the tool read the level out of the container, which
# is what ``level=None`` asks for on the binding side, rather than handing the
# core a resolved number one arm worked out and the other did not.
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
        # Two multibyte floats "auto" decides differently -- the probe declines
        # the ramp and accepts the noise. A heuristic keyed on dtype alone would
        # give both the same answer and still look green on one array.
        "step_1_disp": np.linspace(-2.0, 5.0, 300).reshape(100, 3),
        "step_1_noise": rng.random(97).astype(np.float32),
        # integer: never a candidate under "auto", always one under True
        "step_1_ids": np.arange(64, dtype=np.int64),
        # itemsize 1: excluded by the dtype gate under every mode
        "step_1_flags": np.ones(41, dtype=np.uint8),
        # empty multibyte: shuffling nothing is a no-op, but the header byte
        # that records it is not
        "step_1_empty": np.zeros(0, dtype=np.float64),
        # Over 1 MiB, which is the point: below that zstd emits identical bytes
        # at threads=0 and threads=1, so the worker count would go unmeasured.
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
        spec_lines.append("\t".join([name, contiguous.dtype.str, dtype_name, shape_csv, str(raw)]))
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
    """
    Both entry points into the append produce the same file, byte for byte.

    A failure means the binding hands the core something different from what
    the tool hands it, for the same arrays and options.
    """
    seed = tmp_path / f"{label}.pv"
    pz.write(DATASETS[label](), seed, progress_bar=False)

    via_binding = tmp_path / "via_binding.pv"
    via_tool = tmp_path / "cpp.pv"
    shutil.copyfile(seed, via_binding)
    shutil.copyfile(seed, via_tool)

    arrays = _payloads()
    pz.append_arrays(via_binding, arrays, shuffle=shuffle)
    _cpp_append(via_tool, _write_spec(tmp_path, arrays), shuffle=shuffle)

    expected = via_binding.read_bytes()
    actual = via_tool.read_bytes()
    if expected != actual:  # pragma: no cover - failure path
        first = next(
            (i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
            min(len(expected), len(actual)),
        )
        pytest.fail(
            f"{label} shuffle={shuffle}: {len(expected)} vs {len(actual)} bytes, first difference at byte {first}"
        )


def test_the_parity_gate_can_fail(tmp_path) -> None:
    """
    Negative control: a wrong dtype spelling must redden the comparison above.

    The format records a dtype string ("<f8") in the frame header and a dtype
    name ("float64") in the dataset metadata. Getting the second wrong changes
    only a few bytes inside a compressed metadata frame, and reading the file
    back would not reveal it.
    """
    seed = tmp_path / "seed.pv"
    pz.write(_sphere(), seed, progress_bar=False)

    via_binding = tmp_path / "via_binding.pv"
    via_tool = tmp_path / "cpp.pv"
    shutil.copyfile(seed, via_binding)
    shutil.copyfile(seed, via_tool)

    arrays = {"step_1_disp": np.linspace(0.0, 1.0, 90).reshape(30, 3)}
    pz.append_arrays(via_binding, arrays)
    # "f8" is a real numpy dtype spelling -- just not the one str() produces.
    wrong = _write_spec(tmp_path, arrays, dtype_names={"step_1_disp": "f8"})
    _cpp_append(via_tool, wrong, shuffle=False)

    assert _digest(via_binding) != _digest(via_tool), (
        "the wrong dtype name produced an identical file; the parity gate is not comparing the dataset metadata"
    )


def test_auto_shuffle_actually_probes(tmp_path) -> None:
    """
    Under "auto", two multibyte floats get *different* answers.

    A dtype-only rule agrees with the probe on every other array in the
    fixture, so the disagreement is the only thing that distinguishes them.
    """
    seed = tmp_path / "seed.pv"
    pz.write(_sphere(), seed, progress_bar=False)
    cpp = tmp_path / "cpp.pv"
    shutil.copyfile(seed, cpp)

    rng = np.random.default_rng(17)
    arrays = {
        "step_1_disp": np.linspace(-2.0, 5.0, 300).reshape(100, 3),
        "step_1_noise": rng.random(97).astype(np.float32),
    }
    _cpp_append(cpp, _write_spec(tmp_path, arrays), shuffle="auto")

    filters = {h.name: h.filter_id for h in ref_reader.read(cpp).headers if "__field_data" in h.name}
    assert len(filters) == len(arrays)
    assert len(set(filters.values())) == DISTINCT_FILTER_DECISIONS, (
        f"both appended floats got the same filter decision ({filters}); 'auto' is not running the trial compression"
    )


def test_kept_frames_are_copied_not_recompressed(tmp_path) -> None:
    """
    Everything before the first regenerated frame is unchanged, byte for byte.

    Not implied by byte-identity with the reference: an implementation that
    recompressed every frame could still match exactly.
    """
    seed = tmp_path / "seed.pv"
    pz.write(_unstructured(), seed, progress_bar=False)
    cpp = tmp_path / "cpp.pv"
    shutil.copyfile(seed, cpp)

    before = seed.read_bytes()
    arrays = {"step_1_ids": np.arange(32, dtype=np.int64)}
    _cpp_append(cpp, _write_spec(tmp_path, arrays), shuffle=False)
    after = cpp.read_bytes()

    shared = next(
        (i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b),
        min(len(before), len(after)),
    )
    # Half is a floor, not a target: only the two metadata frames are regenerated.
    assert shared > len(before) // 2, (
        f"only {shared} of {len(before)} bytes survived unchanged; kept frames are being rewritten rather than copied"
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
    # A partial write that failed late would be worse than the overwrite.
    assert _digest(seed) == committed
