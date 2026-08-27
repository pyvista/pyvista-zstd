"""
Streaming commits against the append path.

Both arms end at the same C entry points, so agreement pins the streaming
driver rather than the format. The timing arms below bound the FLAT arm
only: a bound on the control measures the machine, not the code.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pytest
import pyvista as pv
import ref_reader

import pyvista_zstd as pz

STREAM = os.environ.get("PVZSTD_STREAM")

pytestmark = pytest.mark.skipif(
    not STREAM or not Path(STREAM).exists(),
    reason="set PVZSTD_STREAM to a built cpp/ pvzstd_stream binary to run stream parity",
)

SHUFFLE_CODE = {False: "0", True: "1", "auto": "2"}
# pvzstd_status; the tool reports codes rather than messages so the test can name
# the one it means.
PVZSTD_E_INVALID = 7
# Enough commits that a per-commit cost proportional to the file shows itself.
# The control's wall time is machine-dependent, so nothing below bounds it.
N_COMMITS = 24
# The stream must not be meaningfully slower at the end than at the start.
MAX_STREAM_GROWTH = 3.0
# The container's growth is arithmetic under a fixed seed, not a measurement.
# Below this the fixture has stopped posing the problem.
MIN_CONTROL_CONTAINER_GROWTH = 3.0
# Only that streaming is faster, not by how much.
MIN_TOTAL_SPEEDUP = 1.5
HEAD = 5


def _seed(path: Path) -> None:
    rng = np.random.default_rng(41)
    ds = pv.Sphere(theta_resolution=10, phi_resolution=10)
    ds.point_data["disp"] = rng.random((ds.n_points, 3))
    ds.field_data["born"] = np.array([1.5, 2.5])
    pz.write(ds, path, progress_bar=False)


def _commits() -> list[dict[str, np.ndarray]]:
    """Five commits covering the shapes a result stream really carries."""
    rng = np.random.default_rng(17)
    return [
        {"step_0_u": rng.random((30, 3)), "step_0_ids": np.arange(7, dtype=np.int32)},
        {"step_1_u": rng.random((30, 3))},
        {"step_2_noise": rng.random(97).astype(np.float32), "step_2_flags": np.ones(11, np.uint8)},
        # Shuffling nothing is a no-op, but the header byte recording it is not.
        {"step_3_empty": np.zeros(0, np.float64)},
        # Over 1 MiB: below that zstd emits the same bytes threaded or not.
        {"step_4_bulk": rng.random(1 << 18)},
    ]


def _write_spec(tmp_path: Path, tag: str, arrays: dict[str, np.ndarray]) -> str:
    lines = []
    for i, (name, arr) in enumerate(arrays.items()):
        contiguous = np.ascontiguousarray(arr)
        raw = tmp_path / f"{tag}_{i}.bin"
        raw.write_bytes(contiguous.tobytes())
        lines.append(
            "\t".join(
                [
                    name,
                    contiguous.dtype.str,
                    str(contiguous.dtype),
                    ",".join(str(d) for d in contiguous.shape),
                    str(raw),
                ]
            )
        )
    spec = tmp_path / f"{tag}.tsv"
    spec.write_text("\n".join(lines) + "\n")
    return str(spec)


def _run_stream(container: Path, specs: list[str], *, shuffle) -> list[tuple[float, int]]:
    """Run the stream, returning (seconds, bytes_read) per commit from stderr."""
    result = subprocess.run(  # noqa: S603
        [STREAM, str(container), SHUFFLE_CODE[shuffle], *specs],
        capture_output=True,
        text=True,
        check=True,
    )
    return [(float(m.group(1)), int(m.group(2))) for m in re.finditer(r"commit \d+ ([\d.]+) (-?\d+)", result.stderr)]


@pytest.mark.parametrize("shuffle", [False, True, "auto"])
def test_stream_matches_separate_appends_byte_for_byte(tmp_path, shuffle) -> None:
    """A stream of N commits produces the file N separate appends would."""
    seed = tmp_path / "seed.pv"
    _seed(seed)
    reference = tmp_path / "reference.pv"
    cpp = tmp_path / "cpp.pv"
    shutil.copyfile(seed, reference)
    shutil.copyfile(seed, cpp)

    specs = []
    for i, arrays in enumerate(_commits()):
        pz.append_arrays(reference, arrays, shuffle=shuffle)
        specs.append(_write_spec(tmp_path, f"c{i}", arrays))
    _run_stream(cpp, specs, shuffle=shuffle)

    expected = reference.read_bytes()
    actual = cpp.read_bytes()
    if expected != actual:  # pragma: no cover - failure path
        first = next(
            (i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
            min(len(expected), len(actual)),
        )
        pytest.fail(f"shuffle={shuffle}: {len(expected)} vs {len(actual)} bytes, first difference at byte {first}")


def test_streamed_blocks_read_back_through_the_reference_reader(tmp_path) -> None:
    """
    The result of streaming is an ordinary container.

    Read back through the reference reader, which knows nothing about either
    writer, so two writers wrong in the same way cannot pass.
    """
    container = tmp_path / "stream.pv"
    _seed(container)
    commits = _commits()
    specs = [_write_spec(tmp_path, f"r{i}", a) for i, a in enumerate(commits)]
    _run_stream(container, specs, shuffle=False)

    # ref_reader keys by the full frame name, which carries the dataset UID.
    back = ref_reader.read(container).arrays
    for arrays in commits:
        for name, arr in arrays.items():
            suffix = f"{name}__field_data"
            found = [k for k in back if k.endswith(suffix)]
            assert len(found) == 1, f"{suffix} matched {found}"
            assert np.array_equal(back[found[0]].ravel(), arr.ravel()), name


def test_stream_cost_does_not_grow_with_what_is_already_committed(tmp_path) -> None:
    """
    Streaming cost does not grow with what is already committed.

    Asserted against a control measured in the same run; absolute numbers are
    not stable across machines. The container must keep growing, or the
    copying path never repeats work and this measures nothing -- checked on
    file size, which is arithmetic, not on the control's wall time.
    """
    rng = np.random.default_rng(7)
    ds = pv.ImageData(dimensions=(30, 30, 30))
    ds.point_data["base"] = rng.random(ds.n_points)
    block = rng.random((27000, 3))
    raw = tmp_path / "blk.bin"
    raw.write_bytes(block.tobytes())

    control = tmp_path / "control.pv"
    pz.write(ds, control, progress_bar=False)
    control_times = []
    control_sizes = []
    for i in range(N_COMMITS):
        start = time.perf_counter()
        pz.append_arrays(control, {f"step_{i}_u": block}, shuffle=False)
        control_times.append(time.perf_counter() - start)
        control_sizes.append(control.stat().st_size)

    streamed = tmp_path / "streamed.pv"
    pz.write(ds, streamed, progress_bar=False)
    specs = []
    for i in range(N_COMMITS):
        spec = tmp_path / f"s{i}.tsv"
        spec.write_text(f"step_{i}_u\t{block.dtype.str}\t{block.dtype}\t27000,3\t{raw}\n")
        specs.append(str(spec))
    commits = _run_stream(streamed, specs, shuffle=False)
    assert len(commits) == N_COMMITS
    stream_times = [t for t, _ in commits]
    stream_reads = [b for _, b in commits]

    control_growth = sum(control_times[-HEAD:]) / sum(control_times[:HEAD])
    stream_growth = sum(stream_times[-HEAD:]) / sum(stream_times[:HEAD])
    speedup = sum(control_times) / sum(stream_times)
    # Measured, not timed: whether the fixture still poses the problem is settled
    # by how big the file got, which page cache cannot flatter.
    container_growth = sum(control_sizes[-HEAD:]) / sum(control_sizes[:HEAD])
    # All four numbers on every failure: one alone cannot say whether the stream
    # or the runner was slow.
    measured = (
        f"[control growth {control_growth:.2f}x, stream growth {stream_growth:.2f}x, "
        f"container growth {container_growth:.2f}x, total speedup {speedup:.2f}x] "
    )
    assert container_growth > MIN_CONTROL_CONTAINER_GROWTH, (
        f"{measured}the container barely grew across the run, so the copying path was "
        "never made to repeat meaningful work and there is nothing here for the stream "
        "to be better than -- this needs a bigger fixture, not a lower bound"
    )
    assert stream_growth < MAX_STREAM_GROWTH, (
        f"{measured}per-commit cost grew from the first {HEAD} commits to the last "
        f"{HEAD}; the stream is re-reading something it should be holding"
    )
    assert speedup > MIN_TOTAL_SPEEDUP, f"{measured}streaming did not beat {N_COMMITS} separate appends"

    # The assertion that holds the line: a stream that re-reads the container
    # passes every timing bound above, because page cache hides it. Bytes do not.
    if all(b >= 0 for b in stream_reads):
        total_read = sum(stream_reads)
        produced = streamed.stat().st_size
        assert total_read < produced, (
            f"the stream read {total_read} bytes to produce a {produced}-byte container; "
            "holding the state means never reading back what was already committed"
        )

    # Same edit, so the same file -- the cost is all that should differ.
    assert control.read_bytes() == streamed.read_bytes()


def test_close_refuses_a_stream_shorter_than_it_declared(tmp_path) -> None:
    """
    A stream that committed fewer sets than declared is an error, not a file.

    Silence would produce a container that reads back perfectly and is simply
    missing results, which no reader can distinguish from a shorter run.
    """
    container = tmp_path / "short.pv"
    _seed(container)
    specs = [_write_spec(tmp_path, "one", {"step_0_u": np.zeros((4, 3))})]

    result = subprocess.run(  # noqa: S603
        [STREAM, "--expect=2", str(container), "0", *specs],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "closing 1 commit against a declared 2 should fail"
    assert "pvzstd_stream_close" in result.stderr
    # The commit succeeded: the refusal is about the declared total.
    assert "count out of range" in result.stderr


def test_a_name_already_in_the_container_is_refused(tmp_path) -> None:
    """
    A colliding field name is refused rather than shadowing the array on disk.

    Nothing in the format forbids two frames with one name; a reader resolving
    by name would silently return whichever it reached first.
    """
    container = tmp_path / "collide.pv"
    _seed(container)
    specs = [_write_spec(tmp_path, "dup", {"step_0_u": np.zeros((4, 3))})]

    _run_stream(container, specs, shuffle=False)
    before = container.read_bytes()

    result = subprocess.run(  # noqa: S603
        [STREAM, str(container), "0", *specs],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "a duplicate field name should be refused"
    # The collision has its own status, so this pins the reason and not merely
    # that something was rejected -- "invalid argument" would also have covered
    # a malformed spec file, which is a different bug with the same exit code.
    assert "already in the container" in result.stderr
    # Refused before anything was written: the container is untouched.
    assert container.read_bytes() == before


def test_a_commit_that_failed_part_way_poisons_the_stream(tmp_path) -> None:
    """
    After a commit fails mid-write, every later call on that stream is refused.

    A duplicate name is caught before anything is touched, so the stream
    survives it. A commit that fails after it has begun leaves the stream
    describing frames that are not where it says they are; appending on top
    would parse and give wrong data. The fault is injected by asking for an
    array of 2**60 bytes.
    """
    container = tmp_path / "poison.pv"
    _seed(container)
    specs = [_write_spec(tmp_path, "after", {"step_0_u": np.zeros((4, 3))})]

    result = subprocess.run(  # noqa: S603
        [STREAM, "--poison", str(container), "0", *specs],
        capture_output=True,
        text=True,
        check=True,
    )
    reported = dict(line.split() for line in result.stdout.splitlines())

    assert reported["poison"] != "0", "an array of 2**60 bytes should not have been committed"
    assert reported["append"] == str(PVZSTD_E_INVALID), (
        f"a valid append on a poisoned stream returned {reported['append']}; "
        "the commit before it stopped part-way and left the frame list wrong"
    )
    assert reported["close"] == str(PVZSTD_E_INVALID), (
        f"closing a poisoned stream returned {reported['close']}; it cannot say where the failed commit stopped writing"
    )
    # Nothing was committed, so nothing may be counted.
    assert reported["commits"] == "0"
