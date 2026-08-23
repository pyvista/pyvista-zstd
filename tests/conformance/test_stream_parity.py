"""
Hold the streaming writer to the copying one: same bytes, flat cost.

Skipped unless ``PVZ_STREAM`` points at a built ``pvz_stream`` binary.

:func:`~pyvista_zstd.append_arrays` is handed a path and nothing else, so every
call rediscovers the container: trailer, both metadata frames, and a copy of
the body into a temporary file it renames. A stream keeps that state instead.
The two must produce the same file -- N commits through a stream and N separate
appends are the same edit -- while costing very different amounts.

Both halves are asserted, and the cost half is asserted *relative to a control
measured in the same run*. Absolute timings here are not a property of the
code: the same control measured 4.24x growth on one occasion and 10.30x on
another, because how much of the container is in page cache dominates. What is
stable is that one arm grows with what is already committed and the other does
not, so that is what gets compared, never a millisecond figure.

Nor is the *control's* growth an absolute the way it first looked. Requiring it
to exceed a fixed factor reddened all three CI platforms, because on those
runners this fixture only drives it to 2.52x (Windows) and 2.97x (macOS) where
a workstation reaches 10.30x. That was replaced by a separation between the two
arms, on the reasoning that a ratio survives the machine changing underneath it.

That fix reddened CI a second time, one assertion further down: the total
speedup floor was 5.0, and a Linux runner managed 4.39x against a workstation's
87.6x. Same mistake, same file, two lines apart. The rule that separates the
two cases: **a bound on the arm that is supposed to stay flat is a claim about
this code and transfers; a bound on how large the control gets, or on any ratio
the control dominates, is a claim about the machine's storage and does not.**
So `MAX_STREAM_GROWTH` stays absolute and the speedup floor became a bare
"faster than", with the magnitude left to the docs where it belongs.

The separation then became the third instance of the same mistake, and how it
got there is worth more than the fix. It was written as a two-arm ratio --
control growth over stream growth -- which the rule above permits. But the
denominator was floored at 1.0, to stop noise in a flat stream arm inflating the
result. A healthy stream *is* flat: measured 0.42x to 0.89x across eight runs,
so the floor bound every one of them, the denominator was the constant 1.0, and
the ratio was control growth wearing a ratio's clothes. It failed 8 of 12 runs
on an idle machine, at 1.39x to 1.76x against a bound of 1.8. **Clamping one arm
of a ratio turns it into a bound on the other**, and the tell was printed in
every failure message: separation and control growth were the same number to two
decimal places, run after run.

What that assertion was really for is not a claim about this code -- the flat
arm and the bytes-read assertion below already carry that, and both passed
throughout. It asks whether the fixture still poses the problem the stream
exists to solve. That is settled by how much container the copying path is made
to re-read, which is a property of the fixture rather than of the machine:
`MIN_CONTROL_CONTAINER_GROWTH` measures it in bytes on disk, where page cache
cannot flatter it, and it cannot go flaky because nothing about it is timed.

One limit worth knowing: the bytes-read assertion -- the one that actually
holds this line -- reads ``/proc/self/io`` and therefore only runs on Linux.
The tool reports -1 elsewhere and the assertion skips. On macOS and Windows
these timing bounds are all there is, which is precisely why they are kept
even though they are not what catches the regression that matters.

What these tests catch was established by breaking the implementation, and one
break is the reason the cost test asserts on bytes rather than on time alone.
Making every commit re-read the whole container -- the exact regression the
streaming path exists to prevent -- passes all three timing bounds: the re-read
is served from page cache, which is too cheap to separate from compression
noise at this size. It fails the bytes-read assertion immediately. The timing
bounds are kept because they pin the comparison against the copying path, but
they are not what holds this line.

Of the other breaks: splicing new keys at the head of ``field_data_keys``
instead of the tail reddens five, compressing payload frames with workers > 0
reddens four (which is also what shows the >=1 MiB array in the fixture is
doing its job -- below that threshold zstd emits identical bytes either way),
dropping the duplicate-name check reddens one, and accepting any declared
commit count reddens one. Removing the final truncate reddens nothing, and
that is correct rather than a gap: a stream only ever appends, so the file
grows monotonically and the truncate has nothing to remove. It is a guard
against a shrinking rewrite that this API cannot currently perform, and no
test here should be read as covering it.
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

import pyvista_zstd as pz

STREAM = os.environ.get("PVZ_STREAM")

pytestmark = pytest.mark.skipif(
    not STREAM or not Path(STREAM).exists(),
    reason="set PVZ_STREAM to a built cpp/ pvz_stream binary to run stream parity",
)

SHUFFLE_CODE = {False: "0", True: "1", "auto": "2"}
# pvz_status; the tool reports codes rather than messages so the test can name
# the one it means.
PVZ_E_INVALID = 7
# Enough commits that a per-commit cost proportional to the file shows itself.
# How much depends on the machine -- 10.30x control growth on a workstation but
# 2.52x on a Windows runner -- so nothing below is asserted against the control's
# wall time.
N_COMMITS = 24
# A bound on the arm that should stay flat is a statement about this code and
# transfers; a bound on the control, or on a ratio it dominates, is a statement
# about the machine's storage and does not. Both were absolute here at first and
# CI reddened on each in turn.
#
# About the code: the stream must not be meaningfully slower at the end than at
# the start. Measured 0.97x; 3.0 leaves room for a loaded runner.
MAX_STREAM_GROWTH = 3.0
# About the fixture, which is the same everywhere: the container grows 0.81 MB to
# 14.82 MB, a ratio of 6.70x that is arithmetic under a fixed seed rather than a
# measurement. Below this the fixture has stopped posing the problem.
MIN_CONTROL_CONTAINER_GROWTH = 3.0
# Only that streaming is faster, not by how much: measured 87.6x on a workstation
# and 4.39x on a Linux runner, and an earlier floor of 5.0 reddened CI.
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

    Byte-identity with the append path already implies this, but only while
    that test passes. If both writers were wrong in the same way the
    comparison would stay green, so the file is also read back through the
    reference reader, which knows nothing about either.
    """
    container = tmp_path / "stream.pv"
    _seed(container)
    commits = _commits()
    specs = [_write_spec(tmp_path, f"r{i}", a) for i, a in enumerate(commits)]
    _run_stream(container, specs, shuffle=False)

    back = pz.read(container)
    for arrays in commits:
        for name, arr in arrays.items():
            assert name in back.field_data, name
            assert np.array_equal(back.field_data[name].ravel(), arr.ravel()), name


def test_stream_cost_does_not_grow_with_what_is_already_committed(tmp_path) -> None:
    """
    The property the whole streaming path exists for.

    Asserted against a control measured in the same run, because the absolute
    numbers are not stable across machines or even across runs on one machine
    -- page-cache residency dominates them. The fixture also has to keep posing
    the problem: if the container stops growing, the copying path is never made
    to repeat any work and this test is measuring nothing. That is checked on
    the size of the file rather than on how long the control took, because the
    first is arithmetic and the second is the machine's storage.
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
    # All four numbers on every failure: the first CI failure reported only the one
    # that tripped, which could not say whether the stream or the runner was slow.
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

    Silence here would be the worst outcome available: the container reads
    back perfectly and is simply missing results, which no reader can tell
    from a result set that was always that size.
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
    assert "pvz_stream_close" in result.stderr
    # The commit succeeded: the refusal is about the declared total.
    assert "count out of range" in result.stderr


def test_a_name_already_in_the_container_is_refused(tmp_path) -> None:
    """
    A colliding field name is refused rather than shadowing the array on disk.

    Two frames can carry the same name -- nothing in the format forbids it --
    and a reader resolving by name would then silently return whichever one
    it reached first.
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

    The two are not the same kind of refusal as the one above. A duplicate name
    is caught before anything is touched, so the stream is still good and the
    caller may carry on. A commit that fails after it has begun -- the tail's
    frame entries dropped, some frames written, ``body_end`` not yet moved --
    leaves the stream describing frames that are not where it says they are.
    Appending on top of that produces a file that parses and gives wrong data,
    and closing it would present that file as complete.

    The fault is injected by asking for an array of 2**60 bytes: validation
    passes, mutation starts, and the payload buffer is the first thing that
    cannot be had. Which failure it is does not matter here -- only that the
    stream is dead afterwards.

    Removing the ``failed`` flag reddens both assertions below: the append
    returns 0 and the close returns 0.
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
    assert reported["append"] == str(PVZ_E_INVALID), (
        f"a valid append on a poisoned stream returned {reported['append']}; "
        "the commit before it stopped part-way and left the frame list wrong"
    )
    assert reported["close"] == str(PVZ_E_INVALID), (
        f"closing a poisoned stream returned {reported['close']}; it cannot say where the failed commit stopped writing"
    )
    # Nothing was committed, so nothing may be counted.
    assert reported["commits"] == "0"
