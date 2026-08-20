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
# Enough commits that a per-commit cost proportional to the file has room to
# show itself; the control below confirms it does.
N_COMMITS = 24
# The stream must not be meaningfully slower at the end than at the start.
# Measured 0.97x; 3.0 leaves room for a loaded runner without admitting growth.
MAX_STREAM_GROWTH = 3.0
# The stream must beat the copying path over the whole run. Measured 87.6x on
# this fixture; 5 is a floor, not a target.
MIN_TOTAL_SPEEDUP = 5.0
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
        # Over 1 MiB: below that, zstd emits the same bytes threaded or not, so
        # a fixture of small arrays cannot tell the framing apart.
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
    native = tmp_path / "native.pv"
    shutil.copyfile(seed, reference)
    shutil.copyfile(seed, native)

    specs = []
    for i, arrays in enumerate(_commits()):
        pz.append_arrays(reference, arrays, shuffle=shuffle)
        specs.append(_write_spec(tmp_path, f"c{i}", arrays))
    _run_stream(native, specs, shuffle=shuffle)

    expected = reference.read_bytes()
    actual = native.read_bytes()
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
    -- page-cache residency dominates them. The control has to *show growth*
    for its arm to mean anything: if it ever comes out flat, the fixture has
    stopped exercising the thing being compared and this test is measuring
    nothing.
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
    for i in range(N_COMMITS):
        start = time.perf_counter()
        pz.append_arrays(control, {f"step_{i}_u": block}, shuffle=False)
        control_times.append(time.perf_counter() - start)

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

    assert control_growth > MAX_STREAM_GROWTH, (
        f"the control did not grow ({control_growth:.2f}x); the fixture is too small "
        "for the copying path's cost to show, so the comparison proves nothing"
    )
    assert stream_growth < MAX_STREAM_GROWTH, (
        f"per-commit cost grew {stream_growth:.2f}x from the first {HEAD} commits to the "
        f"last {HEAD}; the stream is re-reading something it should be holding"
    )
    assert speedup > MIN_TOTAL_SPEEDUP, f"streaming was only {speedup:.2f}x faster than {N_COMMITS} separate appends"

    # The assertion that actually holds the line. A stream that re-reads the
    # container still passes every timing bound above, because the re-read is
    # served from page cache and disappears into compression noise -- measured,
    # not assumed. Bytes read does not disappear.
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
    # The commit itself succeeded, so it is on disk: the refusal is about the
    # declared total, not about the write.
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
    assert "invalid argument" in result.stderr
    # Refused before anything was written: the container is untouched.
    assert container.read_bytes() == before
