"""
Boundary properties its sibling cannot see.

``test_c_abi_boundary`` reads the sources, so it answers one question about
every entry point at once. The cost is that it can only see what is written in
them: it passed while ``pvz_open`` leaked its file descriptor down one refusal
path, while the loader raised ``AttributeError`` instead of the ABI-mismatch
message written for exactly that case, and while ``pvz_frame_sizes`` had no way
to be told how much room it had. Each of those is a property of what the
library *does*, so each is checked here by doing it.

Not covered, and left explicit rather than implied: a thread that cannot be
created. ``ParallelStride`` runs the un-spawned stripes inline rather than
destructing a joinable ``std::thread``, but provoking the failure needs the
process to be out of threads, which is not a state one test can enter without
deciding the fate of every other test in the session.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv

from pyvista_zstd import _capi
from pyvista_zstd import write

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(not _capi.available(), reason="the C++ core is not loadable here")


@pytest.fixture
def container(tmp_path: Path) -> str:
    """A small single-dataset container to read back."""
    ds = pv.Sphere()
    ds.point_data["scalar"] = np.arange(ds.n_points, dtype=np.float64)
    path = tmp_path / "runtime.pv"
    write(ds, path)
    return str(path)


def test_frame_sizes_refuses_a_short_destination(container: str) -> None:
    """
    The capacity is the only thing standing between a wrong count and a stomp.

    ``pvz_frame_count`` is a separate call, so a buffer allocated against a
    different reader looks exactly like a correct one from inside
    ``pvz_frame_sizes``. A short capacity has to be refused, not believed.
    """
    with _capi.CoreReader(container) as reader:
        count = int(reader._lib.pvz_frame_count(reader._live))  # noqa: SLF001
        assert count > 1, "need at least two frames for a short capacity to mean anything"

        short = np.zeros(count, dtype=np.uint64)
        status = reader._lib.pvz_frame_sizes(  # noqa: SLF001
            reader._live,
            short.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None,
            ctypes.c_uint64(count - 1),
        )

    assert status == _capi._STATUS_RANGE, f"short capacity returned {status}, not PVZ_E_RANGE"  # noqa: SLF001
    assert not short.any(), "refused, and wrote anyway"


def test_frame_sizes_fills_at_the_real_capacity(container: str) -> None:
    """
    The refusal above is worthless if it also refuses the right answer.

    Frames pair as (header, payload) and a payload may legitimately be empty --
    an array with no entries compresses to a zstd frame carrying nothing. The
    headers cannot be, and neither can any compressed frame, so those are what
    tell a filled buffer from the zeros it was allocated as.
    """
    with _capi.CoreReader(container) as reader:
        decompressed, compressed = reader.frame_sizes()

    assert decompressed.size == compressed.size
    assert decompressed.size > 1
    assert decompressed.size % 2 == 0, "frames pair as (header, payload)"
    assert (decompressed[0::2] > 0).all(), "a header frame decompressing to nothing is not a header"
    assert (compressed > 0).all(), "a zstd frame occupying no bytes was never written"


def test_a_library_without_the_symbols_is_a_load_failure() -> None:
    """
    Binding happens inside the guard, so a missing symbol is a load failure.

    An ABI bump that ADDS a symbol makes an older library fail at the bind
    rather than at the version comparison written to explain it. Offered only a
    shared library that is not this one, the loader must raise
    ``CoreUnavailableError`` naming what it tried -- not whatever ctypes raises
    first.

    Run out of process because the loader caches its answer for the session.
    """
    script = textwrap.dedent(
        """
        import ctypes, sys
        decoy = None
        for candidate in ("libz.so.1", "libz.so", "libm.so.6", "libc.so.6"):
            try:
                ctypes.CDLL(candidate)
            except OSError:
                continue
            decoy = candidate
            break
        if decoy is None:
            print("SKIP")
            sys.exit(0)

        import pyvista_zstd._capi as capi

        # Offer the decoy and nothing else: the real library must not be
        # reachable, or this measures the fallback rather than the bind.
        capi._candidate_paths = lambda: iter([decoy])
        capi._lib = None
        capi._lib_path = None
        capi._load_error = None

        try:
            capi._load()
        except capi.CoreUnavailableError as exc:
            assert decoy in str(exc), f"declined without naming what it tried: {exc}"
            print("CoreUnavailableError")
        else:
            raise AssertionError("a library exporting no pvz_* symbols was accepted")
        """
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    verdict = done.stdout.strip()
    if verdict == "SKIP":
        pytest.skip("no loadable non-pvzstd shared library to offer the loader")
    assert verdict == "CoreUnavailableError", done.stdout


def test_read_arrays_decodes_the_same_bytes_at_every_thread_setting(container: str) -> None:
    """
    The worker count is a schedule, never an answer.

    What this pins is result-invariance, and only that. It does NOT witness the
    fix that made ``-1`` mean every core rather than falling through a signed
    clamp into single-threaded: neutering that fix and re-running leaves this
    green, because both dispatches decode the same bytes. It is measured, not
    assumed -- the neutered build was built and run.

    Keep it anyway. The invariance is the property worth defending: the day a
    thread setting changes an answer, this is what says so, and nothing about
    the worker count is otherwise observable from outside the library.
    """
    settings = [_capi.THREADS_AUTO, 0, 1, 2, -1]
    results = []
    for n_threads in settings:
        with _capi.CoreReader(container) as reader:
            arrays = reader.read_arrays(n_threads=n_threads)
        results.append({name: arr.tobytes() for name, arr in arrays.items()})

    first = results[0]
    assert first, "read nothing, so agreeing about it proves nothing"
    for setting, got in zip(settings[1:], results[1:], strict=True):
        assert got == first, f"n_threads={setting} decoded different bytes"
