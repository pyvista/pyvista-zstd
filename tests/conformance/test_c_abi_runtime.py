"""Runtime behaviour of the C ABI: what the library does, not what it declares."""

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
    """Return a small single-dataset container to read back."""
    ds = pv.Sphere()
    ds.point_data["scalar"] = np.arange(ds.n_points, dtype=np.float64)
    path = tmp_path / "runtime.pv"
    write(ds, path)
    return str(path)


def test_frame_sizes_refuses_a_short_destination(container: str) -> None:
    """A capacity smaller than the frame count is refused, not written to."""
    with _capi.CoreReader(container) as reader:
        count = int(reader._lib.pvz_frame_count(reader._live))  # noqa: SLF001
        assert count > 1, "need at least two frames for a short capacity to mean anything"

        short = np.zeros(count, dtype=np.uint64)
        status = reader._lib.pvz_frame_sizes(  # noqa: SLF001
            reader._live,  # noqa: SLF001
            short.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None,
            ctypes.c_uint64(count - 1),
        )

    assert status == _capi._STATUS_RANGE, f"short capacity returned {status}, not PVZ_E_RANGE"  # noqa: SLF001
    assert not short.any(), "refused, and wrote anyway"


def test_frame_sizes_fills_at_the_real_capacity(container: str) -> None:
    """At the true capacity the buffers come back filled."""
    with _capi.CoreReader(container) as reader:
        decompressed, compressed = reader.frame_sizes()

    assert decompressed.size == compressed.size
    assert decompressed.size > 1
    assert decompressed.size % 2 == 0, "frames pair as (header, payload)"
    # A payload may legitimately be empty; a header or a zstd frame cannot.
    assert (decompressed[0::2] > 0).all(), "a header frame decompressing to nothing is not a header"
    assert (compressed > 0).all(), "a zstd frame occupying no bytes was never written"


def test_a_library_without_the_symbols_is_a_load_failure() -> None:
    """A shared library that is not this one is declined by name, not by AttributeError."""
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

        # Offer the decoy and nothing else, or this measures the fallback.
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
    # Out of process: the loader caches its answer for the session.
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    verdict = done.stdout.strip()
    if verdict == "SKIP":
        pytest.skip("no loadable non-pvzstd shared library to offer the loader")
    assert verdict == "CoreUnavailableError", done.stdout


def test_read_arrays_decodes_the_same_bytes_at_every_thread_setting(container: str) -> None:
    """The worker count is a schedule, not an answer: every setting decodes the same bytes."""
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
