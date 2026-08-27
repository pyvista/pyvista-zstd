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
        count = int(reader._lib.pvzstd_frame_count(reader._live))  # noqa: SLF001
        assert count > 1, "need at least two frames for a short capacity to mean anything"

        short = np.zeros(count, dtype=np.uint64)
        status = reader._lib.pvzstd_frame_sizes(  # noqa: SLF001
            reader._live,  # noqa: SLF001
            short.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None,
            ctypes.c_uint64(count - 1),
        )

    assert status == _capi._STATUS_RANGE, f"short capacity returned {status}, not PVZSTD_E_RANGE"  # noqa: SLF001
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
            raise AssertionError("a library exporting no pvzstd_* symbols was accepted")
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


def test_read_array_at_accepts_a_null_destination_for_a_zero_byte_array(tmp_path: Path) -> None:
    """A zero-byte read must succeed even with a null destination.

    ``pv.Sphere()`` has no lines, verts, or strips, so its
    ``lines_connectivity``, ``verts_connectivity`` and ``strips_connectivity``
    frames are zero bytes. numpy always hands back a non-null pointer for a
    zero-size array, so going through :meth:`CoreReader.read_at` cannot
    exercise a null destination the way a C or C++ caller sizing its buffer
    from ``nbytes`` would; this reads every array through the raw ctypes
    entry point instead, passing ``None`` whenever ``nbytes`` is zero.
    """
    ds = pv.Sphere()
    path = tmp_path / "empty_topology.pv"
    write(ds, path)

    with _capi.CoreReader(str(path)) as reader:
        count = len(reader)
        assert count > 0

        saw_empty = False
        for index in range(count):
            info = reader._info(index)  # noqa: SLF001
            nbytes = int(info.nbytes)
            if nbytes == 0:
                saw_empty = True
                dst = None
            else:
                buf = np.empty(nbytes, dtype=np.uint8)
                dst = buf.ctypes.data_as(ctypes.c_void_p)

            status = reader._lib.pvzstd_read_array_at(  # noqa: SLF001
                reader._live,  # noqa: SLF001
                ctypes.c_uint64(index),
                dst,
                ctypes.c_uint64(nbytes),
            )
            name = info.name.decode("utf-8")
            assert status == _capi._STATUS_OK, f"array {index} ({name}, {nbytes} bytes) returned {status}"  # noqa: SLF001

        assert saw_empty, "need at least one zero-byte array for this test to mean anything"


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
