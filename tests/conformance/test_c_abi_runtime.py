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
        capi._candidate_paths = lambda: iter([(decoy, False)])
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
    """
    A zero-byte read must succeed even with a null destination.

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


class _NoSymbols:
    """
    A stand-in for a core built before a symbol this binding needs existed.

    Only ``pvzstd_abi_version`` resolves; anything else raises the same
    ``AttributeError`` ctypes raises for a symbol the shared object does not
    export. That is the shape of a real ABI gap that adds entry points, and it
    is what makes "which failure gets reported" observable without shipping a
    second compiled library to test against.
    """

    class _Entry:
        def __init__(self, value: int) -> None:
            self._value = value
            self.restype: object = None
            self.argtypes: object = ()

        def __call__(self) -> int:
            return self._value

    def __init__(self, abi: int) -> None:
        self.pvzstd_abi_version = self._Entry(abi)

    def __getattr__(self, name: str) -> object:
        msg = f"undefined symbol: {name}"
        raise AttributeError(msg)


@pytest.fixture
def decoy() -> str:
    """Return a loadable shared library that exports no ``pvzstd_*`` symbol."""
    names = {
        "win32": ("msvcrt.dll", "ucrtbase.dll"),
        "darwin": ("libz.dylib", "libm.dylib", "libSystem.B.dylib"),
    }.get(sys.platform, ("libz.so.1", "libz.so", "libm.so.6", "libc.so.6"))
    for name in names:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        return name
    pytest.skip("no loadable non-pvzstd shared library to offer the loader")
    raise AssertionError  # pragma: no cover - pytest.skip does not return


def test_an_abi_mismatch_is_reported_as_a_version_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A library one ABI behind is declined for its version, not for a symbol.

    Declaring every signature first would trip over whichever entry point the
    newer ABI added and blame that, which reads as a broken build rather than
    as the old library it is.
    """
    stale = _capi.ABI_VERSION - 1
    monkeypatch.setattr(_capi.ctypes, "CDLL", lambda _path: _NoSymbols(stale))

    lib, reason = _capi._try_candidate("stale-core")  # noqa: SLF001

    assert lib is None
    assert reason == f"ABI version {stale}, expected {_capi.ABI_VERSION}"
    assert "undefined symbol" not in reason


def test_a_library_asked_for_by_name_is_never_passed_over(monkeypatch: pytest.MonkeyPatch, decoy: str) -> None:
    """
    A library the caller named and the loader cannot use is an error.

    Falling through to the bundled copy would run the caller's work against a
    library they did not choose, report that library's path as the one in use,
    and say nothing about the one they asked for.
    """
    bundled = _capi.library_path()
    monkeypatch.setattr(_capi, "_candidate_paths", lambda: iter([(decoy, True), (bundled, False)]))
    monkeypatch.setattr(_capi, "_lib", None)
    monkeypatch.setattr(_capi, "_lib_path", None)
    monkeypatch.setattr(_capi, "_load_error", None)

    with pytest.raises(_capi.CoreUnavailableError) as raised:
        _capi._load()  # noqa: SLF001

    message = str(raised.value)
    assert decoy in message, f"refused without naming the library asked for: {message}"
    assert _capi._LIBRARY_ENV_VAR in message  # noqa: SLF001
    assert bundled not in message, "fell back to the bundled library instead of refusing"


def test_a_candidate_nobody_asked_for_is_still_passed_over(monkeypatch: pytest.MonkeyPatch, decoy: str) -> None:
    """
    Guesses stay guesses: one that does not load costs nothing but a turn.

    The counterpart to the test above -- the loader walks a list of places the
    library might be, and a miss there is ordinary, not a failure.
    """
    bundled = _capi.library_path()
    monkeypatch.setattr(_capi, "_candidate_paths", lambda: iter([(decoy, False), (bundled, False)]))
    monkeypatch.setattr(_capi, "_lib", None)
    monkeypatch.setattr(_capi, "_lib_path", None)
    monkeypatch.setattr(_capi, "_load_error", None)

    _capi._load()  # noqa: SLF001

    assert _capi._lib_path == bundled  # noqa: SLF001


def test_the_environment_variable_is_what_marks_a_candidate_chosen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PVZSTD_LIBRARY`` is the only thing that makes a path a request."""
    named = "/nowhere/libpvzstd.so"
    monkeypatch.setenv(_capi._LIBRARY_ENV_VAR, named)  # noqa: SLF001

    candidates = list(_capi._candidate_paths())  # noqa: SLF001

    assert candidates[0] == (named, True)
    assert not any(chosen for _path, chosen in candidates[1:])
