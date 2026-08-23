"""
``ctypes`` binding to the ``pvzstd`` C ABI.

There is no compiled extension module here and no binding framework. The
C++ core is a plain shared library exposing a C ABI, and this module loads
it with :mod:`ctypes`. That choice is what lets the same library be consumed as
a C++ submodule, cross-compiled to WebAssembly, and shipped in a wheel without
three separate binding layers going out of step.

The core is required, not preferred. There is no second implementation to
fall to, so a machine without the library cannot read or write these files
and says so: every entry point raises :class:`CoreUnavailableError` carrying
the load diagnostics. :func:`available` remains for reporting -- it answers
"can this machine work", never "which implementation should run".
"""

from __future__ import annotations

import contextlib
import ctypes
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_int64
from ctypes import c_uint8
from ctypes import c_uint32
from ctypes import c_uint64
from ctypes import c_void_p
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

__all__ = [
    "ABI_VERSION",
    "CoreReader",
    "CoreUnavailableError",
    "PvzstdError",
    "available",
    "library_path",
    "load_error",
]

ABI_VERSION = 2
"""ABI this binding speaks. A library reporting anything else is refused."""

DTYPE_LEN = 16

# Mirrors PVZ_THREADS_AUTO: let the C++ core pick from hardware concurrency.
THREADS_AUTO = -2

# Names *which* library to load; every behavioural knob is a keyword argument.
_LIBRARY_ENV_VAR = "PVZSTD_LIBRARY"

_STATUS_OK = 0
_STATUS_FILTER = 6
_STATUS_NAMES = {
    1: "I/O error: file missing, unreadable, or truncated",
    2: "format error: the trailer or a frame header did not parse",
    3: "zstd error: a frame or a compression parameter was rejected",
    4: "range error: index out of range, or destination too small",
    5: "out of memory",
    6: "unsupported filter: this build cannot reverse the on-disk transform",
    7: "invalid argument",
}


class PvzstdError(RuntimeError):
    """The C++ library reported a failure."""

    def __init__(self, status: int, detail: str = "", message: str | None = None) -> None:
        if message is None:
            described = _STATUS_NAMES.get(status, f"unknown status {status}")
            message = f"{described} ({detail})" if detail else described
        super().__init__(message)
        self.status = status


class UnsupportedFilterError(PvzstdError, ValueError):
    """
    An array carries a byte filter this build cannot reverse.

    Inherits from :class:`ValueError` because that is what this condition has
    raised since before the core existed. Nothing about the error changed when
    the reader moved into C++, so neither should what callers catch.
    """

    def __init__(self, filter_id: int, name: str) -> None:
        # Wording preserved from the reader this replaced; the code stays on .status.
        super().__init__(
            _STATUS_FILTER,
            message=(
                f"Unsupported per-array filter id {filter_id} for array '{name}'. Upgrade `pyvista-zstd` to read it."
            ),
        )


class CoreUnavailableError(RuntimeError):
    """The C++ library could not be loaded on this machine."""


class _ArrayInfo(Structure):
    _fields_ = (
        ("name", c_char_p),
        ("shape", POINTER(c_uint64)),
        ("ndim", c_uint32),
        ("filter_id", c_uint8),
        ("dtype", c_char * (DTYPE_LEN + 1)),
        ("nbytes", c_uint64),
    )


def _candidate_names() -> list[str]:
    """Return shared-library file names to try, most specific first."""
    if sys.platform == "win32":
        return ["pvzstd.dll", "libpvzstd.dll"]
    if sys.platform == "darwin":
        return ["libpvzstd.dylib"]
    return ["libpvzstd.so"]


def _candidate_paths() -> Iterator[str]:
    """
    Yield places the shared library may live, in priority order.

    The bundled copy inside the package wins over anything on the system
    search path, so an installed wheel is self-contained and cannot be
    silently served by an unrelated build sitting in the loader path.
    """
    override = os.environ.get(_LIBRARY_ENV_VAR)
    if override:
        yield override

    here = Path(__file__).parent
    for directory in (here / "lib", here):
        for name in _candidate_names():
            candidate = directory / name
            if candidate.exists():
                yield str(candidate)

    # Last: let the platform loader search.
    yield from _candidate_names()


_lib: ctypes.CDLL | None = None
_lib_path: str | None = None
_load_error: str | None = None


def _bind(lib: ctypes.CDLL) -> None:
    """Declare every signature. ctypes defaults are wrong for 64-bit returns."""
    lib.pvz_abi_version.restype = c_uint32
    lib.pvz_abi_version.argtypes = []

    lib.pvz_status_message.restype = c_char_p
    lib.pvz_status_message.argtypes = [c_int]

    lib.pvz_open.restype = c_int
    lib.pvz_open.argtypes = [c_char_p, POINTER(c_void_p)]

    lib.pvz_close.restype = None
    lib.pvz_close.argtypes = [c_void_p]

    # Without an explicit restype ctypes truncates this to a C int.
    lib.pvz_array_count.restype = c_uint64
    lib.pvz_array_count.argtypes = [c_void_p]

    lib.pvz_array_info_at.restype = c_int
    lib.pvz_array_info_at.argtypes = [c_void_p, c_uint64, POINTER(_ArrayInfo)]

    lib.pvz_array_info_range.restype = c_int
    lib.pvz_array_info_range.argtypes = [c_void_p, c_uint64, c_uint64, POINTER(_ArrayInfo)]

    lib.pvz_find_array.restype = c_int64
    lib.pvz_find_array.argtypes = [c_void_p, c_char_p]

    lib.pvz_read_array_at.restype = c_int
    lib.pvz_read_array_at.argtypes = [c_void_p, c_uint64, c_void_p, c_uint64]

    lib.pvz_read_arrays.restype = c_int
    lib.pvz_read_arrays.argtypes = [
        c_void_p,
        POINTER(c_uint64),
        c_uint64,
        POINTER(c_void_p),
        POINTER(c_uint64),
        c_int,
    ]

    lib.pvz_field_array_count.restype = c_uint64
    lib.pvz_field_array_count.argtypes = [c_void_p]

    lib.pvz_field_array_name_at.restype = c_char_p
    lib.pvz_field_array_name_at.argtypes = [c_void_p, c_uint64]

    lib.pvz_find_field_array.restype = c_int64
    lib.pvz_find_field_array.argtypes = [c_void_p, c_char_p]

    lib.pvz_ds_metadata_json.restype = c_char_p
    lib.pvz_ds_metadata_json.argtypes = [c_void_p]

    lib.pvz_file_metadata_json.restype = c_char_p
    lib.pvz_file_metadata_json.argtypes = [c_void_p]


def _load() -> ctypes.CDLL:
    global _lib, _lib_path, _load_error  # noqa: PLW0603 - module-level cache

    if _lib is not None:
        return _lib
    if _load_error is not None:
        raise CoreUnavailableError(_load_error)

    attempts: list[str] = []
    for candidate in _candidate_paths():
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as exc:
            attempts.append(f"{candidate}: {exc}")
            continue

        _bind(lib)
        found = int(lib.pvz_abi_version())
        if found != ABI_VERSION:
            # Worse than a missing one: this module's struct layout would be read
            # against a different contract.
            attempts.append(f"{candidate}: ABI version {found}, expected {ABI_VERSION}")
            continue

        _lib = lib
        _lib_path = candidate
        return lib

    _load_error = "could not load the pvzstd shared library. Tried:\n  " + "\n  ".join(attempts)
    raise CoreUnavailableError(_load_error)


def available() -> bool:
    """
    Return whether the C++ core can be loaded.

    Returns
    -------
    bool
        True when :class:`CoreReader` will work on this machine.

    """
    try:
        _load()
    except CoreUnavailableError:
        return False
    return True


def library_path() -> str | None:
    """
    Return the file the C++ core was loaded from, or None.

    Useful when diagnosing which of several builds is actually in play.

    Returns
    -------
    str | None

    """
    if _lib is None and not available():
        return None
    return _lib_path


def load_error() -> str | None:
    """
    Return why the C++ core could not be loaded, or None if it did.

    ``available()`` collapses every reason to False, which answers "can this
    machine work" and nothing else: a missing
    library, a library built against a different ABI, and one that failed to
    link all look identical. This reports the candidates that were tried and
    what each of them said.

    Returns
    -------
    str | None

    """
    if available():
        return None
    return _load_error


def _check(status: int, detail: str = "") -> None:
    if status != _STATUS_OK:
        raise PvzstdError(status, detail)


class CoreReader:
    """
    Read a container through the C++ core.

    Opening parses only the trailer and the per-array headers; payloads are
    decompressed on demand. Reading two arrays out of a large file therefore
    costs two frames, not the whole file.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to a ``.pv`` or ``.zvtk`` file.

    Examples
    --------
    >>> from pyvista_zstd import _capi
    >>> with _capi.CoreReader("dataset.pv") as reader:  # doctest: +SKIP
    ...     names = reader.names()

    """

    def __init__(self, path: Path | str) -> None:
        lib = _load()
        self._lib = lib
        self._handle: c_void_p | None = None

        handle = c_void_p()
        status = lib.pvz_open(str(path).encode("utf-8"), byref(handle))
        _check(status, str(path))
        self._handle = handle

    # PYI034 wants `Self`, which is 3.11+; this package supports 3.10.
    def __enter__(self) -> CoreReader:  # noqa: PYI034
        """Return self; the reader is already open."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the C++ reader."""
        self.close()

    def close(self) -> None:
        """Release the C++ reader. Safe to call more than once."""
        if self._handle is not None:
            self._lib.pvz_close(self._handle)
            self._handle = None

    def __del__(self) -> None:
        """Release the C++ reader if the caller forgot to."""
        # ctypes handles are not garbage-collected, so a dropped reader would leak
        # the mapping until exit. Errors here are unreportable during teardown.
        with contextlib.suppress(Exception):
            self.close()

    @property
    def _live(self) -> c_void_p:
        if self._handle is None:
            msg = "operation on a closed CoreReader"
            raise ValueError(msg)
        return self._handle

    def __len__(self) -> int:
        """Return the number of arrays, excluding the JSON metadata frames."""
        return int(self._lib.pvz_array_count(self._live))

    def _info(self, index: int) -> _ArrayInfo:
        info = _ArrayInfo()
        _check(self._lib.pvz_array_info_at(self._live, c_uint64(index), byref(info)), f"index {index}")
        return info

    def names(self) -> list[str]:
        """
        Return every array name, in frame order.

        Returns
        -------
        list[str]
            Names as stored, including the 16-character UID prefix.

        """
        return [self._info(i).name.decode("utf-8") for i in range(len(self))]

    def read_at(self, index: int) -> NDArray[Any]:
        """
        Decompress one array by index.

        Parameters
        ----------
        index : int
            Position in frame order.

        Returns
        -------
        numpy.ndarray
            The array, with its filter already reversed by the C++ core.

        """
        info = self._info(index)
        dtype = np.dtype(info.dtype.decode("utf-8"))
        shape = tuple(info.shape[i] for i in range(info.ndim))

        out = np.empty(shape, dtype=dtype)
        if info.nbytes:
            # The destination's size, not the file's: passing the declared payload
            # size would authorise exactly what the core is about to produce, so a
            # payload larger than its announced shape would be written past the end
            # of this array rather than reported.
            status = self._lib.pvz_read_array_at(
                self._live,
                c_uint64(index),
                c_void_p(out.ctypes.data),
                c_uint64(out.nbytes),
            )
            # Restates the core's verdict as the exception this library raises.
            if status == _STATUS_FILTER:
                raise UnsupportedFilterError(info.filter_id, info.name.decode("utf-8"))
            _check(status, info.name.decode("utf-8"))
        return out

    def find(self, name: str) -> int | None:
        """
        Return the index of *name*, or None when the file has no such array.

        Parameters
        ----------
        name : str
            Full stored name, including the UID prefix.

        Returns
        -------
        int | None

        """
        found = int(self._lib.pvz_find_array(self._live, name.encode("utf-8")))
        return None if found < 0 else found

    def field_array_names(self) -> list[str]:
        """
        Return the root dataset's field-array names, in metadata order.

        These are bare names -- the UID prefix and the ``__field_data`` suffix
        the frame carries are not part of them, so they are what a caller
        passed to :func:`~pyvista_zstd.append_arrays`.

        Returns
        -------
        list[str]
            Empty for a MultiBlock container, which has no single root dataset.

        """
        count = int(self._lib.pvz_field_array_count(self._live))
        names = []
        for i in range(count):
            raw = self._lib.pvz_field_array_name_at(self._live, c_uint64(i))
            names.append(raw.decode("utf-8"))
        return names

    def find_field(self, name: str) -> int | None:
        """
        Return the array index of field array *name*, or None.

        Parameters
        ----------
        name : str
            Bare field-array name, without the UID prefix or suffix.

        Returns
        -------
        int | None
            An index usable with :meth:`read_at`.

        """
        found = int(self._lib.pvz_find_field_array(self._live, name.encode("utf-8")))
        return None if found < 0 else found

    def read_arrays(self, keep: set[str] | None = None, n_threads: int = THREADS_AUTO) -> dict[str, NDArray[Any]]:
        """
        Decompress arrays into a name-keyed mapping.

        All wanted frames are handed to the C++ core in one call so it can
        decompress them in parallel. Doing this one array at a time leaves
        every core but one idle, which measured *slower* than the Python
        reader this replaced -- that one batched through zstd's own threaded
        decompressor.

        Parameters
        ----------
        keep : set[str] | None, optional
            Names to decompress. When omitted, every array is read. Arrays
            that are not kept are never decompressed at all -- the saving is
            the whole frame, not just the copy.
        n_threads : int, optional
            Workers to spread the frames over. The default follows the
            hardware concurrency; 1 decompresses inline.

        Returns
        -------
        dict[str, numpy.ndarray]

        """
        # Hoisted out of the loop on purpose: profiling put more time in this
        # Python body than in the C++ decompression it wraps.
        handle = self._live
        dtypes: dict[bytes, np.dtype[Any]] = {}
        empty = np.empty

        # Every header in one crossing: a foreign call per array outweighed the
        # decompression it was preparing.
        n_arrays = int(self._lib.pvz_array_count(handle))
        infos = (_ArrayInfo * n_arrays)()
        if n_arrays:
            _check(self._lib.pvz_array_info_range(handle, 0, n_arrays, infos))

        wanted: list[tuple[int, str, NDArray[Any]]] = []
        for index in range(n_arrays):
            info = infos[index]
            name = info.name.decode("utf-8")
            if keep is not None and name not in keep:
                continue
            raw_dtype = info.dtype
            dtype = dtypes.get(raw_dtype)
            if dtype is None:
                dtype = np.dtype(raw_dtype.decode("utf-8"))
                dtypes[raw_dtype] = dtype
            # Slicing converts the dimensions in one C-level step.
            shape = tuple(info.shape[: info.ndim])
            wanted.append((index, name, empty(shape, dtype=dtype)))

        payloads = [(i, n, a) for i, n, a in wanted if a.nbytes]
        if payloads:
            count = len(payloads)
            # One pass: three generator expressions walked the list three times.
            indices = (c_uint64 * count)()
            dsts = (c_void_p * count)()
            sizes = (c_uint64 * count)()
            for slot, (i, _, arr) in enumerate(payloads):
                indices[slot] = i
                # C-contiguous, so the core writes straight into the final array.
                dsts[slot] = arr.ctypes.data
                sizes[slot] = arr.nbytes
            status = self._lib.pvz_read_arrays(handle, indices, c_uint64(count), dsts, sizes, c_int(n_threads))
            if status == _STATUS_FILTER:
                self._raise_filter_error(payloads)
            _check(status)

        return {name: arr for _, name, arr in wanted}

    def _raise_filter_error(self, payloads: list[tuple[int, str, NDArray[Any]]]) -> None:
        """Re-read one at a time to name the array the batch call rejected."""
        for index, name, arr in payloads:
            info = self._info(index)
            if (
                self._lib.pvz_read_array_at(
                    self._live, c_uint64(index), c_void_p(arr.ctypes.data), c_uint64(arr.nbytes)
                )
                == _STATUS_FILTER
            ):
                raise UnsupportedFilterError(info.filter_id, name)
        raise PvzstdError(_STATUS_FILTER)  # pragma: no cover - batch said yes, singles said no

    @property
    def ds_metadata_json(self) -> str | None:
        """Return the dataset metadata JSON document, or None."""
        raw = self._lib.pvz_ds_metadata_json(self._live)
        return None if raw is None else raw.decode("utf-8")

    @property
    def file_metadata_json(self) -> str | None:
        """Return the file metadata JSON document, or None."""
        raw = self._lib.pvz_file_metadata_json(self._live)
        return None if raw is None else raw.decode("utf-8")
