"""
``ctypes`` binding to the ``pvzstd`` C ABI.

The C++ core is a plain shared library exposing a C ABI, loaded here with
:mod:`ctypes`. It is required: there is no second implementation, so on a
machine without the library every entry point raises
:class:`CoreUnavailableError` carrying the load diagnostics.
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
    "ArrayExistsError",
    "ContainerFormatError",
    "ContainerShapeError",
    "CoreReader",
    "CoreUnavailableError",
    "PvzstdError",
    "UnsupportedFileVersionError",
    "append_arrays",
    "available",
    "library_path",
    "load_error",
    "max_file_version",
]

ABI_VERSION = 7
"""ABI this binding speaks. A library reporting anything else is refused."""

DTYPE_LEN = 16

# Mirrors PVZ_LEVEL_FROM_FILE: append at whatever level the container records,
# so added blocks match the ones already there.
LEVEL_FROM_FILE = -1000

# Mirrors PVZ_THREADS_AUTO: let the C++ core pick from hardware concurrency.
THREADS_AUTO = -2

# Mirrors pvz_shuffle_mode. AUTO is not "decide in Python and pass a bool": the
# core runs the trial compression itself, so the policy has one home.
SHUFFLE_NEVER = 0
SHUFFLE_ALWAYS = 1
SHUFFLE_AUTO = 2

# Names *which* library to load; every behavioural knob is a keyword argument.
_LIBRARY_ENV_VAR = "PVZSTD_LIBRARY"

_STATUS_OK = 0
_STATUS_FORMAT = 2
_STATUS_RANGE = 4
_STATUS_FILTER = 6
_STATUS_UNSUPPORTED = 8
_STATUS_EXISTS = 9
_STATUS_VERSION = 10
_STATUS_NAMES = {
    1: "I/O error: file missing, unreadable, or truncated",
    2: "format error: the trailer or a frame header did not parse",
    3: "zstd error: a frame or a compression parameter was rejected",
    4: "range error: index out of range, or destination too small",
    5: "out of memory",
    6: "unsupported filter: this build cannot reverse the on-disk transform",
    7: "invalid argument",
    8: "this operation cannot serve a container of that shape",
    9: "an array of that name is already in the container",
    10: "unsupported file version: the container is newer than this build decodes",
}


class PvzstdError(RuntimeError):
    """The C++ library reported a failure."""

    def __init__(self, status: int, detail: str = "", message: str | None = None) -> None:
        if message is None:
            described = _STATUS_NAMES.get(status, f"unknown status {status}")
            message = f"{described} ({detail})" if detail else described
        super().__init__(message)
        self.status = status


class ContainerFormatError(PvzstdError):
    """
    The bytes are not a ``.pv`` container this build can parse.

    Named separately because callers act on it: such a file is corrupt or not
    a container at all, rather than one that cannot serve the operation.
    """


class UnsupportedFilterError(PvzstdError, ValueError):
    """
    An array carries a byte filter this build cannot reverse.

    A :class:`ValueError`; the status code is on ``.status``.
    """

    def __init__(self, filter_id: int, name: str) -> None:
        super().__init__(
            _STATUS_FILTER,
            message=(
                f"Unsupported per-array filter id {filter_id} for array '{name}'. Upgrade `pyvista-zstd` to read it."
            ),
        )


class UnsupportedFileVersionError(PvzstdError, ValueError):
    """
    The container is stamped with a format version this build cannot decode.

    A :class:`ValueError`. The core makes the comparison; this restates the
    verdict in the wording callers match on.
    """

    def __init__(self, found: int, supported: int) -> None:
        super().__init__(
            _STATUS_VERSION,
            message=(
                f"The file version {found} of this pyvista-zstd file is newer than the "
                f"version supported by this library {supported}. "
                "Upgrade `pyvista-zstd` to read it."
            ),
        )
        self.found = found
        self.supported = supported


class ContainerShapeError(PvzstdError, NotImplementedError):
    """
    The operation has nothing to work on in a container of this shape.

    A :class:`NotImplementedError` -- appending to a MultiBlock, for instance.
    """


class ArrayExistsError(PvzstdError, ValueError):
    """
    The name is taken, and the call would have overwritten it.

    A :class:`ValueError`: the argument is wrong for the file it was given.
    """


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


class _AppendArray(Structure):
    # Field order is the C declaration's; ctypes reproduces its padding from that,
    # and a reordering here would misread every member past the first difference.
    _fields_ = (
        ("name", c_char_p),
        ("dtype", c_char_p),
        ("dtype_name", c_char_p),
        ("shape", POINTER(c_uint64)),
        ("ndim", c_uint32),
        ("data", c_void_p),
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

    The bundled copy inside the package wins over the system search path, so
    an installed wheel cannot be silently served by an unrelated build.
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
    _bind_reader(lib)
    _bind_writer(lib)


def _bind_reader(lib: ctypes.CDLL) -> None:
    """Reader-side entry points, plus the two that belong to no handle."""
    lib.pvz_abi_version.restype = c_uint32
    lib.pvz_abi_version.argtypes = []

    lib.pvz_status_message.restype = c_char_p
    lib.pvz_status_message.argtypes = [c_int]

    lib.pvz_open.restype = c_int
    lib.pvz_open.argtypes = [c_char_p, POINTER(c_void_p)]

    lib.pvz_open_versioned.restype = c_int
    lib.pvz_open_versioned.argtypes = [c_char_p, POINTER(c_void_p), POINTER(c_uint32)]

    lib.pvz_max_file_version.restype = c_uint32
    lib.pvz_max_file_version.argtypes = []

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

    lib.pvz_metadata_count.restype = c_uint64
    lib.pvz_metadata_count.argtypes = [c_void_p]

    lib.pvz_metadata_name_at.restype = c_char_p
    lib.pvz_metadata_name_at.argtypes = [c_void_p, c_uint64]

    lib.pvz_metadata_json_at.restype = c_char_p
    lib.pvz_metadata_json_at.argtypes = [c_void_p, c_uint64]

    lib.pvz_frame_count.restype = c_uint64
    lib.pvz_frame_count.argtypes = [c_void_p]

    lib.pvz_frame_sizes.restype = c_int
    lib.pvz_frame_sizes.argtypes = [c_void_p, POINTER(c_uint64), POINTER(c_uint64), c_uint64]


def _bind_writer(lib: ctypes.CDLL) -> None:
    """Writer-side entry points, plus the handle-less append."""
    lib.pvz_writer_create.restype = c_int
    lib.pvz_writer_create.argtypes = [POINTER(c_void_p)]

    lib.pvz_writer_free.restype = None
    lib.pvz_writer_free.argtypes = [c_void_p]

    lib.pvz_writer_set_level.restype = c_int
    lib.pvz_writer_set_level.argtypes = [c_void_p, c_int]

    lib.pvz_writer_set_threads.restype = c_int
    lib.pvz_writer_set_threads.argtypes = [c_void_p, c_int]

    lib.pvz_writer_set_shuffle.restype = c_int
    lib.pvz_writer_set_shuffle.argtypes = [c_void_p, c_int]

    lib.pvz_writer_set_fixed_width_cells.restype = c_int
    lib.pvz_writer_set_fixed_width_cells.argtypes = [c_void_p, c_int]

    lib.pvz_writer_add_array_borrowed.restype = c_int
    lib.pvz_writer_add_array_borrowed.argtypes = [
        c_void_p,
        c_char_p,
        c_char_p,
        POINTER(c_uint64),
        c_uint32,
        c_void_p,
        c_uint64,
    ]

    lib.pvz_writer_write.restype = c_int
    lib.pvz_writer_write.argtypes = [c_void_p, c_char_p]

    lib.pvz_append_arrays.restype = c_int
    lib.pvz_append_arrays.argtypes = [c_char_p, POINTER(_AppendArray), c_uint64, c_int, c_int]


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
            # Binding is inside the guard because a version bump that ADDS a
            # symbol makes an older library fail here, at the missing
            # attribute, before the check below can name the mismatch.
            _bind(lib)
            found = int(lib.pvz_abi_version())
        except (OSError, AttributeError) as exc:
            attempts.append(f"{candidate}: {exc}")
            continue
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

    Reports the candidates that were tried and what each of them said.

    Returns
    -------
    str | None

    """
    if available():
        return None
    return _load_error


def max_file_version() -> int:
    """
    Return the highest container ``file_version`` the loaded core can decode.

    Read from the library, never copied here.

    Returns
    -------
    int

    """
    return int(_load().pvz_max_file_version())


def _check(status: int, detail: str = "") -> None:
    if status == _STATUS_OK:
        return
    if status == _STATUS_FORMAT:
        raise ContainerFormatError(status, detail)
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
        # The versioned form so a refusal can name the version it refused; the
        # decision to refuse is the core's, and is already made by the time this
        # returns.
        found = c_uint32(0)
        status = lib.pvz_open_versioned(str(path).encode("utf-8"), byref(handle), byref(found))
        if status == _STATUS_VERSION:
            raise UnsupportedFileVersionError(int(found.value), int(lib.pvz_max_file_version()))
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
            # The destination's size, not the file's, so an over-long payload is
            # reported rather than written past the end of this array.
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

        All wanted frames go to the core in one call so it can decompress them
        in parallel.

        Parameters
        ----------
        keep : set[str] | None, optional
            Names to decompress. When omitted, every array is read; arrays
            that are not kept are never decompressed.
        n_threads : int, optional
            Workers to spread the frames over. The default picks from the
            total size; 0 or 1 decompresses inline and negative uses every
            core, matching :meth:`CoreWriter.set_threads`.

        Returns
        -------
        dict[str, numpy.ndarray]

        """
        handle = self._live
        dtypes: dict[bytes, np.dtype[Any]] = {}
        empty = np.empty

        # Every header in one crossing.
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

    def metadata_documents(self) -> list[tuple[str, str]]:
        """
        Return every metadata document as ``(frame_name, json)``, in file order.

        The UID lives in the frame name, not in the document, and rebuilding a
        MultiBlock tree needs both.
        """
        live = self._live
        count = int(self._lib.pvz_metadata_count(live))
        documents = []
        for index in range(count):
            name = self._lib.pvz_metadata_name_at(live, c_uint64(index))
            raw = self._lib.pvz_metadata_json_at(live, c_uint64(index))
            if name is None or raw is None:  # pragma: no cover - count said otherwise
                continue
            documents.append((name.decode("utf-8"), raw.decode("utf-8")))
        return documents

    def frame_sizes(self) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
        """
        Return ``(decompressed, compressed)`` sizes per frame, in file order.

        Two frames per array, ``(header, payload)``, and the metadata frames are
        included -- so these index the file, not :meth:`array_names`.
        """
        count = int(self._lib.pvz_frame_count(self._live))
        decompressed = np.empty(count, dtype=np.uint64)
        compressed = np.empty(count, dtype=np.uint64)
        _check(
            self._lib.pvz_frame_sizes(
                self._live,
                decompressed.ctypes.data_as(POINTER(c_uint64)),
                compressed.ctypes.data_as(POINTER(c_uint64)),
                c_uint64(count),
            )
        )
        return decompressed, compressed


class CoreWriter:
    """
    Write a container through the C++ core.

    Arrays are staged in call order and emitted on :meth:`write`. The core
    owns the format decisions; this class stages buffers.

    Examples
    --------
    >>> import numpy as np
    >>> from pyvista_zstd import _capi
    >>> with _capi.CoreWriter() as writer:  # doctest: +SKIP
    ...     writer.add_array("points", np.zeros(3))
    ...     writer.write("out.pv")

    """

    def __init__(self) -> None:
        lib = _load()
        self._lib = lib
        self._handle: c_void_p | None = None
        # Staged buffers must outlive write(): the core borrows them, so a
        # freed temporary is a dangling read at write time rather than an
        # error at the call that caused it.
        self._pinned: list[Any] = []

        handle = c_void_p()
        _check(lib.pvz_writer_create(byref(handle)))
        self._handle = handle

    def __enter__(self) -> CoreWriter:  # noqa: PYI034
        """Return self; the writer is already created."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the writer."""
        self.close()

    def close(self) -> None:
        """Release the underlying writer, if it is still open."""
        if self._handle is not None:
            self._lib.pvz_writer_free(self._handle)
            self._handle = None
        self._pinned.clear()

    def __del__(self) -> None:
        """Release the writer on collection."""
        with contextlib.suppress(Exception):
            self.close()

    @property
    def _live(self) -> c_void_p:
        if self._handle is None:
            msg = "writer is closed"
            raise ValueError(msg)
        return self._handle

    def set_level(self, level: int) -> None:
        """Set the zstd compression level."""
        _check(self._lib.pvz_writer_set_level(self._live, int(level)))

    def set_threads(self, n_threads: int) -> None:
        """Set the worker count; ``THREADS_AUTO`` sizes it from the payload."""
        _check(self._lib.pvz_writer_set_threads(self._live, int(n_threads)))

    def set_shuffle(self, mode: int) -> None:
        """Set the byte-shuffle policy to one of the ``SHUFFLE_*`` modes."""
        _check(self._lib.pvz_writer_set_shuffle(self._live, int(mode)))

    def set_fixed_width_cells(self, *, enabled: bool) -> None:
        """Record whether any dataset used the fixed-width cell encoding."""
        _check(self._lib.pvz_writer_set_fixed_width_cells(self._live, 1 if enabled else 0))

    def add_array(self, name: str, arr: NDArray[Any]) -> None:
        """
        Stage one named array, in frame order.

        Made contiguous if it is not already, since the ABI takes one pointer.
        The core borrows the buffer, so it is pinned until :meth:`write` or
        :meth:`close`.
        """
        buf = np.ascontiguousarray(arr)
        self._pinned.append(buf)
        shape = (c_uint64 * max(buf.ndim, 1))(*buf.shape)
        _check(
            self._lib.pvz_writer_add_array_borrowed(
                self._live,
                name.encode("utf-8"),
                buf.dtype.str.encode("ascii"),
                shape,
                c_uint32(buf.ndim),
                buf.ctypes.data_as(c_void_p),
                c_uint64(buf.nbytes),
            ),
            name,
        )

    def write(self, path: Path | str) -> None:
        """Compress and emit every staged array to *path*."""
        _check(self._lib.pvz_writer_write(self._live, str(path).encode("utf-8")), str(path))


def _name_collision(path: Path | str, offered: list[str]) -> str:
    """
    Say which name the core refused, by asking it what the file already holds.

    The status says a name was taken but not which one. Error path only.
    """
    try:
        with CoreReader(path) as reader:
            taken = set(reader.field_array_names())
    except PvzstdError:  # pragma: no cover - the append just read this file
        taken = set()

    clashed = [name for name in offered if name in taken]
    if clashed:
        return (
            f"field array {clashed[0]!r} already exists in {Path(path).name}; "
            f"append_arrays does not overwrite existing blocks."
        )
    # The core refuses a name repeated within one call too, but a dict cannot
    # carry one, so reaching here means the two sides disagree about what the
    # file holds. Report that rather than inventing a name to blame.
    return (
        f"the core refused a name already in {Path(path).name}, but none of {sorted(offered)} "
        f"is among the field arrays it reports; append_arrays does not overwrite existing blocks."
    )


def append_arrays(
    path: Path | str,
    arrays: dict[str, NDArray[Any]],
    *,
    level: int = LEVEL_FROM_FILE,
    shuffle: int = SHUFFLE_NEVER,
) -> None:
    """
    Append field arrays to an existing container through the C++ core.

    Every decision the edit encodes -- which blocks are copied by offset, which
    two metadata frames are regenerated, whether a block is shuffled, what file
    version that implies, and the commit-by-rename -- belongs to the core. This
    marshals buffers across the ABI and translates the refusals into the
    exceptions this package has always raised.

    Parameters
    ----------
    path : pathlib.Path | str
        An existing ``.pv`` container.
    arrays : dict[str, numpy.ndarray]
        Mapping of bare name to array. The core adds the dataset-UID prefix and
        the field-data suffix the frame carries.
    level : int, optional
        zstd level for the new blocks. ``LEVEL_FROM_FILE`` reads it out of the
        container so appended blocks match what is already there.
    shuffle : int, optional
        One of the ``SHUFFLE_*`` modes. Under ``SHUFFLE_AUTO`` the core runs the
        trial compression itself, so the policy has one home.

    Raises
    ------
    ArrayExistsError
        A name is already a field array in the file, or is repeated in *arrays*.
    ContainerShapeError
        The container is a MultiBlock, which has no root dataset to append to.

    """
    lib = _load()
    if not arrays:
        return

    # Both the buffers and the encoded strings have to outlive the call: every
    # one of them is handed over as a borrowed pointer, and a temporary freed at
    # the end of the loop iteration would be a dangling read inside the core
    # rather than an error at the line that caused it.
    pinned: list[Any] = []
    items = (_AppendArray * len(arrays))()
    for slot, (name, arr) in enumerate(arrays.items()):
        buf = np.ascontiguousarray(arr)
        shape = (c_uint64 * max(buf.ndim, 1))(*buf.shape)
        encoded = (
            name.encode("utf-8"),
            buf.dtype.str.encode("ascii"),
            # The metadata spelling, which is not derivable from the header one
            # without carrying a copy of numpy's dtype-name table.
            str(buf.dtype).encode("ascii"),
        )
        pinned.append((buf, shape, encoded))

        items[slot].name, items[slot].dtype, items[slot].dtype_name = encoded
        items[slot].shape = shape
        items[slot].ndim = c_uint32(buf.ndim)
        items[slot].data = buf.ctypes.data
        items[slot].nbytes = c_uint64(buf.nbytes)

    status = lib.pvz_append_arrays(
        str(path).encode("utf-8"), items, c_uint64(len(arrays)), c_int(level), c_int(shuffle)
    )
    if status == _STATUS_VERSION:
        # The append status says "too new" but not how new; the open path reports
        # the number even when it refuses, so ask it rather than re-parse the
        # trailer here. Only reached on the error path.
        found = c_uint32(0)
        handle = c_void_p()
        lib.pvz_open_versioned(str(path).encode("utf-8"), byref(handle), byref(found))
        if handle:  # pragma: no cover - append refused it, so the open does too
            lib.pvz_close(handle)
        raise UnsupportedFileVersionError(int(found.value), int(lib.pvz_max_file_version()))
    if status == _STATUS_EXISTS:
        raise ArrayExistsError(status, message=_name_collision(path, list(arrays)))
    if status == _STATUS_UNSUPPORTED:
        raise ContainerShapeError(status, message="appending to MultiBlock .pv files is not supported.")
    _check(status, str(path))
