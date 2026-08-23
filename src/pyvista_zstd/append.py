"""
Incremental append + partial/columnar read for the ``.pv`` container.

Motivation
----------
:class:`pyvista_zstd.Writer` serialises a whole dataset in one shot. Appending
does not: the container's index lives at the *end* and every frame is addressed
by an absolute cumulative-compressed-byte offset, so a new block can be added by
copying the committed frames verbatim, writing the new ones after them, and
emitting a fresh index. No previously-written compressed frame is re-read or
re-compressed, which is what makes the cost proportional to what is added rather
than to the size of the file.

That edit is implemented in the C++ core; this module is its Python surface.

On-disk layout
--------------
::

    [frame 0 ........ frame N-1]                          # body: 2 frames per array
    [<QQ> x N: (cum_compressed_end, decompressed_size)]   # index
    [<Q>: N]                                              # trailing frame count

Appended arrays are recorded so they round-trip through an ordinary read: their
frames join the body and index, their names join the file metadata, and they are
registered as ``field_data`` on the root dataset (under the ``…__field_data``
suffix) so :pyattr:`Reader.available_field_arrays` and :meth:`Reader.read`
surface them with no reader change.

Crash safety
------------
The core stages the new file beside the original and commits it by rename, which
is atomic on POSIX and Windows. An interrupted append either leaves the original
fully intact or completes; it cannot leave a half-written index that destroys
committed blocks.

Partial / columnar read
-----------------------
:func:`read_array` (and :class:`AppendReader`) decompress exactly the two frames
of one named block -- never the rest of the file -- so a single array can be
loaded back without touching any other block.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pyvista_zstd import _capi
from pyvista_zstd.pyvista_zstd import SUPPORTED_READ_SUFFIXES
from pyvista_zstd.pyvista_zstd import ArrayInfo
from pyvista_zstd.pyvista_zstd import _shuffle_mode
from pyvista_zstd.pyvista_zstd import _warn_backend_deprecated

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from pyvista_zstd._capi import CoreReader
    from pyvista_zstd.pyvista_zstd import ShuffleSpec

__all__ = [
    "AppendReader",
    "append_arrays",
    "read_array",
]


def _checked_path(filename: Path | str) -> Path:
    """Return *filename* as a Path, refusing a suffix the format does not use."""
    path = Path(filename)
    if path.suffix not in SUPPORTED_READ_SUFFIXES:
        msg = f"Filename must end in one of {SUPPORTED_READ_SUFFIXES}, not '{path.suffix}'"
        raise ValueError(msg)
    return path


def append_arrays(
    filename: Path | str,
    arrays: dict[str, NDArray],
    *,
    level: int | None = None,
    shuffle: ShuffleSpec = False,
) -> None:
    """
    Append named arrays to an existing ``.pv`` file in place.

    Each array is stored as an independently-zstd-compressed ``field_data``
    block. Previously-written compressed frames are **never** re-read or
    re-compressed; only the small index plus the file and dataset metadata
    frames are rewritten, and the whole update is committed atomically.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to an existing ``.pv`` file (written by :func:`pyvista_zstd.write`
        or a previous :func:`append_arrays`).
    arrays : dict[str, numpy.ndarray]
        Mapping of ``name -> array``. Names must not already exist as a field
        array in the file. Arrays are stored verbatim (bit-exact round trip)
        regardless of dtype/shape.
    level : int, optional
        zstd compression level for the new blocks. Defaults to the file's
        recorded ``compression_level`` so appended blocks match the original.
    shuffle : {"auto", True, False}, default: False
        Optionally apply the reversible byte-shuffle pre-filter to the appended
        blocks (see :func:`pyvista_zstd.write`). Disabled by default; ``"auto"``
        shuffles a multibyte floating-point array only when a trial compression
        shows it shrinks the data. When any appended block is shuffled the file
        is promoted to format version 1; already-written frames are untouched
        and keep their own per-block encoding.

    Raises
    ------
    ValueError
        A name is already a field array in the file, or the suffix is not one
        the format uses.
    NotImplementedError
        The file is a MultiBlock, which has no root dataset to append to.

    Notes
    -----
    The appended blocks become visible as ``field_data`` arrays:
    :meth:`pyvista_zstd.Reader.read` returns them in ``grid.field_data`` and
    :pyattr:`pyvista_zstd.Reader.available_field_arrays` lists them. Use
    :func:`read_array` / :class:`AppendReader` to read one block back without
    decompressing the rest of the file.

    """
    path = _checked_path(filename)
    if not arrays:
        return

    # ``None`` means "whatever the file records"; the core resolves it.
    resolved = _capi.LEVEL_FROM_FILE if level is None else int(level)
    _capi.append_arrays(path, arrays, level=resolved, shuffle=_shuffle_mode(shuffle))


def read_array(
    filename: Path | str,
    name: str,
    *,
    backend: str | None = None,
) -> NDArray:
    """
    Read a single appended (field) array back without full decompression.

    Decompresses exactly the two frames of the named block - the rest of
    the file is never decompressed, so this stays cheap even on a
    multi-gigabyte container.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to a ``.pv`` file.
    name : str
        Field-array name (the key passed to :func:`append_arrays`, or any
        field-data key present in the file).
    backend : str, optional
        Deprecated and ignored; passing it raises a :class:`DeprecationWarning`.

    Returns
    -------
    numpy.ndarray
        The stored array, bit-exact.

    """
    _warn_backend_deprecated(backend)
    return AppendReader(filename).read_array(name)


class AppendReader:
    """
    Partial reader for ``.pv`` field-data blocks.

    Opens a ``.pv`` file and exposes its field arrays for *individual*
    read-back, decompressing only the requested block's two frames. The core
    parses the index and the per-array headers when the file is opened, so
    repeated single-block reads are cheap.

    The file stays open until :meth:`close` is called or the reader is
    garbage-collected. On Windows an open reader blocks
    :func:`append_arrays` from committing to the same path, so use the
    reader as a context manager when both happen in one scope.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to a ``.pv`` file.
    backend : str, optional
        Deprecated and ignored; passing it raises a :class:`DeprecationWarning`.

    Examples
    --------
    >>> from pyvista_zstd.append import AppendReader
    >>> with AppendReader("data.pv") as r:  # doctest: +SKIP
    ...     col = r.read_array("col_0001")

    """

    def __init__(
        self,
        filename: Path | str,
        *,
        backend: str | None = None,
    ) -> None:
        """
        Open ``filename`` for field-array reads.

        ``backend`` is deprecated and ignored.
        """
        _warn_backend_deprecated(backend)
        self._core: CoreReader | None = None
        self._path = _checked_path(filename)

    @property
    def _core_reader(self) -> CoreReader:
        """
        The C++ reader to serve everything from.

        Opened lazily and kept, so a second single-block read is cheap.
        """
        if self._core is None:
            self._core = _capi.CoreReader(self._path)
        return self._core

    @property
    def field_array_names(self) -> list[str]:
        """Names of field-data arrays available for partial read."""
        return self._core_reader.field_array_names()

    @property
    def field_array_info(self) -> dict[str, ArrayInfo]:
        """
        Mapping ``name -> ArrayInfo(shape, dtype)`` for field arrays.

        Taken from the frame header, which is what the payload is read against.
        """
        reader = self._core_reader
        info: dict[str, ArrayInfo] = {}
        for name in reader.field_array_names():
            index = reader.find_field(name)
            if index is None:  # pragma: no cover - metadata and index disagree
                continue
            header = reader._info(index)  # noqa: SLF001
            shape = tuple(int(header.shape[i]) for i in range(header.ndim))
            info[name] = ArrayInfo(shape=shape, dtype=str(np.dtype(header.dtype.decode("utf-8"))))
        return info

    def __contains__(self, name: str) -> bool:
        """Return whether field array ``name`` is present."""
        return name in self._core_reader.field_array_names()

    def read_array(self, name: str) -> NDArray:
        """Decompress and return the single field array ``name``."""
        index = self._core_reader.find_field(name)
        if index is None:
            available = sorted(self._core_reader.field_array_names())
            msg = f"field array {name!r} not found; available: {available}"
            raise KeyError(msg)
        return self._core_reader.read_at(index)

    def read_arrays(self, names: Iterable[str]) -> dict[str, NDArray]:
        """Decompress and return several field arrays by name."""
        return {n: self.read_array(n) for n in names}

    def close(self) -> None:
        """Release the file. Further reads reopen it."""
        if self._core is not None:
            self._core.close()
            self._core = None

    def __enter__(self) -> AppendReader:  # noqa: PYI034
        """Return self; the file opens on first use."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the file."""
        self.close()
