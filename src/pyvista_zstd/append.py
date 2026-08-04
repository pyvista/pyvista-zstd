"""
Incremental append + partial/columnar read for the ``.pv`` container.

Motivation
----------
The upstream :class:`pyvista_zstd.Writer` serialises a whole dataset in
one shot: every array is independently zstd-compressed into a pair of
frames (``[packed-metadata-frame][raw-array-frame]``), the frames are
concatenated, and a fixed-width footer table plus a trailing
``<Q>`` frame-count is written at the very end of the file.

Because the footer lives at the *end* and every body frame is addressed
by an absolute cumulative-compressed-byte offset, a new block can be
appended by:

1. truncating the file at the start of the existing footer
   (``frame_ends[-1]``), leaving every previously-committed compressed
   frame byte-for-byte untouched,
2. writing the new block's compressed frames after it,
3. writing a *new* footer that covers the old frames (offsets unchanged)
   plus the new ones, then the new ``<Q>`` count.

This is what :func:`append_arrays` does.  No previously-written
compressed frame is ever re-read or re-compressed.

On-disk layout (unchanged from upstream)
-----------------------------------------
::

    [frame 0 ........ frame N-1]          # body: 2 frames per array
    [<QQ> x N: (cum_compressed_end, decompressed_size)]   # footer table
    [<Q>: N]                              # trailing frame count

Appended arrays are recorded two ways so they round-trip through the
*unmodified* upstream reader:

* their two frames are added to the body and footer,
* their names are added to the file-metadata ``frame_names`` list, and
* they are registered as ``field_data`` arrays on the root dataset's
  :class:`~pyvista_zstd.pyvista_zstd.DataSetMetadata` (under the
  ``…__field_data`` suffix) so :pyattr:`Reader.available_field_arrays`
  and :meth:`Reader.read` surface them without any reader change.

Crash safety
------------
The append is staged into a sibling temp file (a byte-copy of the body
prefix + new frames + new footer) and then :func:`os.replace`-d over the
original.  ``os.replace`` is atomic on POSIX and Windows, so an
interrupted append either leaves the original fully intact or completes;
it can never leave a half-written footer that destroys committed blocks.

Partial / columnar read
-----------------------
:func:`read_array` (and :class:`AppendReader`) decompress exactly the
two frames of one named block - never the rest of the file - so a single
array can be loaded back without touching any other block.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from typing import IO
from typing import TYPE_CHECKING

import numpy as np
import zstandard as zstd

from pyvista_zstd.pyvista_zstd import _FILTER_NONE
from pyvista_zstd.pyvista_zstd import _FILTER_SHUFFLE
from pyvista_zstd.pyvista_zstd import DS_METADATA_KEY
from pyvista_zstd.pyvista_zstd import FIELD_DATA_SUFFIX
from pyvista_zstd.pyvista_zstd import FILE_METADATA_KEY
from pyvista_zstd.pyvista_zstd import FILE_VERSION_SHUFFLE
from pyvista_zstd.pyvista_zstd import MULTIBLOCK_METADATA_KEY
from pyvista_zstd.pyvista_zstd import SUPPORTED_READ_SUFFIXES
from pyvista_zstd.pyvista_zstd import UID_N_CHAR
from pyvista_zstd.pyvista_zstd import ArrayInfo
from pyvista_zstd.pyvista_zstd import DataSetMetadata
from pyvista_zstd.pyvista_zstd import ZstdFileMetadata
from pyvista_zstd.pyvista_zstd import _pack_array_metadata
from pyvista_zstd.pyvista_zstd import _reconstruct_array
from pyvista_zstd.pyvista_zstd import _resolve_shuffle
from pyvista_zstd.pyvista_zstd import _shuffle_bytes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from pyvista_zstd.pyvista_zstd import ShuffleSpec

__all__ = [
    "AppendReader",
    "append_arrays",
    "read_array",
]

# Mirror the upstream Reader's sanity bound on the trailing frame count.
_MAX_FRAMES = 1_000_000


def _read_footer(f: IO[bytes]) -> tuple[int, list[tuple[int, int]]]:
    """
    Return ``(num_frames, frame_meta)`` from an open ``"rb"`` handle.

    ``frame_meta[i]`` is ``(cumulative_compressed_end, decompressed_size)``
    exactly as written by :meth:`pyvista_zstd.Writer.write`.
    """
    f.seek(-8, os.SEEK_END)
    num_frames = struct.unpack("<Q", f.read(8))[0]
    if num_frames == 0 or num_frames > _MAX_FRAMES:
        msg = "Bad number of frames. File may be corrupted."
        raise RuntimeError(msg)
    f.seek(-(8 + num_frames * UID_N_CHAR), os.SEEK_END)
    raw = f.read(num_frames * UID_N_CHAR)
    frame_meta = [struct.unpack("<QQ", raw[i * UID_N_CHAR : (i + 1) * UID_N_CHAR]) for i in range(num_frames)]
    return num_frames, frame_meta


def _frame_bounds(frame_meta: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Return per-frame ``(starts, ends)`` of compressed byte ranges."""
    ends = [end for end, _ in frame_meta]
    starts = [0, *ends[:-1]]
    return starts, ends


def _decompress_two_blobs(
    meta_blob: bytes,
    data_blob: bytes,
    decompressed_sizes: tuple[int, int],
) -> tuple[str, NDArray]:
    """Decompress one array from its two already-sliced compressed frames."""
    dctx = zstd.ZstdDecompressor()
    segments = dctx.multi_decompress_to_buffer(
        [meta_blob, data_blob],
        decompressed_sizes=struct.pack("<QQ", *decompressed_sizes),
        threads=0,
    )
    return _reconstruct_array(segments[0], segments[1])


def _read_frame_pair(f: IO[bytes], starts: list[int], ends: list[int], ai: int) -> tuple[bytes, bytes]:
    """Seek-read the two compressed frames of array index ``ai`` from ``f``."""
    mi, di = ai * 2, ai * 2 + 1
    f.seek(starts[mi])
    meta_blob = f.read(ends[mi] - starts[mi])
    f.seek(starts[di])
    data_blob = f.read(ends[di] - starts[di])
    return meta_blob, data_blob


def _load_file_meta_and_root_ds(
    f: IO[bytes],
    frame_meta: list[tuple[int, int]],
) -> tuple[ZstdFileMetadata, str, DataSetMetadata]:
    """
    Decode the file-metadata frame and the root ``__ds_metadata`` frame.

    ``f`` is an open seekable ``"rb"`` handle.  Only the two small
    metadata frame-pairs are read - never the array bodies - so this is
    cheap even on multi-gigabyte files.

    Returns ``(file_meta, root_ds_id, root_ds_meta)``.
    """
    starts, ends = _frame_bounds(frame_meta)
    n_arrays = len(frame_meta) // 2

    # File metadata is always the final array (last two frames).
    meta_blob, data_blob = _read_frame_pair(f, starts, ends, n_arrays - 1)
    _, arr = _decompress_two_blobs(meta_blob, data_blob, (frame_meta[-2][1], frame_meta[-1][1]))
    file_meta = ZstdFileMetadata.from_json(arr.tobytes().decode("utf-8"))

    # ``MULTIBLOCK_METADATA_KEY`` also ends with ``DS_METADATA_KEY``; a
    # MultiBlock file has no single root dataset to append to, so reject it
    # cleanly instead of misparsing its multiblock metadata as a dataset.
    frame_names = file_meta.frame_names
    if any(fname.endswith(MULTIBLOCK_METADATA_KEY) for fname in frame_names):
        msg = "appending to MultiBlock .pv files is not supported."
        raise NotImplementedError(msg)

    # Find the root dataset metadata frame by name.
    root_idx = None
    for i, fname in enumerate(frame_names):
        if fname.endswith(DS_METADATA_KEY):
            root_idx = i
            break
    if root_idx is None:  # pragma: no cover - malformed file
        msg = "No dataset metadata frame found; cannot append."
        raise RuntimeError(msg)

    ds_id = frame_names[root_idx][:UID_N_CHAR]
    mi, di = root_idx * 2, root_idx * 2 + 1
    meta_blob, data_blob = _read_frame_pair(f, starts, ends, root_idx)
    _, ds_arr = _decompress_two_blobs(meta_blob, data_blob, (frame_meta[mi][1], frame_meta[di][1]))
    ds_meta = DataSetMetadata.from_array(ds_arr)
    return file_meta, ds_id, ds_meta


def _compress_frames(payloads: list[bytes], *, level: int) -> list[bytes]:
    """Compress each payload into its own single-frame zstd buffer."""
    cctx = zstd.ZstdCompressor(level=level, threads=0)
    buff = cctx.multi_compress_to_buffer(payloads, threads=0)
    return [buff[i].tobytes() for i in range(len(buff))]


def append_arrays(  # noqa: C901, PLR0912, PLR0915
    filename: Path | str,
    arrays: dict[str, NDArray],
    *,
    level: int | None = None,
    shuffle: ShuffleSpec = False,
) -> None:
    """
    Append named arrays to an existing ``.pv`` file in place.

    Each array is stored as an independently-zstd-compressed
    ``field_data`` block.  Previously-written compressed frames are
    **never** re-read or re-compressed; only the small footer + the
    file/dataset metadata frames are rewritten, and the whole update is
    committed atomically via :func:`os.replace`.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to an existing ``.pv`` file (written by
        :func:`pyvista_zstd.write` or a previous :func:`append_arrays`).
    arrays : dict[str, numpy.ndarray]
        Mapping of ``name -> array``.  Names must not already exist as a
        field array in the file.  Arrays are stored verbatim (bit-exact
        round trip) regardless of dtype/shape.
    level : int, optional
        zstd compression level for the new blocks.  Defaults to the
        file's recorded ``compression_level`` so appended blocks match
        the original.
    shuffle : {"auto", True, False}, default: False
        Optionally apply the reversible byte-shuffle pre-filter to the
        appended blocks (see :func:`pyvista_zstd.write`).  Disabled by
        default; ``"auto"`` shuffles a multibyte floating-point array only
        when a quick trial compression shows it shrinks the data.  When any
        appended block is shuffled the file is promoted to format version 1;
        already-written frames are untouched and keep their own per-block
        encoding.

    Notes
    -----
    The appended blocks become visible as ``field_data`` arrays:
    :meth:`pyvista_zstd.Reader.read` returns them in ``grid.field_data``
    and :pyattr:`pyvista_zstd.Reader.available_field_arrays` lists them.
    Use :func:`read_array` / :class:`AppendReader` to read one block back
    without decompressing the rest of the file.

    """
    path = Path(filename)
    if path.suffix not in SUPPORTED_READ_SUFFIXES:
        msg = f"Filename must end in one of {SUPPORTED_READ_SUFFIXES}, not '{path.suffix}'"
        raise ValueError(msg)
    if not arrays:
        return

    with path.open("rb") as f:
        _num_frames, frame_meta = _read_footer(f)
        file_meta, ds_id, ds_meta = _load_file_meta_and_root_ds(f, frame_meta)

    level = file_meta.compression_level if level is None else int(level)

    # Validate names against the existing field arrays.
    existing_fields = set(ds_meta.field_data_keys)
    for name in arrays:
        if name in existing_fields:
            msg = (
                f"field array {name!r} already exists in {path.name}; append_arrays does not overwrite existing blocks."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # The file-metadata frame and the root dataset-metadata frame must be
    # rewritten (they grow), so we drop their *old* frames and re-emit
    # them at the tail.  Their old compressed bytes are simply not copied
    # into the new body prefix.
    #
    # frame_names ordering invariant (upstream): per array there are two
    # frames; the file-metadata array is always last.  The root
    # __ds_metadata array sits somewhere in the body.  We rebuild the
    # frame_names list with the new field arrays inserted *before* the
    # ds-metadata + file-metadata tail so the upstream reader's
    # name->index mapping stays consistent.
    # ------------------------------------------------------------------
    starts, ends = _frame_bounds(frame_meta)
    frame_names = list(file_meta.frame_names)

    # Upstream invariant: there are ``n_arrays = num_frames // 2`` arrays
    # (2 frames each), but ``frame_names`` records only the first
    # ``n_arrays - 1`` of them - the trailing file-metadata array's name
    # is deliberately *omitted* (``Writer.write`` snapshots ``frame_names``
    # before appending the file-metadata array).  The reader recovers the
    # file-metadata frame positionally (last two frames), so we MUST keep
    # this exclusion to stay byte/▷reader-compatible.
    n_arrays = len(frame_meta) // 2
    if len(frame_names) != n_arrays - 1:  # pragma: no cover - malformed file
        msg = f"frame_names length {len(frame_names)} != n_arrays-1 {n_arrays - 1}; cannot append."
        raise RuntimeError(msg)
    file_meta_array_idx = n_arrays - 1  # positional; name not in frame_names

    # Index of the root ds-metadata array (in array units).
    root_ds_array_idx = next(i for i, n in enumerate(frame_names) if n.endswith(DS_METADATA_KEY))

    # Keep all body arrays EXCEPT the root ds-meta and the file-meta,
    # whose compressed bytes we will re-emit.  Everything else is copied
    # verbatim (byte-for-byte) from the original body.
    rewritten = {root_ds_array_idx, file_meta_array_idx}
    kept_array_indices = [i for i in range(n_arrays) if i not in rewritten]

    # New body = concatenation of kept compressed frame-pairs, in order
    # (copied verbatim by offset, never decompressed), followed by new
    # field-array frame-pairs, then the regenerated ds-meta frame-pair
    # and file-meta frame-pair.  We record per-frame ``(orig_start,
    # orig_end, decompressed_size)`` for the kept frames so they can be
    # streamed straight from the source file into the temp file.
    kept_frame_specs: list[tuple[int, int, int]] = []  # (start, end, decomp)
    new_frame_names: list[str] = []
    for ai in kept_array_indices:
        mi, di = ai * 2, ai * 2 + 1
        kept_frame_specs.append((starts[mi], ends[mi], frame_meta[mi][1]))
        kept_frame_specs.append((starts[di], ends[di], frame_meta[di][1]))
        new_frame_names.append(frame_names[ai])

    # New field-data arrays.
    new_field_info: dict[str, ArrayInfo] = {}
    new_payloads: list[bytes] = []
    new_payload_decomp: list[int] = []
    new_payload_names: list[str] = []
    any_shuffled = False
    for name, raw in arrays.items():
        arr = np.ascontiguousarray(raw)
        frame_name = f"{ds_id}{name}{FIELD_DATA_SUFFIX}"
        filter_id = _FILTER_SHUFFLE if _resolve_shuffle(arr, shuffle, level) else _FILTER_NONE
        any_shuffled = any_shuffled or filter_id != _FILTER_NONE
        meta_payload = _pack_array_metadata(frame_name, arr, filter_id)
        if filter_id == _FILTER_SHUFFLE:
            data_payload = _shuffle_bytes(arr).tobytes()
        else:
            data_payload = arr.ravel().view(np.uint8).tobytes()
        new_payloads.append(meta_payload)
        new_payloads.append(data_payload)
        new_payload_decomp.append(len(meta_payload))
        new_payload_decomp.append(len(data_payload))
        new_payload_names.append(frame_name)
        new_field_info[name] = ArrayInfo(shape=tuple(int(s) for s in arr.shape), dtype=str(arr.dtype))

    # Regenerate root dataset metadata with merged field_data_keys.
    merged_field_keys = dict(ds_meta.field_data_keys)
    merged_field_keys.update(new_field_info)
    ds_meta_new = _with_field_keys(ds_meta, merged_field_keys)
    ds_meta_arr = ds_meta_new.to_array()
    ds_meta_frame_name = f"{ds_id}{DS_METADATA_KEY}"
    ds_meta_payloads = [
        _pack_array_metadata(ds_meta_frame_name, ds_meta_arr),
        ds_meta_arr.ravel().view(np.uint8).tobytes(),
    ]

    # Regenerate file metadata.  Final on-disk array order is:
    #   [kept arrays ...][new field arrays ...][ds-meta][file-meta]
    # ``frame_names`` records every array EXCEPT the trailing file-meta
    # (upstream invariant - see above), so it ends at ds-meta.
    final_frame_names = [
        *new_frame_names,
        *new_payload_names,
        ds_meta_frame_name,
    ]
    # Promote the file version if any appended block uses a byte-filter, so an
    # older reader cleanly refuses it.  Kept frames retain their own per-block
    # encoding (recorded in their metadata frames), so a previously-unfiltered
    # file gaining a shuffled block stays internally consistent.
    new_version = max(file_meta.file_version, FILE_VERSION_SHUFFLE) if any_shuffled else file_meta.file_version
    file_meta_new = ZstdFileMetadata(
        frame_names=final_frame_names,
        compression_level=file_meta.compression_level,
        compression=file_meta.compression,
        file_version=new_version,
    )
    file_meta_arr = file_meta_new.to_array()
    file_meta_payloads = [
        _pack_array_metadata(FILE_METADATA_KEY, file_meta_arr),
        file_meta_arr.ravel().view(np.uint8).tobytes(),
    ]

    # Compress the regenerated/new payloads (kept frames are reused as-is).
    to_compress = [*new_payloads, *ds_meta_payloads, *file_meta_payloads]
    to_compress_decomp = [
        *new_payload_decomp,
        len(ds_meta_payloads[0]),
        len(ds_meta_payloads[1]),
        len(file_meta_payloads[0]),
        len(file_meta_payloads[1]),
    ]
    compressed = _compress_frames(to_compress, level=level)

    # Stream-assemble the new file: kept frames copied verbatim from the
    # source by offset, then the freshly-compressed new/metadata frames,
    # then the regenerated footer.  Cumulative compressed offsets are
    # tracked as we write so the footer table stays exact.
    tmp_path = path.with_suffix(path.suffix + ".append.tmp")
    copy_chunk = 8 * 1024 * 1024
    try:
        with path.open("rb") as src, tmp_path.open("wb") as out:
            offset = 0
            footer_entries: list[tuple[int, int]] = []

            # 1) kept frames - verbatim offset copy, no decompression.
            for start, end, dsz in kept_frame_specs:
                src.seek(start)
                remaining = end - start
                while remaining > 0:
                    chunk = src.read(min(copy_chunk, remaining))
                    if not chunk:  # pragma: no cover - truncated source
                        msg = "Source .pv truncated while copying kept frames."
                        raise RuntimeError(msg)
                    out.write(chunk)
                    remaining -= len(chunk)
                offset += end - start
                footer_entries.append((offset, dsz))

            # 2) new + regenerated-metadata frames.
            for part, dsz in zip(compressed, to_compress_decomp, strict=True):
                out.write(part)
                offset += len(part)
                footer_entries.append((offset, dsz))

            # 3) footer table + trailing frame count.
            out.writelines(struct.pack("<QQ", off, dsz) for off, dsz in footer_entries)
            out.write(struct.pack("<Q", len(footer_entries)))
            out.flush()
            os.fsync(out.fileno())
        # Atomic commit: replace the original only once the staged file is
        # fully written + fsync'd.  ``Path.replace`` is atomic on POSIX and
        # Windows, so an interrupted append cannot corrupt committed blocks.
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():  # pragma: no cover - cleanup on failure
            tmp_path.unlink()


def _with_field_keys(meta: DataSetMetadata, field_keys: dict[str, ArrayInfo]) -> DataSetMetadata:
    """
    Return a copy of ``meta`` with ``field_data_keys`` replaced.

    :class:`DataSetMetadata` is a frozen slotted dataclass; rebuild it
    from its JSON image (the only public round-trip) with the field keys
    swapped in.
    """
    raw = json.loads(meta.to_json())
    raw["field_data_keys"] = {k: {"shape": list(v.shape), "dtype": v.dtype} for k, v in field_keys.items()}
    return DataSetMetadata.from_json(json.dumps(raw, separators=(",", ":")))


def read_array(filename: Path | str, name: str) -> NDArray:
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

    Returns
    -------
    numpy.ndarray
        The stored array, bit-exact.

    """
    return AppendReader(filename).read_array(name)


class AppendReader:
    """
    Partial reader for ``.pv`` field-data blocks.

    Opens a ``.pv`` file and exposes its field arrays for *individual*
    read-back, decompressing only the requested block's two frames.  The
    footer + a small set of metadata frames are read on construction so
    repeated single-block reads are cheap.

    Parameters
    ----------
    filename : pathlib.Path | str
        Path to a ``.pv`` file.

    Examples
    --------
    >>> from pyvista_zstd.append import AppendReader
    >>> r = AppendReader("data.pv")  # doctest: +SKIP
    >>> r.field_array_names  # doctest: +SKIP
    ['col_0000', 'col_0001']
    >>> col = r.read_array("col_0001")  # doctest: +SKIP

    """

    def __init__(self, filename: Path | str) -> None:
        """Open ``filename`` and read its footer + metadata frames."""
        self._path = Path(filename)
        if self._path.suffix not in SUPPORTED_READ_SUFFIXES:
            msg = f"Filename must end in one of {SUPPORTED_READ_SUFFIXES}, not '{self._path.suffix}'"
            raise ValueError(msg)
        with self._path.open("rb") as f:
            self._num_frames, self._frame_meta = _read_footer(f)
            self._file_meta, self._ds_id, self._ds_meta = _load_file_meta_and_root_ds(f, self._frame_meta)
        self._starts, self._ends = _frame_bounds(self._frame_meta)
        # name -> array index
        self._name_to_idx = {n: i for i, n in enumerate(self._file_meta.frame_names)}

    @property
    def field_array_names(self) -> list[str]:
        """Names of field-data arrays available for partial read."""
        return list(self._ds_meta.field_data_keys)

    @property
    def field_array_info(self) -> dict[str, ArrayInfo]:
        """Mapping ``name -> ArrayInfo(shape, dtype)`` for field arrays."""
        return dict(self._ds_meta.field_data_keys)

    def __contains__(self, name: str) -> bool:
        """Return whether field array ``name`` is present."""
        return name in self._ds_meta.field_data_keys

    def read_array(self, name: str) -> NDArray:
        """Decompress and return the single field array ``name``."""
        if name not in self._ds_meta.field_data_keys:
            msg = f"field array {name!r} not found; available: {sorted(self._ds_meta.field_data_keys)}"
            raise KeyError(msg)
        frame_name = f"{self._ds_id}{name}{FIELD_DATA_SUFFIX}"
        ai = self._name_to_idx.get(frame_name)
        if ai is None:  # pragma: no cover - metadata/name desync
            msg = f"frame for field array {name!r} not found in frame index."
            raise KeyError(msg)
        mi, di = ai * 2, ai * 2 + 1
        with self._path.open("rb") as f:
            meta_blob, data_blob = _read_frame_pair(f, self._starts, self._ends, ai)
        _, arr = _decompress_two_blobs(meta_blob, data_blob, (self._frame_meta[mi][1], self._frame_meta[di][1]))
        return arr

    def read_arrays(self, names: Iterable[str]) -> dict[str, NDArray]:
        """Decompress and return several field arrays by name."""
        return {n: self.read_array(n) for n in names}
