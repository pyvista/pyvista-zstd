"""
Independent ``.pv`` reader, written only from the on-disk byte layout.

This module deliberately imports nothing from :mod:`pyvista_zstd`. It exists to
be the conformance oracle for the format: if it inherited the library's helpers
it would also inherit the library's assumptions, and agreement between the two
would prove nothing.

The layout it implements is specified in ``doc/format/container-v2.md``. Two
properties are easy to get wrong and are asserted here rather than assumed:

* the first field of each index entry is the frame's **end** offset, not its
  start -- misreading it pairs every payload with the wrong header while still
  decompressing cleanly;
* the per-array ``filter_id`` byte is **omitted** when no filter is in use, so
  its presence is signalled by header length, never by the file version.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

import numpy as np
import zstandard as zstd

if TYPE_CHECKING:
    from numpy.typing import NDArray

UID_N_CHAR = 16
"""Width of the dataset-UID name prefix, and of the dtype field."""

INDEX_ENTRY_SIZE = 16
"""Bytes per trailer index entry: two little-endian u64 fields."""

FILTER_NONE = 0
FILTER_SHUFFLE = 1

DS_METADATA_SUFFIX = "__ds_metadata"
FILE_METADATA_SUFFIX = "__pyvista_zstd_metadata"


class Frame(NamedTuple):
    """One zstd frame: its byte range in the file and its decompressed payload."""

    start: int
    end: int
    declared_size: int
    data: bytes


class ArrayHeader(NamedTuple):
    """The binary metadata frame that precedes each array payload."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    filter_id: int

    @property
    def bare_name(self) -> str:
        """
        The frame name with its dataset-UID prefix stripped, if it has one.

        Not every frame is UID-prefixed: the file-metadata frame is not, so
        stripping 16 characters unconditionally would mangle it.
        """
        prefix = self.name[:UID_N_CHAR]
        if len(self.name) > UID_N_CHAR and all(c in "0123456789abcdef" for c in prefix):
            return self.name[UID_N_CHAR:]
        return self.name


class Container(NamedTuple):
    """A fully parsed ``.pv`` file."""

    arrays: dict[str, NDArray[Any]]
    headers: list[ArrayHeader]
    ds_metadata: dict[str, Any] | None
    file_metadata: dict[str, Any] | None


def _unshuffle(buf: bytes, itemsize: int) -> NDArray[np.uint8]:
    """Invert the byte-plane split applied by the shuffle filter."""
    raw = np.frombuffer(buf, dtype=np.uint8)
    return raw.reshape(itemsize, -1).T.reshape(-1)


def read_frames(path: Path | str) -> list[Frame]:
    """Decompress every frame, walking the trailer index backwards from EOF."""
    raw = Path(path).read_bytes()
    (n_frames,) = struct.unpack("<Q", raw[-8:])
    index_offset = len(raw) - 8 - n_frames * INDEX_ENTRY_SIZE

    ends: list[int] = []
    sizes: list[int] = []
    for i in range(n_frames):
        end, size = struct.unpack_from("<QQ", raw, index_offset + i * INDEX_ENTRY_SIZE)
        ends.append(end)
        sizes.append(size)

    # The index stores END offsets; the first frame starts at byte 0. There is
    # no magic number and no file header.
    starts = [0, *ends[:-1]]

    dctx = zstd.ZstdDecompressor()
    frames: list[Frame] = []
    for start, end, size in zip(starts, ends, sizes, strict=True):
        data = dctx.decompress(raw[start:end], max_output_size=max(size, 1) + 4096)
        if len(data) != size:
            msg = f"frame at [{start}, {end}) yielded {len(data)} bytes, index declares {size}"
            raise ValueError(msg)
        frames.append(Frame(start, end, size, data))
    return frames


def parse_header(buf: bytes) -> ArrayHeader:
    """Parse one binary metadata frame."""
    offset = 0
    (name_len,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    name = buf[offset : offset + name_len].decode("utf-8")
    offset += name_len

    (ndim,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    shape = struct.unpack_from(f"<{ndim}Q", buf, offset)
    offset += 8 * ndim

    dtype_str = buf[offset : offset + UID_N_CHAR].strip().decode("utf-8")
    offset += UID_N_CHAR

    # Presence of the filter byte is a function of header length, not of the
    # file version: a version-2 file may or may not carry filtered arrays.
    filter_id = FILTER_NONE
    if offset < len(buf):
        (filter_id,) = struct.unpack_from("<B", buf, offset)
        offset += 1

    if offset != len(buf):
        msg = f"header for {name!r} has {len(buf) - offset} unconsumed trailing bytes"
        raise ValueError(msg)

    return ArrayHeader(name, shape, np.dtype(dtype_str), filter_id)


def _decode_payload(header: ArrayHeader, payload: bytes) -> NDArray[Any]:
    """Reverse any per-array filter and view the bytes as the declared dtype."""
    if header.filter_id == FILTER_NONE:
        return np.frombuffer(payload, dtype=header.dtype).reshape(header.shape)
    if header.filter_id == FILTER_SHUFFLE:
        unshuffled = _unshuffle(payload, header.dtype.itemsize)
        return unshuffled.view(header.dtype).reshape(header.shape)
    # Reading filtered bytes as-is would silently corrupt the array.
    msg = f"unsupported filter id {header.filter_id} for array {header.name!r}"
    raise ValueError(msg)


def read(path: Path | str) -> Container:
    """Parse a ``.pv`` file into its arrays and its two JSON metadata blobs."""
    frames = read_frames(path)
    if len(frames) % 2 != 0:
        msg = f"expected an even frame count -- frames pair as (header, payload) -- got {len(frames)}"
        raise ValueError(msg)

    arrays: dict[str, NDArray[Any]] = {}
    headers: list[ArrayHeader] = []
    ds_metadata: dict[str, Any] | None = None
    file_metadata: dict[str, Any] | None = None

    for i in range(0, len(frames), 2):
        header = parse_header(frames[i].data)
        payload = frames[i + 1].data
        headers.append(header)

        if header.name.endswith(DS_METADATA_SUFFIX):
            ds_metadata = json.loads(payload.decode("utf-8"))
        elif header.name.endswith(FILE_METADATA_SUFFIX):
            file_metadata = json.loads(payload.decode("utf-8"))
        else:
            arrays[header.name] = _decode_payload(header, payload)

    return Container(arrays, headers, ds_metadata, file_metadata)
