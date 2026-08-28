"""
Take a written container apart and put it back together.

Shared by the test modules that assert what the core does with a container it
was not given: both of them need the same two operations -- split the file into
its frames and their trailer entries, and re-emit the trailer over whatever the
frames became -- and a second copy of the trailer layout is a second chance to
write the tests against a format the library does not use.
"""

from __future__ import annotations

import struct

INDEX_ENTRY_SIZE = 16
"""Bytes per trailer index entry: two little-endian u64 fields."""


def split_frames(raw: bytes) -> list[list]:
    """Return ``[compressed_bytes, declared_size]`` per frame, in file order."""
    (n_frames,) = struct.unpack("<Q", raw[-8:])
    index_off = len(raw) - 8 - n_frames * INDEX_ENTRY_SIZE
    frames: list[list] = []
    start = 0
    for i in range(n_frames):
        end, size = struct.unpack_from("<QQ", raw, index_off + i * INDEX_ENTRY_SIZE)
        frames.append([raw[start:end], size])
        start = end
    return frames


def rebuild(frames: list[list]) -> bytes:
    """Concatenate the frames and re-emit the trailer index over them."""
    out = bytearray(b"".join(comp for comp, _ in frames))
    end = 0
    for comp, size in frames:
        end += len(comp)
        out += struct.pack("<QQ", end, size)
    out += struct.pack("<Q", len(frames))
    return bytes(out)
