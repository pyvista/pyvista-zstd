"""Tests for the byte-shuffle pre-filter (format version 1)."""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
import zstandard as zstd

import pyvista_zstd as pz
from pyvista_zstd.append import append_arrays
from pyvista_zstd.append import read_array
from pyvista_zstd.pyvista_zstd import _FILTER_NONE
from pyvista_zstd.pyvista_zstd import _FILTER_SHUFFLE
from pyvista_zstd.pyvista_zstd import FILE_VERSION
from pyvista_zstd.pyvista_zstd import FILE_VERSION_SHUFFLE
from pyvista_zstd.pyvista_zstd import FILE_VERSION_UNFILTERED
from pyvista_zstd.pyvista_zstd import _resolve_shuffle
from pyvista_zstd.pyvista_zstd import _shuffle_bytes
from pyvista_zstd.pyvista_zstd import _unshuffle_bytes

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.int16, np.complex128])
def test_shuffle_unshuffle_roundtrip(dtype: type) -> None:
    """Shuffling then unshuffling returns the original bytes for every dtype."""
    arr = (np.arange(1024, dtype=dtype) * 3).astype(dtype)
    itemsize = arr.dtype.itemsize
    shuffled = _shuffle_bytes(arr)
    recovered = _unshuffle_bytes(shuffled.tobytes(), itemsize).view(arr.dtype)
    assert np.array_equal(arr, recovered)


def test_resolve_shuffle_policy() -> None:
    """``auto`` shuffles multibyte floats only; ``True``/``False`` are explicit."""
    f64 = np.zeros(4, dtype=np.float64)
    i32 = np.zeros(4, dtype=np.int32)
    u8 = np.zeros(4, dtype=np.uint8)
    assert _resolve_shuffle(f64, "auto") is True
    assert _resolve_shuffle(i32, "auto") is False
    assert _resolve_shuffle(u8, "auto") is False
    assert _resolve_shuffle(i32, shuffle=True) is True
    assert _resolve_shuffle(u8, shuffle=True) is False  # single byte: nothing to interleave
    assert _resolve_shuffle(f64, shuffle=False) is False
    assert _FILTER_NONE != _FILTER_SHUFFLE


def _float_grid() -> pv.ImageData:
    grid = pv.ImageData(dimensions=(16, 16, 16))
    grid.point_data["disp"] = np.sin(np.arange(grid.n_points, dtype=np.float64) / 9.0)
    return grid


def test_default_is_unfiltered(tmp_path: Path) -> None:
    """The byte-shuffle filter is opt-in; the default write stays at version 0."""
    grid = _float_grid()
    path = tmp_path / "default.pv"
    pz.write(grid, path)  # no shuffle argument
    back = pz.read(path)
    assert np.array_equal(back.point_data["disp"], grid.point_data["disp"])
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001


def test_write_read_shuffle_bit_exact(tmp_path: Path) -> None:
    """An opt-in auto-shuffled file round-trips bit-exactly and is stamped version 1."""
    grid = _float_grid()
    path = tmp_path / "shuf.pv"
    pz.write(grid, path, shuffle="auto")
    back = pz.read(path)
    assert np.array_equal(back.point_data["disp"], grid.point_data["disp"])
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_SHUFFLE  # noqa: SLF001


def test_unfiltered_stays_legacy_version(tmp_path: Path) -> None:
    """``shuffle=False`` keeps the file at the backward-compatible version 0."""
    grid = _float_grid()
    path = tmp_path / "raw.pv"
    pz.write(grid, path, shuffle=False)
    back = pz.read(path)
    assert np.array_equal(back.point_data["disp"], grid.point_data["disp"])
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001


def test_auto_leaves_integer_only_files_legacy(tmp_path: Path) -> None:
    """A file with no multibyte float arrays is not promoted under ``auto``."""
    grid = pv.ImageData(dimensions=(8, 8, 8))
    grid.point_data["labels"] = np.arange(grid.n_points, dtype=np.int32)
    path = tmp_path / "ints.pv"
    pz.write(grid, path, shuffle="auto")
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    back = pz.read(path)
    assert np.array_equal(back.point_data["labels"], grid.point_data["labels"])


def test_shuffled_file_is_smaller(tmp_path: Path) -> None:
    """The opt-in shuffle filter reduces on-disk size for smooth float data."""
    grid = pv.ImageData(dimensions=(40, 40, 40))
    x = np.linspace(0, 30, grid.n_points)
    grid.point_data["disp"] = np.sin(x) * np.exp(-x / 50.0)
    shuf = tmp_path / "s.pv"
    raw = tmp_path / "r.pv"
    pz.write(grid, shuf, shuffle="auto")
    pz.write(grid, raw, shuffle=False)
    assert shuf.stat().st_size < raw.stat().st_size


def test_append_shuffle_partial_read_bit_exact(tmp_path: Path) -> None:
    """Opt-in append shuffle promotes the file and reads each column back exactly."""
    grid = _float_grid()
    path = tmp_path / "carrier.pv"
    pz.write(grid, path, shuffle=False)  # start at version 0
    cols = {f"col_{j:04d}": np.cos(np.arange(2000, dtype=np.float64) * (j + 1) / 17.0) for j in range(6)}
    append_arrays(path, cols, shuffle="auto")
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_SHUFFLE  # noqa: SLF001
    for name, arr in cols.items():
        assert np.array_equal(read_array(path, name), arr)


def test_append_shuffle_false_keeps_version(tmp_path: Path) -> None:
    """``shuffle=False`` on append leaves an unfiltered file at version 0."""
    grid = _float_grid()
    path = tmp_path / "carrier.pv"
    pz.write(grid, path, shuffle=False)
    append_arrays(path, {"extra": np.arange(500, dtype=np.float64)}, shuffle=False)
    assert pz.Reader(path)._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    assert np.array_equal(read_array(path, "extra"), np.arange(500, dtype=np.float64))


def test_auto_probe_keeps_helpful_shuffle_and_drops_harmful() -> None:
    """The ``auto`` probe shuffles data it shrinks and skips data it would inflate."""
    x = np.linspace(0, 60 * np.pi, 50000)
    helps = np.sin(x) * np.exp(-x / 200.0)
    # A regular coordinate ramp already compresses well raw; byte-shuffle breaks
    # the constant inter-element byte deltas and would enlarge it.
    hurts = np.mgrid[0:20, 0:20, 0:20].reshape(3, -1).T.astype(np.float64).reshape(-1)
    assert _resolve_shuffle(helps, "auto", level=3) is True
    assert _resolve_shuffle(hurts, "auto", level=3) is False


def test_auto_never_inflates_low_entropy_floats(tmp_path: Path) -> None:
    """``auto`` leaves a file no larger than unfiltered when shuffle would not help."""
    grid = pv.ImageData(dimensions=(20, 20, 20))
    grid.point_data["coords"] = np.mgrid[0:20, 0:20, 0:20].reshape(3, -1).T.astype(np.float64)
    auto = tmp_path / "auto.pv"
    raw = tmp_path / "raw.pv"
    pz.write(grid, auto, shuffle="auto")
    pz.write(grid, raw, shuffle=False)
    assert auto.stat().st_size <= raw.stat().st_size
    # nothing was filtered, so the file stays at the backward-compatible version
    assert pz.Reader(auto)._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    assert np.array_equal(pz.read(auto).point_data["coords"], grid.point_data["coords"])


def _frames(raw: bytes) -> list[bytes]:
    """Return every frame's decompressed body, in file order."""
    count = struct.unpack("<Q", raw[-8:])[0]
    table = raw[-8 - 16 * count : -8]
    dctx = zstd.ZstdDecompressor()
    bodies, start = [], 0
    for i in range(count):
        end, dsize = struct.unpack_from("<QQ", table, 16 * i)
        bodies.append(dctx.decompress(raw[start:end], max_output_size=dsize + 64))
        start = end
    return bodies


def _resized(header: bytes, length: int) -> bytes:
    """Return an array-metadata header with its single dimension set to *length*."""
    name_len = struct.unpack_from("<I", header)[0]
    at = 4 + name_len
    ndim = struct.unpack_from("<I", header, at)[0]
    assert ndim == 1, "the metadata blob is a flat uint8 array"
    return header[: at + 4] + struct.pack("<Q", length) + header[at + 12 :]


def _rewrite_frames(path: Path, replacements: dict[int, bytes]) -> None:
    """
    Replace frames' decompressed bytes in place, rebuilding the footer.

    The two tests below need a container this build cannot read. Building one
    by patching the writer's constants only works while the writer is the
    thing under the test's control -- it writes a *valid* file the moment the
    bytes come from somewhere else, and the assertion then passes for the
    wrong reason or fails for no reason. Corrupting the finished artefact
    instead holds the reader to account whoever produced the file.
    """
    bodies = _frames(path.read_bytes())
    for index, body in replacements.items():
        bodies[index] = body

    cctx = zstd.ZstdCompressor(level=3)
    out, table, offset = bytearray(), bytearray(), 0
    for piece in bodies:
        frame = cctx.compress(piece)
        out += frame
        offset += len(frame)
        table += struct.pack("<QQ", offset, len(piece))
    path.write_bytes(bytes(out) + bytes(table) + struct.pack("<Q", len(bodies)))


def test_unsupported_file_version_is_rejected(tmp_path: Path) -> None:
    """
    A file stamped with a newer ``file_version`` is a hard read error, not a warning.

    A future build may rely on a byte-filter this release cannot invert, so
    reading it would corrupt array values; refuse it instead.
    """
    grid = _float_grid()
    path = tmp_path / "future.pv"
    pz.write(grid, path, shuffle=True)

    # Arrays are (header, payload) frame pairs, so the file metadata is the last
    # payload and its length is declared by the header before it. Restamping the
    # version lengthens the JSON; leave the header behind and the core refuses
    # the file for a malformed trailer instead of for the version under test.
    bodies = _frames(path.read_bytes())
    meta = json.loads(bodies[-1].decode("utf-8"))
    assert meta["file_version"] == FILE_VERSION_SHUFFLE, "fixture stopped being a filtered file"
    meta["file_version"] = FILE_VERSION + 99
    blob = json.dumps(meta).encode("utf-8")
    _rewrite_frames(path, {-1: blob, -2: _resized(bodies[-2], len(blob))})

    with pytest.raises(ValueError, match="newer than the version supported"):
        pz.read(path)


def test_unknown_array_filter_id_is_rejected(tmp_path: Path) -> None:
    """
    An array tagged with an unknown filter id is a hard read error.

    Treating it as unfiltered (the old behaviour) would read the transformed
    bytes verbatim and silently corrupt the array.
    """
    grid = _float_grid()
    path = tmp_path / "badfilter.pv"
    pz.write(grid, path, shuffle=True)

    # Array metadata frames are the even-numbered ones, and a filtered array
    # carries its filter id as the last byte. Find the first that says shuffle.
    bodies = _frames(path.read_bytes())
    index = next(i for i in range(0, len(bodies), 2) if bodies[i][-1] == _FILTER_SHUFFLE)
    _rewrite_frames(path, {index: bodies[index][:-1] + bytes([99])})

    with pytest.raises(ValueError, match="Unsupported per-array filter id 99"):
        pz.read(path)
