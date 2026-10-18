"""Hold the thumbnail pair to the byte layout in container-v2.md section 11, read without zstd."""

from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING
import zlib

import numpy as np
from PIL import Image
import pytest
import pyvista as pv
import ref_reader
import zstandard

import pyvista_zstd as pz

if TYPE_CHECKING:
    from pathlib import Path

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
SKIPPABLE_MAGIC = b"\x50\x2a\x4d\x18"
RAW_FRAME_DESCRIPTOR = 0xA0  # 4-byte content size, single segment, no checksum
PNG_OFFSET = 95


def _png(width: int = 24, height: int = 16) -> bytes:
    """Return a small non-uniform PNG."""
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _raw_frame(raw: bytes, offset: int) -> tuple[int, int]:
    """Return ``(content_start, content_size)`` of the raw-block frame at *offset*."""
    assert raw[offset : offset + 4] == ZSTD_MAGIC
    assert raw[offset + 4] == RAW_FRAME_DESCRIPTOR
    (size,) = struct.unpack_from("<I", raw, offset + 5)
    assert int.from_bytes(raw[offset + 9 : offset + 12], "little") == 1 | (size << 3)  # one raw last block
    return offset + 12, size


def _extract(raw: bytes) -> tuple[bytes, int, int, int, bool]:
    """Take ``(png, png_offset, width, height, fresh)`` out of a container with plain byte reads."""
    header_at, header_size = _raw_frame(raw, 0)
    (name_len,) = struct.unpack_from("<I", raw, header_at)
    assert raw[header_at + 4 : header_at + 4 + name_len] == b"__thumbnail"
    assert struct.unpack_from("<IQ", raw, header_at + 4 + name_len) == (1, 0)  # shape (0,)
    skippable_at = header_at + header_size
    assert raw[skippable_at : skippable_at + 4] == SKIPPABLE_MAGIC
    (record_size,) = struct.unpack_from("<I", raw, skippable_at + 4)
    record_at = skippable_at + 8
    magic, version, header_bytes, width, height, png_nbytes, crc = struct.unpack_from("<8s6I", raw, record_at)
    assert (magic, version) == (b"PVZTHUMB", 1)
    assert header_bytes + png_nbytes == record_size
    (n_frames,) = struct.unpack_from("<Q", raw, len(raw) - 8)
    index_at = len(raw) - 8 - 16 * n_frames
    fresh = zlib.crc32(raw[index_at + 32 : len(raw) - 8]) == crc
    png_at = record_at + header_bytes
    return raw[png_at : png_at + png_nbytes], png_at, width, height, fresh


@pytest.fixture(params=["polydata", "multiblock"])
def dataset(request: pytest.FixtureRequest) -> pv.DataSet | pv.MultiBlock:
    """Return a single dataset and a MultiBlock."""
    sphere = pv.Sphere()
    sphere["scal"] = np.arange(sphere.n_points, dtype=np.float32)
    if request.param == "polydata":
        return sphere
    return pv.MultiBlock({"a": sphere, "b": pv.ImageData(dimensions=(3, 4, 5))})


def test_png_read_with_plain_byte_reads(tmp_path: Path, dataset: pv.DataSet | pv.MultiBlock) -> None:
    """The PNG sits at a fixed offset, is found without zstd and decodes."""
    png = _png()
    path = tmp_path / "t.pv"
    pz.write(dataset, path, thumbnail=png)
    got, png_at, width, height, fresh = _extract(path.read_bytes())
    assert got == png
    assert png_at == PNG_OFFSET
    assert (width, height) == (24, 16)
    assert fresh
    image = Image.open(io.BytesIO(got))
    image.verify()
    assert Image.open(io.BytesIO(got)).size == (24, 16)


def test_pair_is_valid_zstd(tmp_path: Path) -> None:
    """A streaming zstd decoder reads the header frame and decodes the record frame to nothing."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=_png())
    frames = ref_reader.read_frames(path)
    raw = path.read_bytes()
    dctx = zstandard.ZstdDecompressor()
    assert dctx.decompressobj().decompress(raw[frames[0].start : frames[0].end]) == frames[0].data
    assert dctx.decompressobj().decompress(raw[frames[1].start : frames[1].end]) == b""
    assert (frames[1].declared_size, frames[1].data) == (0, b"")


def test_large_png(tmp_path: Path) -> None:
    """A PNG larger than one zstd block stays contiguous."""
    png = _png(1024, 1024)
    assert len(png) > 128 * 1024
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=png)
    assert _extract(path.read_bytes())[:2] == (png, PNG_OFFSET)


def test_older_reader_sees_an_ordinary_array(tmp_path: Path, dataset: pv.DataSet | pv.MultiBlock) -> None:
    """The reference reader, which knows nothing of thumbnails, sees one extra byte array and identical data."""
    with_thumb, without = tmp_path / "with.pv", tmp_path / "without.pv"
    pz.write(dataset, with_thumb, thumbnail=_png())
    pz.write(dataset, without)
    got, want = ref_reader.read(with_thumb), ref_reader.read(without)

    thumb = got.arrays.pop("__thumbnail")
    assert thumb.dtype == np.uint8
    assert thumb.shape == (0,)
    assert got.arrays.keys() == want.arrays.keys()
    for name, arr in want.arrays.items():
        assert np.array_equal(got.arrays[name], arr)
    assert got.ds_metadata == want.ds_metadata
    assert got.file_metadata["frame_names"] == ["__thumbnail", *want.file_metadata["frame_names"]]

    # Data frames are byte-identical.
    got_frames = [f.data for f in ref_reader.read_frames(with_thumb)]
    want_frames = [f.data for f in ref_reader.read_frames(without)]
    assert got_frames[2:-2] == want_frames[:-2]


def test_append_marks_stale(tmp_path: Path) -> None:
    """An append keeps the pair in place and breaks the staleness CRC."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=_png())
    pz.append_arrays(path, {"extra": np.arange(3.0)})
    png, png_at, _, _, fresh = _extract(path.read_bytes())
    assert (png, png_at) == (_png(), PNG_OFFSET)
    assert not fresh
