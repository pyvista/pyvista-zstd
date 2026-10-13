"""Payloads over 4 MiB are stored as several zstd frames and read back in parallel."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
import zstandard

import pyvista_zstd
from pyvista_zstd import _capi
from pyvista_zstd.pyvista_zstd import FILE_VERSION_SHUFFLE
from pyvista_zstd.pyvista_zstd import FILE_VERSION_UNFILTERED

if TYPE_CHECKING:
    from pathlib import Path

SUB_FRAME_BYTES = 4 << 20


def _payload_bytes(path: Path, name: str) -> bytes:
    """Return the compressed bytes of array *name*'s payload frame."""
    raw = path.read_bytes()
    n_frames = struct.unpack("<Q", raw[-8:])[0]
    index = raw[-(8 + n_frames * 16) : -8]
    ends = [struct.unpack("<Q", index[i * 16 : i * 16 + 8])[0] for i in range(n_frames)]
    with _capi.CoreReader(path) as reader:
        position = reader.names().index(name)
    frame = position * 2 + 1
    return raw[ends[frame - 1] : ends[frame]]


def _large_image() -> pv.ImageData:
    """Return an image whose point array is 10 MiB, so it splits into three sub-frames."""
    image = pv.ImageData(dimensions=(160, 128, 128))
    rng = np.random.default_rng(0)
    image.point_data["values"] = rng.standard_normal(image.n_points).astype(np.float32)
    return image


@pytest.mark.parametrize("shuffle", [False, True])
def test_large_payload_is_split_and_round_trips(tmp_path: Path, shuffle: bool) -> None:  # noqa: FBT001
    """A 10 MiB array becomes three frames in one index entry, keeps its version, and reads back exactly."""
    image = _large_image()
    path = tmp_path / "image.pv"
    pyvista_zstd.write(image, path, shuffle=shuffle)

    with pyvista_zstd.Reader(path) as reader:
        # The split is a layout every reader of the format accepts, so it needs no version.
        expected = FILE_VERSION_SHUFFLE if shuffle else FILE_VERSION_UNFILTERED
        assert reader._metadata.file_version == expected  # noqa: SLF001
        decompressed, _ = reader._core_reader().frame_sizes()  # noqa: SLF001
    # Still one (header, payload) pair per array: the split is inside the payload.
    assert len(decompressed) == 2 * 3

    payload = _payload_bytes(path, "0000000000000000values__point_data")
    # The first frame alone yields one sub-frame's worth, not the whole array.
    first = zstandard.ZstdDecompressor().decompress(payload)
    assert len(first) == SUB_FRAME_BYTES

    for n_threads in (0, 2, None):
        out = pyvista_zstd.read(path, n_threads=n_threads) if n_threads is not None else pyvista_zstd.read(path)
        assert np.array_equal(out.point_data["values"], image.point_data["values"])


def test_small_payloads_stay_single_frames(tmp_path: Path) -> None:
    """Nothing at or under 4 MiB is split, and such a file keeps its old version."""
    image = pv.ImageData(dimensions=(128, 128, 64))
    image.point_data["values"] = np.arange(image.n_points, dtype=np.float32)
    assert image.point_data["values"].nbytes == SUB_FRAME_BYTES
    path = tmp_path / "image.pv"
    pyvista_zstd.write(image, path)

    with pyvista_zstd.Reader(path) as reader:
        assert reader._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    payload = _payload_bytes(path, "0000000000000000values__point_data")
    assert len(zstandard.ZstdDecompressor().decompress(payload)) == SUB_FRAME_BYTES


def test_single_array_read_uses_sub_frames(tmp_path: Path) -> None:
    """The one-array entry point reads a split payload too, through the core index."""
    image = _large_image()
    path = tmp_path / "image.pv"
    pyvista_zstd.write(image, path)
    with _capi.CoreReader(path) as reader:
        index = reader.find("0000000000000000values__point_data")
        assert index is not None
        assert np.array_equal(reader.read_at(index), image.point_data["values"])


def test_split_payload_survives_append(tmp_path: Path) -> None:
    """An append keeps the split frames byte for byte and the version unchanged."""
    image = _large_image()
    path = tmp_path / "image.pv"
    pyvista_zstd.write(image, path)
    before = _payload_bytes(path, "0000000000000000values__point_data")

    pyvista_zstd.append_arrays(path, {"extra": np.arange(10.0)})
    assert _payload_bytes(path, "0000000000000000values__point_data") == before
    with pyvista_zstd.Reader(path) as reader:
        assert reader._metadata.file_version == FILE_VERSION_UNFILTERED  # noqa: SLF001
    out = pyvista_zstd.read(path)
    assert np.array_equal(out.point_data["values"], image.point_data["values"])
    assert np.array_equal(out.field_data["extra"], np.arange(10.0))


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@pytest.mark.parametrize("n_threads", [0, 2])
def test_damaged_sub_frame_header_is_refused(tmp_path: Path, n_threads: int) -> None:
    """A second sub-frame without a valid frame header is refused, whatever the thread count."""
    image = _large_image()
    path = tmp_path / "image.pv"
    pyvista_zstd.write(image, path)
    raw = bytearray(path.read_bytes())
    payload = _payload_bytes(path, "0000000000000000values__point_data")
    start = raw.find(payload)
    second = payload.find(ZSTD_MAGIC, 1)
    assert second > 0
    raw[start + second : start + second + len(ZSTD_MAGIC)] = bytes(len(ZSTD_MAGIC))
    path.write_bytes(raw)
    with pytest.raises(_capi.PvzstdError):
        pyvista_zstd.read(path, n_threads=n_threads)


def test_two_split_arrays_share_spare_workers(tmp_path: Path) -> None:
    """Two split arrays read back exactly when the workers are split between them."""
    image = _large_image()
    image.point_data["other"] = -image.point_data["values"]
    path = tmp_path / "image.pv"
    pyvista_zstd.write(image, path, shuffle=True)
    for n_threads in (3, -1):
        out = pyvista_zstd.read(path, n_threads=n_threads)
        assert np.array_equal(out.point_data["values"], image.point_data["values"])
        assert np.array_equal(out.point_data["other"], image.point_data["other"])
