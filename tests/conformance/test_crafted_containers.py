"""
Bounds the reader keeps against a container written to break it.

Two of the checks here are 32-bit properties. ``size_t`` is 64 bits on every
machine this suite normally runs on, so arithmetic that wraps on a 32-bit
target -- a WebAssembly build is one -- cannot be caught by running the test
there. The crafted files are therefore written to be refused for a reason that
holds on every target: the value they declare is impossible against the bytes
that follow it, so a check written to be wrap-proof refuses it everywhere, and
a check written the wrapping way passes it exactly where the wrap happens.

Every case is asserted through both doors -- ``pvzstd_open`` on a path and
``pvzstd_open_memory`` on a buffer -- because they meet at one parse and a
guard that only one of them reaches is not a guard.
"""

from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING

from container_surgery import rebuild
from container_surgery import split_frames
import numpy as np
import pytest
import pyvista as pv
import zstandard as zstd

import pyvista_zstd as pz
from pyvista_zstd import _capi

if TYPE_CHECKING:
    from pathlib import Path

POISON_NDIM = 2**29
"""
A dimension count whose shape table cannot fit in any container.

Chosen so that ``ndim * 8`` is exactly 2**32: on a 32-bit target that product
wraps to zero, so ``off + ndim * 8 + PVZSTD_DTYPE_LEN`` measures the same
handful of bytes a well-formed header needs and waves the file through. The
loop that follows then reads eight bytes per dimension past the end of the
container. Written as a divide against the bytes remaining, no value of ndim
can wrap it, and half a billion dimensions is refused for the reason it should
be: the frame does not hold them.
"""

POISON_DECOMPRESSED_SIZE = 2**60 + 2**31
"""
A declared frame size no container could produce.

Both halves matter. 2**60 is absurd on any target: the index says this frame
decompresses to an exabyte, which no allocator will serve, so a reader that
relies on the allocation failing is relying on a throw. Its low 32 bits are
2**31, so a 32-bit build that narrows the declared size into ``resize`` still
asks for 2 GiB rather than truncating to something small and harmless -- which
is what turned this into an abort on WebAssembly instead of a status.
"""


def _dataset() -> pv.DataSet:
    rng = np.random.default_rng(5)
    ds = pv.Sphere(theta_resolution=10, phi_resolution=10)
    ds.point_data["scal_f64"] = rng.random(ds.n_points)
    return ds


@pytest.fixture
def container(tmp_path: Path) -> bytes:
    """Write a small single-dataset container and return its bytes."""
    path = tmp_path / "crafted_source.pv"
    pz.write(_dataset(), path, progress_bar=False)
    return path.read_bytes()


def _patch_first_header(raw: bytes, mutate) -> bytes:
    """Rewrite frame 0 -- always an array header -- through *mutate*."""
    frames = split_frames(raw)
    plain = zstd.ZstdDecompressor().decompress(frames[0][0], max_output_size=1 << 20)
    patched = mutate(bytearray(plain))
    frames[0][0] = zstd.ZstdCompressor(level=3).compress(bytes(patched))
    frames[0][1] = len(patched)
    return rebuild(frames)


def _status_from_path(raw: bytes, tmp_path: Path) -> int:
    """Open *raw* as a file and return the status, closing on success."""
    lib = _capi._load()  # noqa: SLF001
    path = tmp_path / "crafted.pv"
    path.write_bytes(raw)
    handle = ctypes.c_void_p()
    status = lib.pvzstd_open(str(path).encode("utf-8"), ctypes.byref(handle))
    if status == _capi._STATUS_OK:  # noqa: SLF001
        lib.pvzstd_close(handle)
    return int(status)


def _status_from_memory(raw: bytes) -> int:
    """Open *raw* as a buffer and return the status, closing on success."""
    lib = _capi._load()  # noqa: SLF001
    buffer = np.frombuffer(raw, dtype=np.uint8)
    handle = ctypes.c_void_p()
    status = lib.pvzstd_open_memory(
        ctypes.c_void_p(buffer.ctypes.data),
        ctypes.c_uint64(buffer.nbytes),
        ctypes.byref(handle),
    )
    if status == _capi._STATUS_OK:  # noqa: SLF001
        lib.pvzstd_close(handle)
    return int(status)


def _both_doors(raw: bytes, tmp_path: Path) -> tuple[int, int]:
    return _status_from_path(raw, tmp_path), _status_from_memory(raw)


def test_the_control_container_opens_through_both_doors(container: bytes, tmp_path: Path) -> None:
    """Rebuilt but unmodified, the container must still parse, or nothing below means anything."""
    rebuilt = rebuild(split_frames(container))
    assert _both_doors(rebuilt, tmp_path) == (_capi._STATUS_OK, _capi._STATUS_OK)  # noqa: SLF001


def test_an_absurd_ndim_is_refused_rather_than_multiplied(container: bytes, tmp_path: Path) -> None:
    """A shape table the frame cannot hold is a format error, not an out-of-bounds read."""

    def poison(header: bytearray) -> bytearray:
        (name_len,) = struct.unpack_from("<I", header, 0)
        struct.pack_into("<I", header, 4 + name_len, POISON_NDIM)
        return header

    raw = _patch_first_header(container, poison)
    expected = (_capi._STATUS_FORMAT, _capi._STATUS_FORMAT)  # noqa: SLF001
    assert _both_doors(raw, tmp_path) == expected, (
        f"a header declaring {POISON_NDIM} dimensions in a frame of a few dozen bytes was accepted"
    )


def test_an_absurd_declared_size_is_refused_before_the_allocation(container: bytes, tmp_path: Path) -> None:
    """
    A frame size the container could not have produced comes back as a status.

    The point is that it is a status at all. Leaving this to the allocator
    means leaving it to a throw, and a build with exception catching turned off
    -- Emscripten's default -- turns that throw into ``abort()`` rather than
    into ``PVZSTD_E_NOMEM``.
    """
    frames = split_frames(container)
    frames[0][1] = POISON_DECOMPRESSED_SIZE
    raw = rebuild(frames)

    expected = (_capi._STATUS_FORMAT, _capi._STATUS_FORMAT)  # noqa: SLF001
    assert _both_doors(raw, tmp_path) == expected, (
        "an index declaring a frame far larger than the container it lives in was not refused"
    )


def test_a_highly_compressible_array_still_reads(tmp_path: Path) -> None:
    """
    The ceiling on a declared frame size must not refuse a real file.

    A large array of one repeated value is the most compressible input the
    format will ever be handed: it compresses at roughly 10000:1, so its
    payload frame declares a decompressed size thousands of times the length of
    the frame that holds it. That is the case a ratio-based bound gets wrong if
    it is drawn too tight, so it is written deliberately rather than waited for.
    """
    ds = pv.ImageData(dimensions=(128, 128, 128))
    ds.point_data["constant_f64"] = np.full(ds.n_points, 3.25, dtype=np.float64)
    path = tmp_path / "compressible.pv"
    pz.write(ds, path, progress_bar=False)

    raw = path.read_bytes()
    declared = max(size for _, size in split_frames(raw))
    assert declared > 8 * len(raw), (
        f"the container is {len(raw)} bytes and its largest frame declares {declared}; "
        "this input did not compress enough to test the bound"
    )

    back = pz.read(path)
    assert np.array_equal(back.point_data["constant_f64"], ds.point_data["constant_f64"])
    assert _status_from_memory(raw) == _capi._STATUS_OK  # noqa: SLF001


THUMBNAIL_PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000002000000030806000000") + bytes(40)
"""Enough of a PNG for the writer: signature and an IHDR declaring 2 x 3 pixels."""

SKIPPABLE_PREFIX = 8
"""Skippable-frame magic and u32 length, ahead of the thumbnail record."""

DESCRIPTOR = 32
"""Bytes of thumbnail descriptor ahead of the PNG."""


@pytest.fixture
def thumbnailed(tmp_path: Path) -> bytes:
    """Write a small container with a thumbnail and return its bytes."""
    path = tmp_path / "thumbnailed_source.pv"
    pz.write(_dataset(), path, thumbnail=THUMBNAIL_PNG)
    return path.read_bytes()


def _thumbnail_every_door(raw: bytes, tmp_path: Path) -> bytes | None:
    """Return the PNG the core, path and buffer, and the zstd-free reader all report for *raw*."""
    path = tmp_path / "crafted.pv"
    path.write_bytes(raw)
    found = []
    for reader in (_capi.CoreReader(path), _capi.CoreReader(buffer=raw)):
        with reader:
            assert any(name.endswith("points") for name in reader.names())
            thumbnail = reader.thumbnail()
        found.append(None if thumbnail is None else thumbnail[0])
    found.append(pz.read_thumbnail(path))
    assert found[1:] == found[:-1], f"the readers disagree: {found}"
    return found[0]


def _with_record(raw: bytes, mutate) -> bytes:
    """Rewrite the thumbnail record frame through *mutate*, keeping the index consistent."""
    frames = split_frames(raw)
    frames[1][0] = bytes(mutate(bytearray(frames[1][0])))
    return rebuild(frames)


def test_the_control_thumbnail_reads_through_both_doors(thumbnailed: bytes, tmp_path: Path) -> None:
    """Rebuilt but unmodified, the thumbnail must still be found, or nothing below means anything."""
    rebuilt = rebuild(split_frames(thumbnailed))
    assert _thumbnail_every_door(rebuilt, tmp_path) == THUMBNAIL_PNG


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("skippable_len", 0),
        ("skippable_len", 7),
        ("skippable_len", -1),
        ("skippable_len", +1),
        ("skippable_len", 0xFFFFFFFF),
        ("header", 0),
        ("header", 31),
        ("header", "record+1"),
        ("header", 0xFFFFFFFF),
        ("png_len", -1),
        ("png_len", +1),
        ("png_len", 0xFFFFFFE0),
    ],
)
def test_an_inconsistent_thumbnail_record_is_ignored(
    thumbnailed: bytes, tmp_path: Path, field: str, value: int | str
) -> None:
    """A record whose lengths disagree with its frame is no thumbnail, not an out-of-bounds read."""
    record_len = len(THUMBNAIL_PNG) + DESCRIPTOR
    offsets = {"skippable_len": 4, "header": SKIPPABLE_PREFIX + 12, "png_len": SKIPPABLE_PREFIX + 24}
    current = {"skippable_len": record_len, "header": DESCRIPTOR, "png_len": len(THUMBNAIL_PNG)}
    if value == "record+1":
        new = record_len + 1
    elif isinstance(value, int) and abs(value) == 1:
        new = current[field] + value
    else:
        new = value

    def mutate(frame: bytearray) -> bytearray:
        struct.pack_into("<I", frame, offsets[field], new)
        return frame

    assert _thumbnail_every_door(_with_record(thumbnailed, mutate), tmp_path) is None


def test_a_record_frame_shorter_than_its_prefix_is_ignored(thumbnailed: bytes, tmp_path: Path) -> None:
    """A skippable frame cut before its length field is no thumbnail."""
    raw = _with_record(thumbnailed, lambda frame: frame[: SKIPPABLE_PREFIX - 4])
    assert _thumbnail_every_door(raw, tmp_path) is None


def test_a_non_thumbnail_pair_named_thumbnail_is_an_ordinary_array(thumbnailed: bytes, tmp_path: Path) -> None:
    """Pair 0 is the thumbnail only on an exact header; anything else keeps its data."""
    frames = split_frames(thumbnailed)
    data = np.arange(4, dtype=np.uint8)

    def as_four_bytes(header: bytearray) -> bytearray:
        (name_len,) = struct.unpack_from("<I", header, 0)
        struct.pack_into("<Q", header, 4 + name_len + 4, data.size)
        return header

    plain = bytes(as_four_bytes(bytearray(zstd.ZstdDecompressor().decompressobj().decompress(frames[0][0]))))
    frames[0] = [zstd.ZstdCompressor().compress(plain), len(plain)]
    frames[1] = [zstd.ZstdCompressor().compress(data.tobytes()), data.nbytes]
    raw = rebuild(frames)
    path = tmp_path / "crafted.pv"
    path.write_bytes(raw)
    for reader in (_capi.CoreReader(path), _capi.CoreReader(buffer=raw)):
        with reader:
            assert reader.thumbnail() is None
            assert np.array_equal(reader.read_arrays(keep={"__thumbnail"}, n_threads=0)["__thumbnail"], data)


@pytest.mark.parametrize(
    "fields",
    [
        {"skippable_len": len(THUMBNAIL_PNG) + DESCRIPTOR + 100, "png_len": len(THUMBNAIL_PNG) + 100},
        {"header": 0, "png_len": len(THUMBNAIL_PNG) + DESCRIPTOR},
        {"header": len(THUMBNAIL_PNG) + DESCRIPTOR, "png_len": 0},
    ],
    ids=["record-past-its-frame", "png-over-descriptor", "empty-png"],
)
def test_a_self_consistent_but_impossible_record_is_ignored(
    thumbnailed: bytes, tmp_path: Path, fields: dict[str, int]
) -> None:
    """Lengths that agree with each other but not with the frame or the descriptor are no thumbnail."""
    offsets = {"skippable_len": 4, "header": SKIPPABLE_PREFIX + 12, "png_len": SKIPPABLE_PREFIX + 24}

    def mutate(frame: bytearray) -> bytearray:
        for field, value in fields.items():
            struct.pack_into("<I", frame, offsets[field], value)
        return frame

    assert _thumbnail_every_door(_with_record(thumbnailed, mutate), tmp_path) is None


def test_a_payload_that_is_not_a_png_is_ignored(thumbnailed: bytes, tmp_path: Path) -> None:
    """A record whose bytes do not start with the PNG signature is no thumbnail."""

    def mutate(frame: bytearray) -> bytearray:
        frame[SKIPPABLE_PREFIX + DESCRIPTOR] ^= 0xFF
        return frame

    assert _thumbnail_every_door(_with_record(thumbnailed, mutate), tmp_path) is None


@pytest.mark.parametrize("header", [DESCRIPTOR, DESCRIPTOR + 1, DESCRIPTOR + 8])
def test_a_longer_descriptor_moves_the_png(thumbnailed: bytes, tmp_path: Path, header: int) -> None:
    """Readers find the PNG at 63 + header, so a later writer may lengthen the descriptor."""

    def mutate(frame: bytearray) -> bytearray:
        start = SKIPPABLE_PREFIX + DESCRIPTOR
        frame[start:start] = bytes(header - DESCRIPTOR)
        struct.pack_into("<I", frame, 4, header + len(THUMBNAIL_PNG))
        struct.pack_into("<I", frame, SKIPPABLE_PREFIX + 12, header)
        return frame

    raw = _with_record(thumbnailed, mutate)
    assert raw[63 + header : 63 + header + 8] == THUMBNAIL_PNG[:8]
    assert _thumbnail_every_door(raw, tmp_path) == THUMBNAIL_PNG


def test_a_compressed_thumbnail_header_is_an_ordinary_array(thumbnailed: bytes, tmp_path: Path) -> None:
    """Pair 0 is the thumbnail only on its fixed raw bytes, not on a header that decodes to the same."""
    frames = split_frames(thumbnailed)
    header = zstd.ZstdDecompressor().decompressobj().decompress(frames[0][0])
    frames[0] = [zstd.ZstdCompressor().compress(header), len(header)]
    frames[1] = [zstd.ZstdCompressor().compress(b""), 0]
    raw = rebuild(frames)
    assert _thumbnail_every_door(raw, tmp_path) is None
    with _capi.CoreReader(buffer=raw) as reader:
        assert "__thumbnail" in reader.names()
