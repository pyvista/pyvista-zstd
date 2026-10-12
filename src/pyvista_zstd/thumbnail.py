"""Optional PNG thumbnail embedded uncompressed at the start of a container."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct
from typing import TYPE_CHECKING

import pyvista as pv

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista import MultiBlock

__all__ = ["THUMBNAIL_MAX_BYTES", "ThumbnailInfo", "read_thumbnail"]

THUMBNAIL_NAME = "__thumbnail"
THUMBNAIL_MAGIC = b"PVZTHUMB"
THUMBNAIL_VERSION = 1
THUMBNAIL_SIZE = (256, 256)

_DESCRIPTOR_BYTES = 32
_RAW_FRAME_PREFIX = 12
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_RAW_FRAME_DESCRIPTOR = 0xA0
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DTYPE_FIELD = b"|u1".ljust(16)
_HEADER_BYTES = 4 + len(THUMBNAIL_NAME) + 4 + 8 + len(_DTYPE_FIELD)
_RECORD_OFFSET = 2 * _RAW_FRAME_PREFIX + _HEADER_BYTES

THUMBNAIL_MAX_BYTES = 128 * 1024 - _DESCRIPTOR_BYTES
"""Largest PNG a container can embed: the record must fit one raw zstd block."""


@dataclass(frozen=True)
class ThumbnailInfo:
    """Description of an embedded thumbnail."""

    width: int
    height: int
    stale: bool
    """Whether the container's data changed after the thumbnail was written."""


def _raw_frame_size(head: bytes, offset: int) -> int | None:
    """Return the content size of the raw-block zstd frame at *offset*, or None."""
    if head[offset : offset + 4] != _ZSTD_MAGIC or head[offset + 4] != _RAW_FRAME_DESCRIPTOR:
        return None
    (size,) = struct.unpack_from("<I", head, offset + 5)
    block = int.from_bytes(head[offset + 9 : offset + 12], "little")
    if block != 1 | (size << 3):  # last block, raw, holding the whole frame
        return None
    return size


def _array_header(record_bytes: int) -> bytes:
    """Return the array header of a thumbnail pair whose record is *record_bytes* long."""
    name = THUMBNAIL_NAME.encode()
    return struct.pack(f"<I{len(name)}sIQ", len(name), name, 1, record_bytes) + _DTYPE_FIELD


def read_thumbnail(path: Path | str) -> bytes | None:
    """
    Return the PNG thumbnail embedded in a container, or None.

    Reads only the thumbnail's bytes and decompresses nothing. Staleness is
    reported by :attr:`pyvista_zstd.Reader.thumbnail_info`.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to the container.

    Returns
    -------
    bytes | None
        The PNG file contents, or None when the container has no thumbnail.

    Examples
    --------
    >>> import pyvista as pv
    >>> import pyvista_zstd
    >>> pyvista_zstd.write(pv.Sphere(), "sphere.pv", thumbnail=True)  # doctest: +SKIP
    >>> png = pyvista_zstd.read_thumbnail("sphere.pv")  # doctest: +SKIP

    """
    with Path(path).open("rb") as f:
        head = f.read(_RECORD_OFFSET + _DESCRIPTOR_BYTES)
        if len(head) < _RECORD_OFFSET + _DESCRIPTOR_BYTES:
            return None
        record_bytes = _raw_frame_size(head, _RAW_FRAME_PREFIX + _HEADER_BYTES)
        if record_bytes is None or _raw_frame_size(head, 0) != _HEADER_BYTES:
            return None
        if head[_RAW_FRAME_PREFIX : _RAW_FRAME_PREFIX + _HEADER_BYTES] != _array_header(record_bytes):
            return None
        magic, version, header_bytes, _, _, png_nbytes, _ = struct.unpack_from("<8s6I", head, _RECORD_OFFSET)
        if magic != THUMBNAIL_MAGIC or version != THUMBNAIL_VERSION:
            return None
        if header_bytes < _DESCRIPTOR_BYTES or header_bytes + png_nbytes != record_bytes:
            return None
        f.seek(_RECORD_OFFSET + header_bytes)
        png = f.read(png_nbytes)
    return png if len(png) == png_nbytes else None


def _validated_png(png: bytes) -> bytes:
    """Return *png* as bytes after checking it is an embeddable PNG."""
    png = bytes(png)
    if not png.startswith(_PNG_SIGNATURE) or png[12:16] != b"IHDR":
        msg = "`thumbnail` must be the bytes of a PNG file"
        raise ValueError(msg)
    if len(png) > THUMBNAIL_MAX_BYTES:
        msg = f"`thumbnail` PNG is {len(png)} bytes; at most {THUMBNAIL_MAX_BYTES} can be embedded"
        raise ValueError(msg)
    return png


def _render_png(ds: DataSet | MultiBlock) -> bytes:
    """Render *ds* off-screen and return the image as PNG bytes."""
    from PIL import Image  # noqa: PLC0415

    pl = pv.Plotter(off_screen=True, window_size=THUMBNAIL_SIZE)
    try:
        pl.add_mesh(ds, show_scalar_bar=False)
        pl.render()
        img = pl.screenshot(return_img=True)
    finally:
        pl.close()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _thumbnail_png(ds: DataSet | MultiBlock, *, thumbnail: bytes | bool | None) -> bytes | None:
    """Resolve the ``thumbnail`` argument of :func:`pyvista_zstd.write` to PNG bytes."""
    if thumbnail is None or thumbnail is False:
        return None
    if thumbnail is True:
        try:
            png = _render_png(ds)
        except Exception as err:
            msg = "Could not render a thumbnail off-screen; pass PNG bytes as `thumbnail` instead"
            raise RuntimeError(msg) from err
        return _validated_png(png)
    if isinstance(thumbnail, (bytes, bytearray, memoryview)):
        return _validated_png(thumbnail)
    msg = f"`thumbnail` must be PNG bytes or True, not {type(thumbnail).__name__}"
    raise TypeError(msg)
