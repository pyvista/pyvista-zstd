"""Optional PNG thumbnail embedded uncompressed at the start of a container."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista import MultiBlock

__all__ = ["THUMBNAIL_MAX_BYTES", "ThumbnailInfo", "read_thumbnail"]

THUMBNAIL_NAME = "__thumbnail"
THUMBNAIL_MAGIC = b"PVZTHUMB"
THUMBNAIL_VERSION = 1
THUMBNAIL_SIZE = (512, 512)
_RENDER_SCALE = 2
_RENDER_MARGIN = 0.05

_DESCRIPTOR_BYTES = 32
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DTYPE_FIELD = b"|u1".ljust(16)
_HEADER_BYTES = 4 + len(THUMBNAIL_NAME) + 4 + 8 + len(_DTYPE_FIELD)
# Every container with a thumbnail starts with these bytes: the raw-block header frame, then the skippable magic.
_FIXED_PREFIX = (
    b"\x28\xb5\x2f\xfd\xa0"
    + struct.pack("<I", _HEADER_BYTES)
    + (1 | (_HEADER_BYTES << 3)).to_bytes(3, "little")
    + struct.pack(f"<I{len(THUMBNAIL_NAME)}sIQ", len(THUMBNAIL_NAME), THUMBNAIL_NAME.encode(), 1, 0)
    + _DTYPE_FIELD
    + b"\x50\x2a\x4d\x18"
)
_RECORD_OFFSET = len(_FIXED_PREFIX) + 4

THUMBNAIL_MAX_BYTES = 0xFFFFFFFF - _DESCRIPTOR_BYTES
"""Largest PNG a container can embed."""


@dataclass(frozen=True)
class ThumbnailInfo:
    """Description of an embedded thumbnail."""

    width: int
    height: int
    stale: bool
    """Whether the container's data changed after the thumbnail was written."""


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
        if len(head) < _RECORD_OFFSET + _DESCRIPTOR_BYTES or not head.startswith(_FIXED_PREFIX):
            return None
        (record_bytes,) = struct.unpack_from("<I", head, len(_FIXED_PREFIX))
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
    """Render *ds* off-screen on a transparent background, cropped to the content, as PNG bytes."""
    from PIL import Image  # noqa: PLC0415

    pl = pv.Plotter(off_screen=True, window_size=[_RENDER_SCALE * s for s in THUMBNAIL_SIZE])
    try:
        pl.add_mesh(ds, show_scalar_bar=False)
        pl.render()
        img = pl.screenshot(transparent_background=True, return_img=True)
    finally:
        pl.close()

    rows, cols = np.nonzero(img[..., 3])
    if rows.size == 0:
        msg = "the rendered image is empty"
        raise ValueError(msg)
    content = img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    height, width = content.shape[:2]
    side = max(height, width)
    side += 2 * int(np.ceil(_RENDER_MARGIN * side))
    square = np.zeros((side, side, 4), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = content

    buf = io.BytesIO()
    Image.fromarray(square).resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS).save(buf, format="PNG", optimize=True)
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
