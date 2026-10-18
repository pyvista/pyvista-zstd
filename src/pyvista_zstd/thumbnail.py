"""Optional PNG thumbnail embedded uncompressed at the start of a container."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct
from typing import TYPE_CHECKING
from typing import BinaryIO

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyvista import DataSet
    from pyvista import MultiBlock

__all__ = ["THUMBNAIL_MAX_BYTES", "ThumbnailInfo", "read_thumbnail"]

THUMBNAIL_NAME = "__thumbnail"
THUMBNAIL_MAGIC = b"PVZTHUMB"
THUMBNAIL_VERSION = 1
THUMBNAIL_SIZE = (512, 512)
_RENDER_SCALE = 2
_RENDER_MARGIN = 0.05
# Where the sRGB transfer function switches from its linear segment to its power curve.
_SRGB_DECODE_KNEE = 0.04045
_SRGB_ENCODE_KNEE = 0.0031308

_DESCRIPTOR_BYTES = 32
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_MIN_BYTES = 33  # signature and IHDR chunk
_IHDR_CHUNK = b"\x00\x00\x00\x0dIHDR"  # length 13, then the type
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
# The trailer: an index entry of (end offset, decompressed size) per frame, then the frame count.
_COUNT_BYTES = 8
_INDEX_ENTRY_BYTES = 16
_PAIR_FRAMES = 2

THUMBNAIL_MAX_BYTES = 8 * 1024 * 1024
"""Largest PNG a container can embed, 8 MiB."""


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
        if _payload_frame_end(f) != _RECORD_OFFSET + record_bytes:
            return None
        f.seek(_RECORD_OFFSET + header_bytes)
        png = f.read(png_nbytes)
    return png if len(png) == png_nbytes and png.startswith(_PNG_SIGNATURE) else None


def _payload_frame_end(f: BinaryIO) -> int | None:
    """Return where the trailer index says frame 1, the thumbnail record, ends."""
    size = f.seek(0, io.SEEK_END)
    if size < _COUNT_BYTES:
        return None
    f.seek(size - _COUNT_BYTES)
    (n_frames,) = struct.unpack("<Q", f.read(_COUNT_BYTES))
    if n_frames < _PAIR_FRAMES or _COUNT_BYTES + _INDEX_ENTRY_BYTES * n_frames > size:
        return None
    f.seek(size - _COUNT_BYTES - _INDEX_ENTRY_BYTES * (n_frames - 1))
    (end,) = struct.unpack("<Q", f.read(8))
    return end


def _validated_png(png: bytes) -> bytes:
    """Return *png* as bytes after checking it is an embeddable PNG."""
    png = bytes(png)
    valid = len(png) >= _PNG_MIN_BYTES and png.startswith(_PNG_SIGNATURE) and png[8:16] == _IHDR_CHUNK
    if not valid or 0 in struct.unpack_from(">II", png, 16):
        msg = "`thumbnail` must be the bytes of a PNG file with a non-zero width and height"
        raise ValueError(msg)
    if len(png) > THUMBNAIL_MAX_BYTES:
        msg = f"`thumbnail` PNG is {len(png)} bytes; at most {THUMBNAIL_MAX_BYTES} can be embedded"
        raise ValueError(msg)
    return png


def _srgb_to_linear(c: NDArray[np.floating]) -> NDArray[np.floating]:
    """Decode sRGB values in [0, 1] to linear light."""
    return np.where(c <= _SRGB_DECODE_KNEE, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: NDArray[np.floating]) -> NDArray[np.floating]:
    """Encode linear-light values in [0, 1] as sRGB."""
    return np.where(c <= _SRGB_ENCODE_KNEE, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _render_png(ds: DataSet | MultiBlock) -> bytes:
    """Render *ds* off-screen on a transparent background, cropped to the content, as PNG bytes."""
    from PIL import Image  # noqa: PLC0415

    pl = pv.Plotter(off_screen=True, window_size=[_RENDER_SCALE * s for s in THUMBNAIL_SIZE])
    # Over black, multisampled edges hold colour premultiplied by coverage in linear light.
    pl.background_color = "black"
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
    content = img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] / 255
    content[..., :3] = _srgb_to_linear(content[..., :3])
    height, width = content.shape[:2]
    side = max(height, width)
    side += 2 * int(np.ceil(_RENDER_MARGIN * side))
    square = np.zeros((side, side, 4), dtype=np.float32)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = content

    # Resize premultiplied linear channels, then divide out coverage once at the final size.
    small = np.stack(
        [
            np.asarray(Image.fromarray(square[..., k]).resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS))
            for k in range(4)
        ],
        axis=-1,
    ).clip(0, 1)
    alpha = small[..., 3:]
    straight = np.divide(small[..., :3], alpha, out=np.zeros_like(small[..., :3]), where=alpha > 0)
    rgba = np.concatenate([_linear_to_srgb(straight.clip(0, 1)), alpha], axis=-1)

    buf = io.BytesIO()
    Image.fromarray(np.round(rgba * 255).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _thumbnail_png(ds: DataSet | MultiBlock, *, thumbnail: bytes | bool | None) -> bytes | None:
    """Resolve the ``thumbnail`` argument of :func:`pyvista_zstd.write` to PNG bytes."""
    if thumbnail is None or thumbnail is False:
        return None
    if thumbnail is True:
        try:
            png = _render_png(ds)
        except Exception as err:
            msg = "Could not render a thumbnail of this dataset; pass PNG bytes as `thumbnail` instead"
            raise RuntimeError(msg) from err
        return _validated_png(png)
    if isinstance(thumbnail, (bytes, bytearray, memoryview)):
        return _validated_png(thumbnail)
    msg = f"`thumbnail` must be PNG bytes or True, not {type(thumbnail).__name__}"
    raise TypeError(msg)
