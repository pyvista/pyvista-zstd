"""Tests for the optional embedded thumbnail."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import pytest
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd import _capi
from pyvista_zstd.thumbnail import THUMBNAIL_SIZE

if TYPE_CHECKING:
    from pathlib import Path

# Offset of the thumbnail record's version field.
VERSION_AT = 12 + 43 + 8 + 8


def _png(width: int = 24, height: int = 16) -> bytes:
    """Return a small non-uniform PNG."""
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_roundtrip(tmp_path: Path) -> None:
    """The thumbnail reads back through every entry point and leaves the dataset alone."""
    ds = pv.Sphere()
    ds["scal"] = np.arange(ds.n_points, dtype=np.float32)
    path = tmp_path / "t.pv"
    pz.write(ds, path, thumbnail=_png())

    reader = pz.Reader(path)
    assert reader.thumbnail == _png()
    assert reader.thumbnail_info == pz.ThumbnailInfo(width=24, height=16, stale=False)
    assert pz.read_thumbnail(path) == _png()
    assert pz.Reader(buffer=path.read_bytes()).thumbnail == _png()
    assert pz.read(path) == ds
    with _capi.CoreReader(path) as core:
        assert "__thumbnail" not in core.names()

    reader.close()
    assert reader.thumbnail == _png()


def test_without_thumbnail(tmp_path: Path) -> None:
    """A file written without a thumbnail reports none."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path)
    reader = pz.Reader(path)
    assert reader.thumbnail is None
    assert reader.thumbnail_info is None
    assert pz.read_thumbnail(path) is None
    assert "__thumbnail" not in reader._metadata.frame_names  # noqa: SLF001


@pytest.mark.parametrize("thumbnail", [None, False])
def test_off_writes_identical_bytes(tmp_path: Path, thumbnail: bool | None) -> None:  # noqa: FBT001
    """Turning the thumbnail off writes the same bytes as not asking."""
    pz.write(pv.Sphere(), tmp_path / "a.pv")
    pz.write(pv.Sphere(), tmp_path / "b.pv", thumbnail=thumbnail)
    assert (tmp_path / "a.pv").read_bytes() == (tmp_path / "b.pv").read_bytes()


def test_multiblock(tmp_path: Path) -> None:
    """A MultiBlock carries a thumbnail too."""
    mb = pv.MultiBlock({"a": pv.Sphere(), "b": pv.Cube()})
    path = tmp_path / "t.pv"
    pz.write(mb, path, thumbnail=_png())
    assert pz.read_thumbnail(path) == _png()
    reader = pz.Reader(path)
    assert reader.thumbnail == _png()
    assert reader[1].read() == mb["b"]
    assert "humbnail" not in reader.show_frame_compression()


def test_append_marks_stale(tmp_path: Path) -> None:
    """An append keeps the thumbnail and marks it stale."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=_png())
    pz.append_arrays(path, {"extra": np.arange(3.0)})
    reader = pz.Reader(path)
    assert reader.thumbnail == _png()
    assert reader.thumbnail_info.stale
    assert np.array_equal(pz.read_array(path, "extra"), np.arange(3.0))


def test_unknown_record_version_is_ignored(tmp_path: Path) -> None:
    """A record version this build does not read means no thumbnail."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=_png())
    raw = bytearray(path.read_bytes())
    assert raw[VERSION_AT] == 1
    raw[VERSION_AT] = 2
    path.write_bytes(bytes(raw))
    assert pz.read_thumbnail(path) is None
    reader = pz.Reader(path)
    assert reader.thumbnail is None
    assert reader.read() == pv.Sphere()


def test_read_thumbnail_checks_the_array_header(tmp_path: Path) -> None:
    """A thumbnail pair with a different array header is not read."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=_png())
    raw = bytearray(path.read_bytes())
    dtype_at = 12 + 4 + 11 + 4 + 8
    assert raw[dtype_at : dtype_at + 3] == b"|u1"
    raw[dtype_at + 1] = ord("i")
    path.write_bytes(bytes(raw))
    assert pz.read_thumbnail(path) is None


@pytest.mark.parametrize("content", [b"", b"\x28\xb5\x2f\xfd", b"x" * 200])
def test_read_thumbnail_on_other_files(tmp_path: Path, content: bytes) -> None:
    """Files that are not thumbnailed containers report none."""
    path = tmp_path / "t.pv"
    path.write_bytes(content)
    assert pz.read_thumbnail(path) is None


def test_rejects_non_png(tmp_path: Path) -> None:
    """Only PNG bytes or True are accepted."""
    with pytest.raises(ValueError, match="bytes of a PNG"):
        pz.write(pv.Sphere(), tmp_path / "t.pv", thumbnail=b"GIF89a" + bytes(40))
    with pytest.raises(TypeError, match="PNG bytes or True"):
        pz.write(pv.Sphere(), tmp_path / "t.pv", thumbnail="thumb.png")


def test_large_png(tmp_path: Path) -> None:
    """A PNG larger than one zstd block round-trips."""
    png = _png(1024, 1024)
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=png)
    assert pz.read_thumbnail(path) == png
    assert pz.Reader(path).thumbnail == png


def test_core_validates_png() -> None:
    """The core refuses non-PNG thumbnails itself."""
    with _capi.CoreWriter() as writer:
        with pytest.raises(_capi.PvzstdError):
            writer.set_thumbnail(b"not a png" * 10)
        writer.set_thumbnail(_png())


@pytest.mark.skipif(not pv.system_supports_plotting(), reason="needs a render window")
def test_render(tmp_path: Path) -> None:
    """``thumbnail=True`` renders a non-blank image."""
    path = tmp_path / "t.pv"
    pz.write(pv.Sphere(), path, thumbnail=True)
    image = np.asarray(Image.open(io.BytesIO(pz.read_thumbnail(path))))
    assert image.shape[:2] == THUMBNAIL_SIZE[::-1]
    assert image.std() > 0
    assert pz.Reader(path).thumbnail_info.width == THUMBNAIL_SIZE[0]


def test_render_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed render raises and writes nothing."""

    def fail(*_args: object, **_kwargs: object) -> None:
        """Stand in for a plotter that cannot open a render window."""
        msg = "no display"
        raise OSError(msg)

    monkeypatch.setattr(pv, "Plotter", fail)
    path = tmp_path / "t.pv"
    with pytest.raises(RuntimeError, match="Could not render a thumbnail") as excinfo:
        pz.write(pv.Sphere(), path, thumbnail=True)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert not path.exists()
