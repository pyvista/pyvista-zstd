"""Tests for PyVista reader registry integration."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path as _Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import pyvista as pv

import pyvista_zstd

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytest.importorskip("pyvista.core.utilities.reader_registry")

from pyvista.core.utilities.reader_registry import _list_custom_exts as _list_reader_exts
from pyvista.core.utilities.reader_registry import _restore_registry_state as _restore_reader_state
from pyvista.core.utilities.reader_registry import _save_registry_state as _save_reader_state

_HAS_WRITER_REGISTRY = True
try:
    from pyvista.core.utilities.writer_registry import _list_custom_exts as _list_writer_exts
    from pyvista.core.utilities.writer_registry import _restore_registry_state as _restore_writer_state
    from pyvista.core.utilities.writer_registry import _save_registry_state as _save_writer_state
except ImportError:  # pragma: no cover
    _HAS_WRITER_REGISTRY = False


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Restore reader/writer registry state after each test."""
    reader_state = _save_reader_state()
    writer_state = _save_writer_state() if _HAS_WRITER_REGISTRY else None
    yield
    _restore_reader_state(reader_state)
    if _HAS_WRITER_REGISTRY:
        _restore_writer_state(writer_state)


def test_pyvista_zstd_registered() -> None:
    """pyvista-zstd registers .pv on import."""
    assert ".pv" in _list_reader_exts()


def test_pv_read_pyvista_zstd_roundtrip(tmp_path: Path) -> None:
    """pv.read() can read a .pv file when pyvista-zstd is installed."""
    mesh = pv.Sphere()
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(mesh, path)

    result = pv.read(str(path))
    assert isinstance(result, pv.PolyData)
    assert result.n_points == mesh.n_points
    assert result.n_cells == mesh.n_cells


def test_pv_read_dispatches_to_pyvista_zstd(tmp_path: Path) -> None:
    """pv.read() dispatches to pyvista-zstd's reader, not the VTK fallback."""
    mesh = pv.Sphere()
    path = tmp_path / "mesh.pv"
    pyvista_zstd.write(mesh, path)

    mock = MagicMock(return_value=pv.PolyData())
    pv.register_reader(".pv", mock, override=True)

    pv.read(str(path))
    mock.assert_called_once()


def test_entry_point_registered() -> None:
    """Entry point is configured in pyproject.toml."""
    eps = entry_points(group="pyvista.readers")
    names = [ep.name for ep in eps]
    assert ".pv" in names


@pytest.mark.skipif(not _HAS_WRITER_REGISTRY, reason="pyvista writer registry not available")
def test_pyvista_zstd_writer_registered() -> None:
    """pyvista-zstd registers a .pv writer via entry point."""
    assert ".pv" in _list_writer_exts()


@pytest.mark.skipif(not _HAS_WRITER_REGISTRY, reason="pyvista writer registry not available")
def test_mesh_save_pyvista_zstd_roundtrip(tmp_path: Path) -> None:
    """mesh.save() can write a .pv file when pyvista-zstd is installed."""
    mesh = pv.Sphere()
    path = tmp_path / "sphere.pv"
    mesh.save(str(path))

    result = pyvista_zstd.read(path)
    assert isinstance(result, pv.PolyData)
    assert result.n_points == mesh.n_points
    assert result.n_cells == mesh.n_cells


@pytest.mark.skipif(not _HAS_WRITER_REGISTRY, reason="pyvista writer registry not available")
def test_mesh_save_forwards_writer_kwargs(tmp_path: Path) -> None:
    """mesh.save() forwards writer kwargs (e.g. level) to pyvista-zstd."""
    mesh = pv.Sphere()
    path = tmp_path / "sphere.pv"
    mesh.save(str(path), level=22)

    result = pv.read(str(path))
    assert result.n_points == mesh.n_points


@pytest.mark.skipif(not _HAS_WRITER_REGISTRY, reason="pyvista writer registry not available")
def test_save_dispatches_to_pyvista_zstd(tmp_path: Path) -> None:
    """mesh.save() dispatches to pyvista-zstd's writer."""
    mesh = pv.Sphere()
    path = tmp_path / "mesh.pv"

    def fake_writer(_dataset, filename, **_kwargs) -> None:  # noqa: ANN003
        _Path(filename).write_bytes(b"")

    mock = MagicMock(side_effect=fake_writer)
    pv.register_writer(".pv", mock, override=True)

    mesh.save(str(path))
    mock.assert_called_once()


@pytest.mark.skipif(not _HAS_WRITER_REGISTRY, reason="pyvista writer registry not available")
def test_writer_entry_point_registered() -> None:
    """Writer entry point is configured in pyproject.toml."""
    eps = entry_points(group="pyvista.writers")
    names = [ep.name for ep in eps]
    assert ".pv" in names
