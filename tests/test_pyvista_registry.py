"""Tests for PyVista reader registry integration."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import pyvista as pv

import pyvista_zstd

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytest.importorskip("pyvista.core.utilities.reader_registry")

from pyvista.core.utilities.reader_registry import _custom_ext_readers
from pyvista.core.utilities.reader_registry import _restore_registry_state
from pyvista.core.utilities.reader_registry import _save_registry_state


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Restore reader registry state after each test."""
    state = _save_registry_state()
    yield
    _restore_registry_state(state)


def test_pyvista_zstd_registered() -> None:
    """pyvista-zstd registers .pv on import."""
    assert ".pv" in _custom_ext_readers


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
