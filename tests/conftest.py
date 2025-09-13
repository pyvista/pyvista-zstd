"""Configuration for zvtk testing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pyvista as pv
from pyvista import examples

THIS_PATH = Path(__file__).parent


if TYPE_CHECKING:
    from pyvista.core.grid import ImageData
    from pyvista.core.pointset import PolyData
    from pyvista.core.pointset import UnstructuredGrid


@pytest.fixture
def ugrid() -> UnstructuredGrid:
    """Return an unstructured grid."""
    return pv.read(THIS_PATH / "test_data/ugrid-poly.vtk")


@pytest.fixture
def polydata() -> PolyData:
    """Return an unstructured grid."""
    pd = examples.load_airplane()
    pd.clear_data()
    return pd


@pytest.fixture
def imagedata() -> ImageData:
    """Return an unstructured grid."""
    dmat = [
        [0.70710678, 0.70710678, 0.0],
        [-0.70710678, 0.70710678, 0.0],
        [0.0, 0.0, 1.0],
    ]

    return pv.ImageData(
        dimensions=(4, 5, 6),
        spacing=(0.1, 0.2, 0.3),
        origin=(1, 2, 3),
        direction_matrix=dmat,
        offset=(0, 2, 0),
    )
