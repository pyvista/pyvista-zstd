"""Configuration for zvtk testing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
from pyvista.core.composite import MultiBlock
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import PointSet

if TYPE_CHECKING:
    from pyvista.core.grid import ImageData
    from pyvista.core.pointset import PolyData
    from pyvista.core.pointset import UnstructuredGrid

THIS_PATH = Path(__file__).parent


@pytest.fixture
def ugrid() -> UnstructuredGrid:
    """Return an unstructured grid."""
    return pv.read(THIS_PATH / "test_data/ugrid-poly.vtk")


@pytest.fixture
def polydata() -> PolyData:
    """Return a PolyData."""
    pd = pv.Sphere()
    pd.clear_data()
    return pd


@pytest.fixture
def imagedata() -> ImageData:
    """Return ImageData."""
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


@pytest.fixture
def pointset() -> PointSet:
    """Return a PointSet."""
    rng = np.random.default_rng()
    return PointSet(rng.random((100, 3)))


@pytest.fixture
def rgrid() -> RectilinearGrid:
    """Return a RectilinearGrid."""
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 5)
    zrng = np.arange(-10, 10, 1)
    return RectilinearGrid(xrng, yrng, zrng)


@pytest.fixture
def multi_block(
    rgrid: RectilinearGrid,
    pointset: PointSet,
    ugrid: UnstructuredGrid,
    polydata: PolyData,
    imagedata: ImageData,
) -> MultiBlock:
    """Return a MultiBlock."""
    return MultiBlock([rgrid, pointset, ugrid, polydata, imagedata])


@pytest.fixture
def multi_block_nested(multi_block, ugrid) -> MultiBlock:
    """Return a nested MultiBlock."""
    return MultiBlock([ugrid, multi_block.copy(), multi_block.copy(), multi_block.copy()])
