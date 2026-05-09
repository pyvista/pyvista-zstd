"""Configuration for pyvista-zstd testing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
from pyvista import examples
from pyvista.core.composite import MultiBlock
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import ExplicitStructuredGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import StructuredGrid

if TYPE_CHECKING:
    from pyvista.core.grid import ImageData
    from pyvista.core.pointset import PolyData
    from pyvista.core.pointset import UnstructuredGrid


THIS_PATH = Path(__file__).parent


VTK_HAS_POLYHEDRA_API = pv.vtk_version_info >= (9, 4)

requires_polyhedra_api = pytest.mark.skipif(
    not VTK_HAS_POLYHEDRA_API,
    reason="Polyhedron round-trip requires VTK >= 9.4",
)


@pytest.fixture
def ugrid() -> UnstructuredGrid:
    """Return a polyhedron-free unstructured grid (works on all VTK versions)."""
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
    cell_type = np.array([pv.CellType.HEXAHEDRON, pv.CellType.HEXAHEDRON], dtype=np.uint8)
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
            [0, 0, 3],
            [1, 0, 3],
            [1, 1, 3],
            [0, 1, 3],
        ],
        dtype=np.float32,
    )
    return pv.UnstructuredGrid(cells, cell_type, points)


@pytest.fixture
def ugrid_polyhedra() -> UnstructuredGrid:
    """Return an unstructured grid that contains polyhedra (VTK>=9.4 only)."""
    if not VTK_HAS_POLYHEDRA_API:
        pytest.skip("Polyhedron round-trip requires VTK >= 9.4")
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
def sgrid() -> StructuredGrid:
    """Return a StructuredGrid."""
    xrng = np.linspace(-10, 10)
    yrng = np.linspace(-10, 10, 20)
    x, y = np.meshgrid(xrng, yrng, indexing="ij")
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    return StructuredGrid(x, y, z)


@pytest.fixture
def esgrid() -> ExplicitStructuredGrid:
    """Return an ExplicitStructuredGrid."""
    return examples.load_explicit_structured()


@pytest.fixture
def multi_block(
    rgrid: RectilinearGrid,
    pointset: PointSet,
    ugrid: UnstructuredGrid,
    polydata: PolyData,
    imagedata: ImageData,
) -> MultiBlock:
    """Return a MultiBlock (polyhedron-free, works on all VTK versions)."""
    mblock = MultiBlock()
    mblock["RectilinearGrid"] = rgrid
    mblock["PointSet"] = pointset
    mblock["UnstructuredGrid"] = ugrid
    mblock["PolyData"] = polydata
    mblock["ImageData"] = imagedata
    return mblock


@pytest.fixture
def multi_block_nested(multi_block, ugrid) -> MultiBlock:
    """Return a nested MultiBlock (polyhedron-free)."""
    return MultiBlock([ugrid, multi_block.copy(), multi_block.copy(), multi_block.copy()])
