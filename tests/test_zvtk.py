"""Test `zvtk` library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from pyvista.core.dataset import DataSet
    from pyvista.core.pointset import PolyData
    from pyvista.core.pointset import UnstructuredGrid

import zvtk


def supply_data(ds: DataSet) -> None:
    """Supply point, cell, and field data to any dataset."""
    rng = np.random.default_rng()

    # point data
    n_points = ds.n_points
    n_cells = ds.n_cells
    pd = ds.point_data
    cd = ds.cell_data
    pd["int8_data"] = rng.integers(-128, 127, n_points, dtype=np.int8)
    pd["int16_data"] = rng.integers(-32768, 32767, n_points, dtype=np.int16)
    pd["int32_data"] = rng.integers(-2147483648, 2147483647, n_points, dtype=np.int32)
    pd["int64_data"] = rng.integers(
        -9223372036854775808, 9223372036854775807, n_points, dtype=np.int64
    )
    pd["uint8_data"] = rng.integers(0, 255, n_points, dtype=np.uint8)
    pd["uint16_data"] = rng.integers(0, 65535, n_points, dtype=np.uint16)
    pd["uint32_data"] = rng.integers(0, 4294967295, n_points, dtype=np.uint32)
    pd["uint64_data"] = rng.integers(0, 18446744073709551615, n_points, dtype=np.uint64)
    pd["float32_data"] = rng.random(n_points, dtype=np.float32)
    pd["float64_data"] = rng.random(n_points, dtype=np.float64)

    # cell data
    cd["int32_data"] = rng.integers(-2147483648, 2147483647, n_cells, dtype=np.int32)
    cd["float64_data"] = rng.random(n_cells, dtype=np.float64)

    # field data
    n = 10  # arbitrary size for field data
    ds.field_data["int32_field"] = np.arange(n, dtype=np.int32)
    ds.field_data["float64_field"] = rng.random(n, dtype=np.float64)
    ds.field_data["uint8_field"] = rng.integers(0, 255, n, dtype=np.uint8)

    # point data
    scalars_name = "scalars"
    ds.point_data.set_array(np.arange(ds.n_points), scalars_name)
    ds.point_data.active_scalars_name = scalars_name

    vectors_name = "vectors"
    ds.point_data.set_array(rng.random((ds.n_points, 3)), vectors_name)
    ds.point_data.active_vectors_name = vectors_name

    textures_name = "textures"
    ds.point_data.set_array(rng.random((ds.n_points, 2)), textures_name)
    ds.point_data.active_texture_coordinates_name = textures_name

    normals_name = "normals"
    ds.point_data.set_array(rng.random((ds.n_points, 3)), normals_name)
    ds.point_data.active_normals_name = normals_name

    # cell data
    scalars_name = "scalars"
    ds.cell_data.set_array(np.arange(ds.n_cells), scalars_name)
    ds.cell_data.active_scalars_name = scalars_name

    vectors_name = "vectors"
    ds.cell_data.set_array(rng.random((ds.n_cells, 3)), vectors_name)
    ds.cell_data.active_vectors_name = vectors_name

    textures_name = "textures"
    ds.cell_data.set_array(rng.random((ds.n_cells, 2)), textures_name)
    ds.cell_data.active_texture_coordinates_name = textures_name

    normals_name = "normals"
    ds.cell_data.set_array(rng.random((ds.n_cells, 3)), normals_name)
    ds.cell_data.active_normals_name = normals_name


def test_zvtk_ugrid(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test unstructured grid."""
    supply_data(ugrid)

    tmp_filename = tmp_path / "ugrid.zvtu"
    zvtk.compress(ugrid, tmp_filename)
    ugrid_out = zvtk.decompress(tmp_filename)

    assert ugrid.point_data == ugrid_out.point_data
    assert ugrid.cell_data == ugrid_out.cell_data
    assert ugrid.field_data == ugrid_out.field_data
    assert ugrid == ugrid_out


def test_zvtk_polydata(polydata: PolyData, tmp_path: Path) -> None:
    """Test unstructured grid."""
    supply_data(polydata)

    tmp_filename = tmp_path / "polydata.zvtp"
    zvtk.compress(polydata, tmp_filename)
    polydata_out = zvtk.decompress(tmp_filename)

    assert polydata.n_cells == polydata_out.n_cells
    assert polydata.n_points == polydata_out.n_points

    assert polydata.point_data == polydata_out.point_data
    assert polydata.cell_data == polydata_out.cell_data
    assert polydata.field_data == polydata_out.field_data
    assert polydata == polydata_out
