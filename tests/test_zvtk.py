"""Test `zvtk` library."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest
from pyvista.core.composite import MultiBlock
from pyvista.core.grid import ImageData
from pyvista.core.grid import RectilinearGrid
from pyvista.core.pointset import PointSet
from pyvista.core.pointset import PolyData
from pyvista.core.pointset import UnstructuredGrid

import zvtk

if TYPE_CHECKING:
    from pathlib import Path

    from pyvista.core.dataset import DataSet


def populate_data(ds: DataSet) -> None:
    """Supply point, cell, and field data to any dataset."""
    rng = np.random.default_rng()

    # point data
    n_points = ds.n_points

    pd = ds.point_data
    pd["int8_data"] = rng.integers(-128, 127, n_points, dtype=np.int8)
    pd["int16_data"] = rng.integers(-32768, 32767, n_points, dtype=np.int16)
    pd["int32_data"] = rng.integers(-2147483648, 2147483647, n_points, dtype=np.int32)
    pd["int64_data"] = rng.integers(-9223372036854775808, 9223372036854775807, n_points, dtype=np.int64)
    pd["uint8_data"] = rng.integers(0, 255, n_points, dtype=np.uint8)
    pd["uint16_data"] = rng.integers(0, 65535, n_points, dtype=np.uint16)
    pd["uint32_data"] = rng.integers(0, 4294967295, n_points, dtype=np.uint32)
    pd["uint64_data"] = rng.integers(0, 18446744073709551615, n_points, dtype=np.uint64)
    pd["float32_data"] = rng.random(n_points, dtype=np.float32)
    pd["float64_data"] = rng.random(n_points, dtype=np.float64)

    # cell data
    n_cells = ds.n_cells
    if n_cells:
        cd = ds.cell_data
        cd["int32_data"] = rng.integers(-2147483648, 2147483647, n_cells, dtype=np.int32)
        cd["float64_data"] = rng.random(n_cells, dtype=np.float64)

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
    if n_cells:
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

    # field data
    n = 10  # arbitrary size for field data
    ds.field_data["int32_field"] = np.arange(n, dtype=np.int32)
    ds.field_data["float64_field"] = rng.random(n, dtype=np.float64)
    ds.field_data["uint8_field"] = rng.integers(0, 255, n, dtype=np.uint8)


def test_ugrid(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test unstructured grid."""
    populate_data(ugrid)

    tmp_filename = tmp_path / "ugrid.zvtk"
    zvtk.write(ugrid, tmp_filename)
    ugrid_out = zvtk.read(tmp_filename)

    assert ugrid.point_data == ugrid_out.point_data
    assert ugrid.cell_data == ugrid_out.cell_data
    assert ugrid.field_data == ugrid_out.field_data

    # can remove with pyvista >= 0.47.0
    np.allclose(ugrid.polyhedron_faces, ugrid_out.polyhedron_faces)
    np.allclose(ugrid.polyhedron_face_locations, ugrid_out.polyhedron_face_locations)

    assert ugrid == ugrid_out


@pytest.mark.parametrize("strip", [False, True])
def test_polydata(polydata: PolyData, tmp_path: Path, *, strip: bool) -> None:
    """Test unstructured grid."""
    # add in separate lines and vertices
    polydata.lines = [2, 0, 1, 2, 1, 2, 2, 8, 9]
    polydata.verts = [1, 0, 1, 1, 1, 2, 1, 3, 1, 4]
    if strip:
        polydata = polydata.strip()

    populate_data(polydata)

    tmp_filename = tmp_path / "polydata.zvtk"
    zvtk.write(polydata, tmp_filename)
    polydata_out = zvtk.read(tmp_filename)

    assert polydata.n_cells == polydata_out.n_cells
    assert polydata.n_strips == polydata_out.n_strips
    assert polydata.n_points == polydata_out.n_points

    assert polydata.point_data == polydata_out.point_data
    assert polydata.cell_data == polydata_out.cell_data
    assert polydata.field_data == polydata_out.field_data
    assert polydata == polydata_out


def test_imagedata(imagedata: ImageData, tmp_path: Path) -> None:
    """Test unstructured grid."""
    populate_data(imagedata)

    tmp_filename = tmp_path / "imagedata.zvtk"
    zvtk.write(imagedata, tmp_filename)
    imagedata_out = zvtk.read(tmp_filename)

    assert imagedata.n_cells == imagedata_out.n_cells
    assert imagedata.n_points == imagedata_out.n_points

    assert imagedata.dimensions == imagedata_out.dimensions
    assert imagedata.spacing == imagedata_out.spacing
    assert imagedata.origin == imagedata_out.origin
    assert np.allclose(imagedata.direction_matrix, imagedata_out.direction_matrix)
    assert imagedata.offset == imagedata_out.offset

    assert imagedata.point_data == imagedata_out.point_data
    assert imagedata.cell_data == imagedata_out.cell_data
    assert imagedata.field_data == imagedata_out.field_data
    assert imagedata == imagedata_out


def test_pointset(pointset: PointSet, tmp_path: Path) -> None:
    """Test compressing a pointset."""
    populate_data(pointset)

    tmp_filename = tmp_path / "pointset.zvtk"
    zvtk.write(pointset, tmp_filename)
    pointset_out = zvtk.read(tmp_filename)

    assert pointset.n_points == pointset_out.n_points

    assert pointset.point_data == pointset_out.point_data
    assert pointset.field_data == pointset_out.field_data
    assert pointset == pointset_out


def test_rectilineargrid(rgrid: RectilinearGrid, tmp_path: Path) -> None:
    """Test compressing a RectilinearGrid."""
    populate_data(rgrid)

    tmp_filename = tmp_path / "rgrid.zvtk"
    zvtk.write(rgrid, tmp_filename)
    rgrid_out = zvtk.read(tmp_filename)

    assert rgrid.dimensions == rgrid_out.dimensions
    assert rgrid.n_points == rgrid_out.n_points
    assert rgrid.n_cells == rgrid_out.n_cells

    assert np.array_equal(rgrid.x, rgrid_out.x)
    assert np.array_equal(rgrid.y, rgrid_out.y)
    assert np.array_equal(rgrid.z, rgrid_out.z)

    assert rgrid.point_data == rgrid_out.point_data
    assert rgrid.cell_data == rgrid_out.cell_data
    assert rgrid.field_data == rgrid_out.field_data
    assert rgrid == rgrid_out


def test_reader_array_selection(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test Reader class with array sub-selection."""
    populate_data(ugrid)

    tmp_filename = tmp_path / "ugrid.zvtk"
    zvtk.write(ugrid, tmp_filename)

    reader = zvtk.Reader(tmp_filename)

    # by default, all arrays available
    assert reader.available_point_arrays
    assert reader.available_cell_arrays
    assert reader.available_field_arrays

    # restrict to one array from each category
    point_choice = next(iter(reader.available_point_arrays))
    cell_choice = next(iter(reader.available_cell_arrays))
    field_choice = next(iter(reader.available_field_arrays))

    reader.selected_point_arrays = {point_choice}
    assert reader.selected_point_arrays == {point_choice}
    reader.selected_cell_arrays = {cell_choice}
    assert reader.selected_cell_arrays == {cell_choice}
    reader.selected_field_arrays = {field_choice}
    assert reader.selected_field_arrays == {field_choice}

    out = reader.read()

    # only selected arrays must be present
    assert set(out.point_data.keys()) == {point_choice}
    assert set(out.cell_data.keys()) == {cell_choice}
    assert set(out.field_data.keys()) == {field_choice}

    # test with no arrays

    reader.selected_point_arrays = set()
    reader.selected_cell_arrays = set()
    reader.selected_field_arrays = set()

    out = reader.read()

    # only selected arrays must be present
    assert not out.point_data
    assert not out.cell_data
    assert not out.field_data

    # test wrong arrays
    with pytest.raises(ValueError, match=r"point array\(s\) are not available"):
        reader.selected_point_arrays = {"this is not a point array name"}
    with pytest.raises(ValueError, match=r"cell array\(s\) are not available"):
        reader.selected_cell_arrays = {"this is not a cell array name"}
    with pytest.raises(ValueError, match=r"field array\(s\) are not available"):
        reader.selected_field_arrays = {"this is not a field array name"}


@pytest.mark.parametrize(
    "ds_type",
    [
        ImageData,
        RectilinearGrid,
        PointSet,
        PolyData,
        UnstructuredGrid,
    ],
)
def test_empty_objects(ds_type: str, tmp_path: Path) -> None:
    """Test reading/writing empty objects."""
    filename = tmp_path / f"{ds_type}.zvtk"
    ds = ds_type()
    zvtk.write(ds, filename)

    reader = zvtk.Reader(filename)
    out = reader.read()

    assert isinstance(out, type(ds))
    assert out.n_points == 0
    assert out.n_cells == 0
    assert not out.point_data
    assert not out.cell_data
    assert not out.field_data


def test_invalid_filename(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test reading and writing with the wrong extension raises an error."""
    with pytest.raises(ValueError, match="Filename must end in"):
        zvtk.write(ugrid, tmp_path / "tmp.vtk")

    # ensure we can't read in the wrong file type
    vtk_file = tmp_path / "ugrid.vtk"
    ugrid.save(vtk_file)
    with pytest.raises(ValueError, match="Filename must end in"):
        zvtk.read(vtk_file)

    # even if it's corrupted
    renamed_file = vtk_file.with_suffix(".zvtk")
    vtk_file.rename(renamed_file)
    with pytest.raises(RuntimeError, match="File may be corrupted"):
        zvtk.read(renamed_file)

    empty_file = tmp_path / "empty.zvtk"
    with empty_file.open("wb") as fid:
        fid.write(b"\x00" * 1024)  # write 1 KB of zeros

    with pytest.raises(RuntimeError, match="File may be corrupted"):
        zvtk.read(empty_file)


def test_use_int64(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test Reader class with array sub-selection."""
    populate_data(ugrid)

    tmp_filename = tmp_path / "ugrid.zvtk"

    zvtk.write(ugrid, tmp_filename, force_int32=False)
    ugrid_out = zvtk.read(tmp_filename)
    assert ugrid_out.cell_connectivity.dtype == np.int64

    zvtk.write(ugrid, tmp_filename, force_int32=True)
    ugrid_out = zvtk.read(tmp_filename)
    assert ugrid_out.cell_connectivity.dtype == np.int32


def test_future_version_warning(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Check that a warning is issued when reading a future version."""
    filename = tmp_path / "future.zvtk"

    orig_version = zvtk.FILE_VERSION
    zvtk.write(ugrid, filename)

    zvtk.zvtk.FILE_VERSION = -1
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zvtk.read(filename)
            assert any("newer than the version" in str(wi.message) for wi in w)
    finally:
        zvtk.zvtk.FILE_VERSION = orig_version


def test_reader_repr(ugrid: UnstructuredGrid, tmp_path: Path) -> None:
    """Test Reader class with array sub-selection."""
    populate_data(ugrid)

    tmp_filename = tmp_path / "ugrid.zvtk"
    zvtk.write(ugrid, tmp_filename)
    reader = zvtk.Reader(tmp_filename)
    repr_str = repr(reader)

    # check key metadata is mentioned
    assert str(tmp_filename) in repr_str
    assert type(ugrid).__name__ in repr_str
    assert str(ugrid.n_points) in repr_str
    assert str(ugrid.n_cells) in repr_str

    # check that some of the point/cell/field arrays are listed
    for arr_name in ugrid.point_data:
        assert arr_name in repr_str
    for arr_name in ugrid.cell_data:
        assert arr_name in repr_str
    for arr_name in ugrid.field_data:
        assert arr_name in repr_str

    ugrid.clear_data()
    tmp_filename = tmp_path / "ugrid.zvtk"
    zvtk.write(ugrid, tmp_filename)
    reader = zvtk.Reader(tmp_filename)
    repr_no_data_str = repr(reader)

    assert "Point arrays" not in repr_no_data_str
    assert "Cell arrays" not in repr_no_data_str
    assert "Field arrays" not in repr_no_data_str


def test_compression_level(tmp_path: Path) -> None:
    """Test setting compression level changes resulting file size."""
    tmp_filename = tmp_path / "tmp.zvtk"
    ds = ImageData(dimensions=(30, 30, 30)).cast_to_unstructured_grid()

    cell_bytes = ds.celltypes.nbytes + ds.cell_connectivity.nbytes + ds.offset.nbytes
    nbytes_orig = int(cell_bytes // 2) + ds.points.nbytes

    sizes = []
    for level in [-10, 10, 20]:
        zvtk.write(ds, tmp_filename, level=level)
        sizes.append(tmp_filename.stat().st_size)

    assert np.array_equal(np.sort(sizes)[::-1], sizes)
    assert (np.array(sizes) < nbytes_orig).all()


def test_multiblock(multi_block: MultiBlock, tmp_path: Path) -> None:
    """Test setting compression level changes resulting file size."""
    tmp_filename = tmp_path / "tmp.zvtk"
    for ds in multi_block:
        populate_data(ds)

    zvtk.write(multi_block, tmp_filename)
    multi_block_out = zvtk.read(tmp_filename)

    assert multi_block == multi_block_out


def test_multiblock_reader_class(multi_block: MultiBlock, tmp_path: Path) -> None:
    """Test setting compression level changes resulting file size."""
    tmp_filename = tmp_path / "tmp.zvtk"
    for ds in multi_block:
        populate_data(ds)

    zvtk.write(multi_block, tmp_filename)
    reader = zvtk.Reader(tmp_filename)
    assert "MultiBlock" in repr(reader)

    # test selective reading
    for ii in range(len(multi_block)):
        assert reader[ii].read() == multi_block[ii]

    assert reader.read() == multi_block


def test_multiblock_nested(multi_block_nested: MultiBlock, tmp_path: Path) -> None:
    """Test reading a nested MultiBlock hierarchy."""
    tmp_filename = tmp_path / "tmp_nested.zvtk"
    for ds in multi_block_nested:
        if isinstance(ds, MultiBlock):
            for sub_ds in ds:
                populate_data(sub_ds)
        else:
            populate_data(ds)

    zvtk.write(multi_block_nested, tmp_filename)
    reader = zvtk.Reader(tmp_filename)

    # Check top-level MultiBlock hierarchy
    assert "MultiBlock" in repr(reader)
    raise RuntimeError(repr(reader))  # temp

    # Test selective reading of top-level blocks
    for ii in range(len(reader._ds_reader)):
        top_block = reader[ii].read()
        expected_block = multi_block_nested[ii]
        assert top_block == expected_block

    # Test recursive reading of nested blocks
    for ii, child_reader in enumerate(reader._ds_reader._children):
        if isinstance(multi_block_nested[ii], MultiBlock):
            nested_reader = child_reader
            for jj in range(len(nested_reader)):
                nested_ds = nested_reader[jj].read()
                expected_ds = multi_block_nested[ii][jj]
                assert nested_ds == expected_ds

    # Read entire hierarchy
    full_ds = reader.read()
    assert full_ds == multi_block_nested
