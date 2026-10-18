"""Test reading metadata and single arrays without a full read."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest
import pyvista as pv
from pyvista import examples

import pyvista_zstd

if TYPE_CHECKING:
    from pyvista.core.dataset import DataSet


def _populated(ds: DataSet) -> DataSet:
    """Give *ds* point, cell and field arrays of assorted dtypes and shapes."""
    rng = np.random.default_rng(0)
    ds.point_data["float32"] = rng.random(ds.n_points, dtype=np.float32)
    ds.point_data["vectors"] = rng.random((ds.n_points, 3))
    ds.point_data["mask"] = rng.integers(0, 2, ds.n_points).astype(bool)
    ds.point_data.active_vectors_name = "vectors"
    ds.cell_data["ids"] = np.arange(ds.n_cells, dtype=np.int64)
    ds.cell_data["uint8"] = rng.integers(0, 255, ds.n_cells, dtype=np.uint8)
    ds.cell_data.active_scalars_name = "ids"
    ds.field_data["time"] = np.array([1.5])
    ds.field_data["table"] = rng.random((4, 2))
    return ds


DATASETS = {
    "PolyData": pv.Sphere,
    "UnstructuredGrid": lambda: pv.Sphere().cast_to_unstructured_grid(),
    "ImageData": lambda: pv.ImageData(dimensions=(4, 5, 6), spacing=(0.5, 1, 2), origin=(1, 2, 3)),
    "StructuredGrid": lambda: pv.ImageData(dimensions=(3, 4, 5)).cast_to_structured_grid(),
    "RectilinearGrid": lambda: pv.RectilinearGrid(np.arange(3.0), np.arange(4.0), np.arange(5.0)),
    "PointSet": lambda: pv.PointSet(np.random.default_rng(0).random((10, 3))),
    "ExplicitStructuredGrid": examples.load_explicit_structured,
}


@pytest.fixture(params=list(DATASETS))
def dataset(request: pytest.FixtureRequest) -> DataSet:
    """Return one populated dataset of each supported type."""
    return _populated(DATASETS[request.param]())


def _assert_same_array(actual: np.ndarray, expected: np.ndarray) -> None:
    """Assert two arrays hold the same values with the same dtype and shape."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def _assert_arrays_match_full_read(block: pyvista_zstd.Reader, expected: DataSet) -> None:
    """Read every array of *block* one at a time and compare with a full read."""
    for association, data in (
        ("point", expected.point_data),
        ("cell", expected.cell_data),
        ("field", expected.field_data),
    ):
        for name in data.keys():  # noqa: SIM118
            _assert_same_array(block.read_array(name, association=association), data[name])


def _assert_metadata_matches(meta: pyvista_zstd.DataSetMetadata, ds: DataSet) -> None:
    """Assert *meta* describes *ds* without having read it."""
    assert meta.ds_type == type(ds).__name__
    assert meta.n_points == ds.n_points
    assert meta.n_cells == ds.n_cells
    for keys, data in (
        (meta.point_data_keys, ds.point_data),
        (meta.cell_data_keys, ds.cell_data),
        (meta.field_data_keys, ds.field_data),
    ):
        assert keys == {
            name: pyvista_zstd.ArrayInfo(shape=data[name].shape, dtype=str(data[name].dtype))
            for name in data.keys()  # noqa: SIM118
        }
    assert meta.point_data_active_vectors_name == ds.point_data.active_vectors_name
    assert meta.cell_data_active_scalars_name == ds.cell_data.active_scalars_name


def test_dataset_metadata_and_read_array(dataset: DataSet, tmp_path: Path) -> None:
    """Metadata and single-array reads agree with a full read for every dataset type."""
    path = tmp_path / "ds.pv"
    pyvista_zstd.write(dataset, path)
    full = pyvista_zstd.read(path)

    with pyvista_zstd.Reader(path) as reader:
        meta = reader.dataset_metadata
        assert isinstance(meta, pyvista_zstd.DataSetMetadata)
        _assert_metadata_matches(meta, full)
        _assert_arrays_match_full_read(reader, full)

    if isinstance(dataset, (pv.ImageData, pv.StructuredGrid)):
        assert meta.dimensions == dataset.dimensions
    if isinstance(dataset, pv.ImageData):
        assert meta.spacing == dataset.spacing
        assert meta.origin == dataset.origin
        assert meta.offset == dataset.offset


def test_read_array_from_buffer(tmp_path: Path) -> None:
    """A reader over an in-memory container reads single arrays too."""
    ds = _populated(pv.Sphere())
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(ds, path)
    with pyvista_zstd.Reader(buffer=path.read_bytes()) as reader:
        _assert_same_array(reader.read_array("vectors", association="point"), ds.point_data["vectors"])


def test_multiblock_metadata_and_read_array(tmp_path: Path) -> None:
    """A MultiBlock's metadata tree and per-block arrays mirror a full read."""
    shared = _populated(pv.Sphere())
    mb = pv.MultiBlock(
        {
            "sphere": shared,
            "nested": pv.MultiBlock({"cube": _populated(pv.Cube()), "again": shared}),
            "empty": None,
            "image": _populated(pv.ImageData(dimensions=(3, 3, 3))),
        }
    )
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(mb, path)
    full = pyvista_zstd.read(path)

    with pyvista_zstd.Reader(path) as reader:
        meta = reader.dataset_metadata
        assert isinstance(meta, pyvista_zstd.MultiBlockMetadata)
        assert meta.children_keys == ["sphere", "nested", "empty", "image"]
        children = [meta.children_ds[uid] for uid in meta.children]
        assert children[2] is None
        nested = children[1]
        assert isinstance(nested, pyvista_zstd.MultiBlockMetadata)
        assert nested.children_keys == ["cube", "again"]
        assert nested.children[1] == children[0].uid

        for index in (0, 3):
            _assert_metadata_matches(children[index], full[index])
            assert reader[index].dataset_metadata is children[index]
            _assert_arrays_match_full_read(reader[index], full[index])
        for index in range(2):
            _assert_metadata_matches(nested.children_ds[nested.children[index]], full[1][index])
            _assert_arrays_match_full_read(reader[1][index], full[1][index])

        assert reader[2].dataset_metadata is None
        with pytest.raises(TypeError, match="arrays with reader"):
            reader.read_array("float32", association="point")
        with pytest.raises(TypeError, match="Only dataset blocks"):
            reader[1].read_array("float32", association="point")
        with pytest.raises(TypeError, match="Only dataset blocks"):
            reader[2].read_array("float32", association="point")


def test_multiblock_field_metadata_and_read_array(tmp_path: Path) -> None:
    """Root and nested MultiBlock fields use their own metadata and array frames."""
    nested = pv.MultiBlock({"sphere": _populated(pv.Sphere())})
    root = pv.MultiBlock({"nested": nested})
    root.field_data["table"] = np.arange(6, dtype=np.int16).reshape(2, 3)
    nested.field_data["table"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    path = tmp_path / "mb-fields.pv"
    pyvista_zstd.write(root, path)

    with pyvista_zstd.Reader(path) as reader:
        for block_reader, block in ((reader, root), (reader[0], nested)):
            expected = block.field_data["table"]
            info = block_reader.dataset_metadata.field_data_keys["table"]
            assert info == pyvista_zstd.ArrayInfo(expected.shape, str(expected.dtype))
            _assert_same_array(block_reader.read_array("table", association="field"), expected)
        with pytest.raises(KeyError, match="field array 'missing' not found"):
            reader[0].read_array("missing", association="field")
        with pytest.raises(TypeError, match="Only dataset blocks"):
            reader[0].read_array("table", association="cell")
    _assert_same_array(pyvista_zstd.read_array(path, "table"), root.field_data["table"])


@pytest.mark.parametrize(
    ("ds", "level", "shuffle", "file_version"),
    [
        (pv.ImageData(dimensions=(4, 4, 4)), 3, False, 0),
        (_populated(pv.ImageData(dimensions=(4, 4, 4))), 7, True, 1),
        # Fixed-width cells promote the file to version 2.
        (pv.Sphere(), 3, False, 2),
    ],
)
def test_file_metadata(tmp_path: Path, ds: DataSet, level: int, shuffle: bool, file_version: int) -> None:  # noqa: FBT001
    """File-level facts report the level and the format version written."""
    path = tmp_path / "ds.pv"
    pyvista_zstd.write(ds, path, level=level, shuffle=shuffle)
    file_meta = pyvista_zstd.Reader(path).file_metadata
    assert isinstance(file_meta, pyvista_zstd.ZstdFileMetadata)
    assert file_meta.compression == "zstandard"
    assert file_meta.compression_level == level
    assert file_meta.file_version == file_version
    # The file is its compressed frames plus a 16-byte index entry per frame and an 8-byte count.
    reader = pyvista_zstd.Reader(path)
    assert reader.compressed_nbytes + 16 * len(reader.decompressed_sizes) + 8 == path.stat().st_size


def test_read_array_after_append(tmp_path: Path) -> None:
    """An appended array is listed in the metadata and read like any field array."""
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(pv.Sphere(), path)
    extra = np.arange(12.0).reshape(3, 4)
    pyvista_zstd.append_arrays(path, {"extra": extra})
    with pyvista_zstd.Reader(path) as reader:
        assert reader.dataset_metadata.field_data_keys["extra"] == pyvista_zstd.ArrayInfo((3, 4), "float64")
        _assert_same_array(reader.read_array("extra", association="field"), extra)


def test_read_array_errors(tmp_path: Path) -> None:
    """Unknown names, wrong associations and closed readers are refused."""
    path = tmp_path / "sphere.pv"
    ds = _populated(pv.Sphere())
    pyvista_zstd.write(ds, path)
    reader = pyvista_zstd.Reader(path)
    with pytest.raises(KeyError, match="point array 'missing' not found"):
        reader.read_array("missing", association="point")
    with pytest.raises(KeyError, match="cell array 'vectors' not found"):
        reader.read_array("vectors", association="cell")
    with pytest.raises(ValueError, match="association must be one of"):
        reader.read_array("vectors", association="points")  # type: ignore[arg-type]
    reader.close()
    assert reader.dataset_metadata.n_points == ds.n_points
    with pytest.raises(ValueError, match="closed Reader"):
        reader.read_array("vectors", association="point")


def test_read_array_function_association(tmp_path: Path) -> None:
    """The module-level read_array reads point and cell arrays as well as field arrays."""
    ds = _populated(pv.Sphere())
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(ds, path)
    _assert_same_array(pyvista_zstd.read_array(path, "time"), ds.field_data["time"])
    _assert_same_array(pyvista_zstd.read_array(path, "vectors", association="point"), ds.point_data["vectors"])
    _assert_same_array(pyvista_zstd.read_array(path, "ids", association="cell"), ds.cell_data["ids"])
    with pytest.raises(KeyError, match="cell array 'vectors' not found"):
        pyvista_zstd.read_array(path, "vectors", association="cell")


def test_read_array_same_name_point_and_cell(tmp_path: Path) -> None:
    """A name shared by point and cell data reads the array of the association asked for."""
    ds = pv.Sphere()
    ds.point_data["x"] = np.arange(ds.n_points, dtype=float)
    ds.cell_data["x"] = -np.arange(ds.n_cells, dtype=float)
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(ds, path)
    with pyvista_zstd.Reader(path) as reader:
        _assert_same_array(reader.read_array("x", association="point"), ds.point_data["x"])
        _assert_same_array(reader.read_array("x", association="cell"), ds.cell_data["x"])


@pytest.mark.parametrize(
    ("ds", "points_dtype"), [(pv.Sphere(), "float32"), (pv.Sphere().points_to_double(), "float64")]
)
def test_points_dtype_is_a_dtype_name(tmp_path: Path, ds: DataSet, points_dtype: str) -> None:
    """The points dtype reads back as a NumPy dtype name."""
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(ds, path)
    assert pyvista_zstd.Reader(path).dataset_metadata.points_dtype == points_dtype


def test_points_dtype_of_older_files() -> None:
    """Files that stored the repr of the NumPy type report a dtype name too."""
    path = Path(__file__).parent / "test_data" / "sphere_file_version_0.pv"
    assert pyvista_zstd.Reader(path).dataset_metadata.points_dtype == "float32"


@pytest.mark.parametrize(("association", "error"), [("field", KeyError), ("point", TypeError), ("cell", TypeError)])
def test_read_array_function_multiblock(tmp_path: Path, association: str, error: type[Exception]) -> None:
    """A MultiBlock's child arrays must be read by indexing to the child."""
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(pv.MultiBlock({"sphere": _populated(pv.Sphere())}), path)
    message = "field array 'time' not found" if association == "field" else "A MultiBlock file"
    with pytest.raises(error, match=message):
        pyvista_zstd.read_array(path, "time", association=association)


def test_read_array_function_legacy_field_no_warning(tmp_path: Path) -> None:
    """Reading a field array of a legacy .zvtk file through read_array raises no FutureWarning."""
    ds = _populated(pv.Sphere())
    path = tmp_path / "sphere.zvtk"
    with pytest.warns(FutureWarning):
        pyvista_zstd.write(ds, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _assert_same_array(pyvista_zstd.read_array(path, "time"), ds.field_data["time"])


def test_invalid_association_checked_first(tmp_path: Path) -> None:
    """An invalid association raises ValueError even where a MultiBlock would raise TypeError."""
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(pv.MultiBlock({"sphere": pv.Sphere(), "empty": None}), path)
    with pytest.raises(ValueError, match="association must be one of"):
        pyvista_zstd.read_array(path, "Normals", association="points")  # type: ignore[arg-type]
    with pyvista_zstd.Reader(path) as reader:
        with pytest.raises(ValueError, match="association must be one of"):
            reader.read_array("Normals", association="points")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="association must be one of"):
            reader[1].read_array("Normals", association="points")  # type: ignore[arg-type]


def test_missing_array_names_its_association(tmp_path: Path) -> None:
    """A name stored under another association says which one."""
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(pv.Sphere(), path)
    with pytest.raises(KeyError, match="'Normals' is a point array"):
        pyvista_zstd.read_array(path, "Normals")
    with pytest.raises(KeyError, match="'Normals' is a point array"):
        pyvista_zstd.Reader(path).read_array("Normals", association="cell")
