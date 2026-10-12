"""Test reading metadata and single arrays without a full read."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
from pyvista import examples

import pyvista_zstd

if TYPE_CHECKING:
    from pathlib import Path

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
            _assert_same_array(block.read_array(name, association), data[name])


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
    # Compare an ExplicitStructuredGrid with what was written until #71 fixes its full read; then drop this.
    full = dataset if isinstance(dataset, pv.ExplicitStructuredGrid) else pyvista_zstd.read(path)

    with pyvista_zstd.Reader(path) as reader:
        meta = reader.dataset_metadata
        assert isinstance(meta, pyvista_zstd.DataSetMetadata)
        _assert_metadata_matches(meta, full)
        _assert_arrays_match_full_read(reader, full)

    if isinstance(dataset, pv.ImageData):
        assert tuple(meta.dimensions) == dataset.dimensions
        assert tuple(meta.spacing) == dataset.spacing
        assert tuple(meta.origin) == dataset.origin


def test_read_array_from_buffer(tmp_path: Path) -> None:
    """A reader over an in-memory container reads single arrays too."""
    ds = _populated(pv.Sphere())
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(ds, path)
    with pyvista_zstd.Reader(buffer=path.read_bytes()) as reader:
        _assert_same_array(reader.read_array("vectors"), ds.point_data["vectors"])


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
        with pytest.raises(TypeError, match="index the reader to a block"):
            reader.read_array("float32")
        with pytest.raises(TypeError, match="Only dataset blocks"):
            reader[1].read_array("float32")
        with pytest.raises(TypeError, match="Only dataset blocks"):
            reader[2].read_array("float32")


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


def test_read_array_after_append(tmp_path: Path) -> None:
    """An appended array is listed in the metadata and read like any field array."""
    path = tmp_path / "sphere.pv"
    pyvista_zstd.write(pv.Sphere(), path)
    extra = np.arange(12.0).reshape(3, 4)
    pyvista_zstd.append_arrays(path, {"extra": extra})
    with pyvista_zstd.Reader(path) as reader:
        assert reader.dataset_metadata.field_data_keys["extra"] == pyvista_zstd.ArrayInfo((3, 4), "float64")
        _assert_same_array(reader.read_array("extra", "field"), extra)


def test_read_array_errors(tmp_path: Path) -> None:
    """Unknown names, wrong associations and closed readers are refused."""
    path = tmp_path / "sphere.pv"
    ds = _populated(pv.Sphere())
    pyvista_zstd.write(ds, path)
    reader = pyvista_zstd.Reader(path)
    with pytest.raises(KeyError, match="point array 'missing' not found"):
        reader.read_array("missing")
    with pytest.raises(KeyError, match="cell array 'vectors' not found"):
        reader.read_array("vectors", "cell")
    with pytest.raises(ValueError, match="association must be one of"):
        reader.read_array("vectors", "points")  # type: ignore[arg-type]
    reader.close()
    assert reader.dataset_metadata.n_points == ds.n_points
    with pytest.raises(ValueError, match="closed Reader"):
        reader.read_array("vectors")


def test_array_sizes(dataset: DataSet, tmp_path: Path) -> None:
    """Each array's decompressed size matches its data, and the sizes sum to the file's."""
    path = tmp_path / "ds.pv"
    pyvista_zstd.write(dataset, path)
    with pyvista_zstd.Reader(path) as reader:
        compressed = 0
        for association, data in (
            ("point", dataset.point_data),
            ("cell", dataset.cell_data),
            ("field", dataset.field_data),
        ):
            sizes = reader.array_sizes(association)
            assert set(sizes) == set(data.keys())
            for name, size in sizes.items():
                assert size.nbytes == data[name].nbytes
                assert size.compressed_nbytes > 0
                compressed += size.compressed_nbytes
        assert compressed < reader.compressed_nbytes < path.stat().st_size


def test_array_sizes_multiblock(tmp_path: Path) -> None:
    """Block arrays are sized after indexing to the block."""
    sphere = _populated(pv.Sphere())
    path = tmp_path / "mb.pv"
    pyvista_zstd.write(pv.MultiBlock({"sphere": sphere, "empty": None}), path)
    with pyvista_zstd.Reader(path) as reader:
        assert reader[0].array_sizes("cell")["ids"].nbytes == sphere.cell_data["ids"].nbytes
        with pytest.raises(TypeError, match="index the reader to a block"):
            reader.array_sizes()
        with pytest.raises(TypeError, match="Only dataset blocks"):
            reader[1].array_sizes()
        with pytest.raises(ValueError, match="association must be one of"):
            reader[0].array_sizes("points")  # type: ignore[arg-type]


@pytest.mark.parametrize("from_buffer", [False, True])
def test_read_metadata(tmp_path: Path, from_buffer: bool) -> None:  # noqa: FBT001
    """read_metadata reports what a reader does, from a path or from bytes."""
    path = tmp_path / "mb.pv"
    level = 5
    pyvista_zstd.write(pv.MultiBlock({"sphere": _populated(pv.Sphere())}), path, level=level)
    meta = pyvista_zstd.read_metadata(path.read_bytes() if from_buffer else str(path))
    assert isinstance(meta, pyvista_zstd.ContainerMetadata)
    with pyvista_zstd.Reader(path) as reader:
        assert meta.file == reader.file_metadata
        assert meta.dataset == reader.dataset_metadata
        assert meta.nbytes == reader.nbytes
        assert meta.compressed_nbytes == reader.compressed_nbytes
    assert meta.file.compression_level == level
    assert isinstance(meta.dataset, pyvista_zstd.MultiBlockMetadata)


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
