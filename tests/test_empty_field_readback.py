"""Empty tuple fields remain visible to dataset and exact array readers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd as pvzstd
from pyvista_zstd.append import append_arrays
from pyvista_zstd.append import read_array

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("shape", [(0, 1), (0, 17), (0, 2, 3)])
@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint8, np.bool_, np.complex128])
def test_empty_tuple_fields_read_through_dataset_and_array_apis(
    tmp_path: Path, shape: tuple[int, ...], dtype: type
) -> None:
    """An empty field must not prevent reading the rest of its dataset."""
    path = tmp_path / "empty-field.pv"
    ds = pv.UnstructuredGrid()
    ds.field_data["kept_ids"] = np.array([2**53 + 1, -7], dtype=np.int64)
    pvzstd.write(ds, path)
    empty = np.empty(shape, dtype=dtype)
    append_arrays(path, {"empty": empty})
    original = path.read_bytes()
    raw = read_array(path, "empty")
    assert raw.dtype == empty.dtype
    assert raw.shape == empty.shape
    assert raw.tobytes() == empty.tobytes()
    restored = pvzstd.read(path)
    assert set(restored.field_data) == {"kept_ids", "empty"}
    np.testing.assert_array_equal(restored.field_data["kept_ids"], ds.field_data["kept_ids"])
    actual = np.asarray(restored.field_data["empty"])
    assert actual.dtype == empty.dtype
    assert actual.shape == ((0,) if np.prod(shape[1:]) == 1 else (0, int(np.prod(shape[1:]))))
    assert actual.tobytes() == empty.tobytes()
    assert path.read_bytes() == original


@pytest.mark.parametrize("components", [1, 3])
def test_empty_tuple_multiblock_field_reads_back(tmp_path: Path, components: int) -> None:
    """A MultiBlock's own empty-tuple field array reads back like a dataset's."""
    path = tmp_path / "empty-mb-field.pv"
    mb = pv.MultiBlock({"sphere": pv.Sphere()})
    empty = pv.convert_array(np.empty((0, components)), name="empty")
    mb.GetFieldData().AddArray(empty)
    pvzstd.write(mb, path)

    restored = pvzstd.read(path)
    actual = np.asarray(restored.field_data["empty"])
    assert actual.dtype == np.float64
    assert actual.shape == ((0,) if components == 1 else (0, components))
    assert restored["sphere"] == mb["sphere"]
