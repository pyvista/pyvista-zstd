"""Application suffixes do not bypass container validation."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from pyvista_zstd import AppendReader
from pyvista_zstd import Reader
from pyvista_zstd import append_arrays
from pyvista_zstd import write
from pyvista_zstd._capi import ContainerFormatError


def test_application_extension_partial_read(tmp_path) -> None:
    """Custom filenames retain selective geometry and field reads."""
    original = tmp_path / "data.pv"
    mesh = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
    write(mesh, original)
    values = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    append_arrays(original, {"modes": values}, shuffle=True)
    target = original.rename(tmp_path / "data.gmm")
    with pytest.raises(ValueError, match="Filename"):
        Reader(target)
    with pytest.raises(ValueError, match="Filename"):
        AppendReader(target)
    with Reader(target, check_extension=False) as reader:
        reader.selected_field_arrays = set()
        actual = reader.read()
        assert not actual.field_data
        np.testing.assert_array_equal(actual.points, mesh.points)
    with AppendReader(target, check_extension=False) as reader:
        assert reader.read_array("modes").tobytes() == values.tobytes()


def test_application_extension_still_validates_contents(tmp_path) -> None:
    """Disabling the suffix check does not allow invalid container bytes."""
    target = tmp_path / "bad.gmm"
    target.write_bytes(b"not a PV container")
    with pytest.raises(RuntimeError, match="container"):
        Reader(target, check_extension=False)
    with AppendReader(target, check_extension=False) as reader, pytest.raises(ContainerFormatError):
        reader.read_array("modes")
