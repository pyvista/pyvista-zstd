"""
Prove the documented byte layout against the live writer.

These tests exist to keep ``doc/format/container-v2.md`` honest. They compare
an independent parser (:mod:`ref_reader`, written only from the spec) against
whatever :mod:`pyvista_zstd` actually produces. When the C++ core lands it is
held to the same oracle, so a divergence surfaces here rather than in a
downstream reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest
import pyvista as pv
import ref_reader

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

import pyvista_zstd as pz

UID_N_CHAR = 16

POINT_SUFFIX = "__point_data"
CELL_SUFFIX = "__cell_data"

FIXED_WIDTH_CELLS_VERSION = 2


def _sphere() -> pv.DataSet:
    ds = pv.Sphere(theta_resolution=16, phi_resolution=16)
    rng = np.random.default_rng(0)
    ds.point_data["scal_f32"] = np.arange(ds.n_points, dtype=np.float32)
    ds.point_data["vec_f64"] = rng.random((ds.n_points, 3))
    ds.point_data["flag_u8"] = np.ones(ds.n_points, dtype=np.uint8)
    ds.cell_data["ids_i64"] = np.arange(ds.n_cells, dtype=np.int64)
    return ds


def _unstructured() -> pv.DataSet:
    ds = pv.ImageData(dimensions=(20, 20, 20)).cast_to_unstructured_grid()
    rng = np.random.default_rng(1)
    ds.point_data["smooth_f64"] = np.linspace(0.0, 1.0, ds.n_points)
    ds.point_data["noisy_f32"] = rng.random(ds.n_points).astype(np.float32)
    ds.point_data["ramp_i32"] = np.arange(ds.n_points, dtype=np.int32)
    return ds


def _image() -> pv.DataSet:
    # ImageData carries no explicit points frame -- geometry is implied by
    # dimensions/origin/spacing -- so it needs attached arrays to be a
    # meaningful round-trip case at all.
    ds = pv.ImageData(dimensions=(8, 9, 10))
    ds.point_data["temp_f64"] = np.linspace(-1.0, 1.0, ds.n_points)
    ds.cell_data["mat_i32"] = np.arange(ds.n_cells, dtype=np.int32)
    return ds


def _structured() -> pv.DataSet:
    x, y, z = np.meshgrid(np.arange(5.0), np.arange(6.0), np.arange(7.0))
    return pv.StructuredGrid(x, y, z)


DATASETS = {
    "polydata": _sphere,
    "unstructured": _unstructured,
    "image": _image,
    "structured": _structured,
}


def _round_trip_arrays(
    dataset: pv.DataSet, path: Path, **write_kwargs: object
) -> tuple[ref_reader.Container, pv.DataSet]:
    """Write ``dataset``, then read it back with both the library and the oracle."""
    pz.write(dataset, path, progress_bar=False, **write_kwargs)
    return ref_reader.read(path), pz.read(path)


def _library_array(dataset: pv.DataSet, bare_name: str) -> NDArray[Any] | None:
    """Look up the array the library reconstructed for a given frame name."""
    if bare_name.endswith(POINT_SUFFIX):
        return dataset.point_data[bare_name[: -len(POINT_SUFFIX)]]
    if bare_name.endswith(CELL_SUFFIX):
        return dataset.cell_data[bare_name[: -len(CELL_SUFFIX)]]
    if bare_name == "points":
        return dataset.points
    return None


@pytest.mark.parametrize("label", sorted(DATASETS))
@pytest.mark.parametrize("shuffle", [False, True, "auto"])
def test_spec_reader_matches_library(tmp_path, label, shuffle) -> None:
    """The spec-only parser reproduces every array bit-exactly."""
    dataset = DATASETS[label]()
    path = tmp_path / f"{label}.pv"
    container, restored = _round_trip_arrays(dataset, path, shuffle=shuffle)

    compared = 0
    for header in container.headers:
        expected = _library_array(restored, header.bare_name)
        if expected is None:
            continue
        actual = container.arrays[header.name]
        assert actual.dtype == expected.dtype, header.bare_name
        assert actual.shape == expected.shape, header.bare_name
        assert np.array_equal(actual, expected), header.bare_name
        compared += 1

    assert compared > 0, "no arrays were compared -- the test proved nothing"


@pytest.mark.parametrize("label", sorted(DATASETS))
def test_frames_pair_header_then_payload(tmp_path, label) -> None:
    """Frame count is even and frame order matches ``frame_names`` exactly."""
    path = tmp_path / f"{label}.pv"
    pz.write(DATASETS[label](), path, progress_bar=False)
    container = ref_reader.read(path)

    frames = ref_reader.read_frames(path)
    assert len(frames) % 2 == 0
    assert len(container.headers) == len(frames) // 2

    # ``frame_names`` covers every frame except the trailing file-metadata
    # pair, which is written last and is not addressable by name.
    names = [header.name for header in container.headers]
    assert names[-1].endswith("__pyvista_zstd_metadata")
    assert names[:-1] == container.file_metadata["frame_names"]


def test_shuffle_is_opt_in_and_auto_is_per_array(tmp_path) -> None:
    """``shuffle`` defaults off; ``auto`` decides separately for each array."""
    dataset = _unstructured()

    pz.write(dataset, tmp_path / "off.pv", progress_bar=False)
    off = ref_reader.read(tmp_path / "off.pv")
    assert all(h.filter_id == ref_reader.FILTER_NONE for h in off.headers)

    pz.write(dataset, tmp_path / "on.pv", shuffle=True, progress_bar=False)
    on = ref_reader.read(tmp_path / "on.pv")
    named = [h for h in on.headers if h.bare_name.endswith(POINT_SUFFIX)]
    assert all(h.filter_id == ref_reader.FILTER_SHUFFLE for h in named)

    pz.write(dataset, tmp_path / "auto.pv", shuffle="auto", progress_bar=False)
    auto = ref_reader.read(tmp_path / "auto.pv")
    decisions = {h.bare_name: h.filter_id for h in auto.headers if h.bare_name.endswith(POINT_SUFFIX)}
    # A mix proves the heuristic is evaluated per array rather than per file.
    assert len(set(decisions.values())) > 1, decisions


def test_unshuffle_is_actually_exercised(tmp_path) -> None:
    """
    Negative control: crippling the unshuffle must break the comparison.

    Without this, ``test_spec_reader_matches_library`` could pass while the
    filter branch was never entered or was silently a no-op.
    """
    dataset = _unstructured()
    path = tmp_path / "shuffled.pv"
    pz.write(dataset, path, shuffle=True, progress_bar=False)
    restored = pz.read(path)

    good = ref_reader.read(path)
    mismatched = [
        h.bare_name
        for h in good.headers
        if (ref := _library_array(restored, h.bare_name)) is not None and not np.array_equal(good.arrays[h.name], ref)
    ]
    assert mismatched == [], "baseline must be clean before crippling anything"

    original = ref_reader._unshuffle  # noqa: SLF001
    try:
        ref_reader._unshuffle = lambda buf, itemsize: np.frombuffer(buf, dtype=np.uint8)  # noqa: ARG005, SLF001
        broken = ref_reader.read(path)
        now_wrong = [
            h.bare_name
            for h in broken.headers
            if (ref := _library_array(restored, h.bare_name)) is not None
            and not np.array_equal(broken.arrays[h.name], ref)
        ]
    finally:
        ref_reader._unshuffle = original  # noqa: SLF001

    assert now_wrong, "control did not redden -- the shuffle path is not under test"


def test_fixed_width_cells_omits_the_offsets_frame(tmp_path) -> None:
    """Version 2 drops ``cells_offset`` and records the stride in metadata."""
    homogeneous = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    path = tmp_path / "homogeneous.pv"
    pz.write(homogeneous, path, progress_bar=False)
    container = ref_reader.read(path)

    bare = {h.bare_name for h in container.headers}
    assert "cells_connectivity" in bare
    assert "cells_offset" not in bare
    assert container.ds_metadata["fixed_cell_sizes"] == {"cells": 8}
    assert container.file_metadata["file_version"] == pz.FILE_VERSION


def test_file_version_does_not_imply_filter_presence(tmp_path) -> None:
    """
    A version-2 file may still carry shuffled arrays.

    This is the trap that makes ``file_version >= 1`` an unsafe test for
    whether the optional ``filter_id`` byte is present.
    """
    homogeneous = pv.ImageData(dimensions=(12, 12, 12)).cast_to_unstructured_grid()
    homogeneous.point_data["smooth_f64"] = np.linspace(0.0, 1.0, homogeneous.n_points)

    path = tmp_path / "v2_shuffled.pv"
    pz.write(homogeneous, path, shuffle=True, progress_bar=False)
    container = ref_reader.read(path)

    assert container.file_metadata["file_version"] == FIXED_WIDTH_CELLS_VERSION
    assert any(h.filter_id == ref_reader.FILTER_SHUFFLE for h in container.headers)
