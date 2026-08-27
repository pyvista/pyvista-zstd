"""
Hold the field-array index and reads to what was appended.

Requires the core; ``PVZSTD_LIBRARY`` selects which build answers. The C ABI
names field arrays the way a caller does -- bare, without the dataset-UID
prefix or the ``__field_data`` suffix -- and takes that list from the dataset
metadata rather than by scanning frame names.

Reads are compared bit-for-bit against the arrays handed to ``append_arrays``,
which holds the writer to account too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd import _capi
from pyvista_zstd.append import AppendReader

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

# No availability gate: the core is the only implementation, so a machine that
# cannot load it must fail these tests rather than skip them.

UID_N_CHAR = 16
FIELD_DATA_SUFFIX = "__field_data"

# A name ending in the frame suffix, so its frame is doubled. Both index-building
# strategies handle it, but a user can really choose this name.
TRAP_NAME = "odd__field_data"


def _seed(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Build a container, and hand back the field arrays it was given."""
    rng = np.random.default_rng(29)
    ds = pv.Sphere(theta_resolution=10, phi_resolution=10)
    ds.point_data["disp"] = rng.random((ds.n_points, 3))
    ds.cell_data["ids"] = np.arange(ds.n_cells, dtype=np.int64)
    ds.field_data["born_with_it"] = np.array([2.5, -1.0])

    path = tmp_path / "seed.pv"
    pz.write(ds, path, progress_bar=False)
    appended = {
        "step_1_u": rng.random((40, 3)),
        "step_1_ids": np.arange(9, dtype=np.int32),
        TRAP_NAME: np.linspace(0.0, 1.0, 12),
    }
    pz.append_arrays(path, appended)
    # ``born_with_it`` rides along from the original write, so the expected set
    # is the appended arrays plus it -- not the append call alone.
    return path, {**appended, "born_with_it": ds.field_data["born_with_it"]}


def test_index_lists_exactly_the_field_arrays_written(tmp_path) -> None:
    """The index is the set that was written -- no more, no fewer."""
    path, expected = _seed(tmp_path)

    with _capi.CoreReader(path) as reader:
        assert set(reader.field_array_names()) == set(expected)
    assert set(AppendReader(path).field_array_names) == set(expected)

    # If the trap name stops being written the case above weakens silently.
    assert TRAP_NAME in expected


def test_reads_are_bit_identical_to_what_was_written(tmp_path) -> None:
    """Every field array reads back exactly as it was handed in."""
    path, expected = _seed(tmp_path)
    reader = AppendReader(path)

    for name, want in expected.items():
        got = reader.read_array(name)
        assert got.dtype == want.dtype, name
        assert got.shape == want.shape, name
        assert np.array_equal(got, want), name


def test_find_field_resolves_to_the_frame_the_name_belongs_to(tmp_path) -> None:
    """
    The returned index addresses the block's own frame, not a neighbour's.

    ``pvzstd_find_field_array`` indexes the ordinary array list, so the array it
    points at must be the one whose stored name is built from this bare name.
    """
    path, _ = _seed(tmp_path)
    with _capi.CoreReader(path) as reader:
        stored = reader.names()
        for name in reader.field_array_names():
            index = reader.find_field(name)
            assert index is not None, name
            assert stored[index].endswith(f"{name}{FIELD_DATA_SUFFIX}")
            assert stored[index][UID_N_CHAR:] == f"{name}{FIELD_DATA_SUFFIX}"


def test_find_field_declines_arrays_that_are_not_field_data(tmp_path) -> None:
    """
    Negative control: point and cell arrays are not field arrays.

    Both names are really in the container, just not field data, so a lookup
    that answers for them is answering the wrong question.
    """
    path, _ = _seed(tmp_path)
    with _capi.CoreReader(path) as reader:
        assert reader.find_field("disp") is None
        assert reader.find_field("ids") is None
        # ...and the frames really are there, under their stored names.
        stored = reader.names()
        assert any(n.endswith("disp__point_data") for n in stored)
        assert any(n.endswith("ids__cell_data") for n in stored)


def test_read_array_helper_lands_on_the_written_bytes(tmp_path) -> None:
    """The module-level helper returns what was appended, not merely something."""
    path, expected = _seed(tmp_path)
    want = expected["step_1_u"]
    assert np.array_equal(AppendReader(path).read_array("step_1_u"), want)
    assert np.array_equal(pz.read_array(path, "step_1_u"), want)
