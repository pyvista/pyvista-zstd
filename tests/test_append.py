"""
Tests for the append + partial/columnar-read extension.

Covers:

* append-then-read-back round trip (bit-exact for appended arrays),
* incremental column-block streaming + single-column partial read,
* backward compat: a file written by the *unmodified* writer (and never
  appended to) still reads identically; the upstream reader still reads
  an appended file,
* crash / partial-write safety: a truncated append leaves the prior
  committed blocks fully intact.
"""

from __future__ import annotations

import json
import os
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd import _capi
from pyvista_zstd.append import AppendReader
from pyvista_zstd.append import append_arrays
from pyvista_zstd.append import read_array
from pyvista_zstd.pyvista_zstd import DS_METADATA_KEY

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def base_grid() -> pv.PolyData:
    """Return a small grid with point, cell, and field data."""
    ds = pv.Sphere()
    ds.point_data["pdata"] = np.arange(ds.n_points, dtype=np.float64)
    ds.cell_data["cdata"] = np.arange(ds.n_cells, dtype=np.int32)
    ds.field_data["meta"] = np.array([1.0, 2.0, 3.0])
    return ds


def _write_base(path: Path, ds: pv.DataSet) -> Path:
    pz.write(ds, str(path))
    return path


# --------------------------------------------------------------------------
# 1. Append-then-read round trip, bit-exact.
# --------------------------------------------------------------------------
def test_append_then_read_back_bit_exact(tmp_path, base_grid) -> None:
    """An appended array reads back bit-exact."""
    path = _write_base(tmp_path / "a.pv", base_grid)
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((1000, 3)).astype(np.float64)
    append_arrays(path, {"block0": arr})

    back = read_array(path, "block0")
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    assert np.array_equal(back, arr)  # bit-exact


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.uint8])
def test_append_preserves_dtype_and_shape(tmp_path, base_grid, dtype) -> None:
    """Append round-trips any dtype and shape unchanged."""
    path = _write_base(tmp_path / "d.pv", base_grid)
    arr = (np.arange(60).reshape(20, 3)).astype(dtype)
    append_arrays(path, {"blk": arr})
    back = read_array(path, "blk")
    assert back.dtype == np.dtype(dtype)
    assert back.shape == (20, 3)
    assert np.array_equal(back, arr)


def test_multiple_arrays_one_call(tmp_path, base_grid) -> None:
    """Several arrays can be appended in a single call."""
    path = _write_base(tmp_path / "m.pv", base_grid)
    a = np.arange(10, dtype=np.float64)
    b = np.arange(20, dtype=np.float32).reshape(4, 5)
    append_arrays(path, {"a": a, "b": b})
    r = AppendReader(path)
    assert set(r.field_array_names) == {"meta", "a", "b"}
    assert np.array_equal(r.read_array("a"), a)
    assert np.array_equal(r.read_array("b"), b)


# --------------------------------------------------------------------------
# 2. Incremental column-block streaming + single-column partial read.
#    A common pattern: a wide (N, M) result is produced one column at a
#    time, each appended as it becomes available, then individual columns
#    are read back later without loading the whole block.
# --------------------------------------------------------------------------
def test_column_block_streaming_pattern(tmp_path) -> None:
    """Stream columns one at a time, then read single columns back."""
    n_rows, n_cols = 5000, 16
    rng = np.random.default_rng(123)
    full = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    header = rng.standard_normal(n_cols).astype(np.float64)

    # Seed a tiny carrier grid.
    carrier = pv.PolyData(np.zeros((1, 3)))
    path = tmp_path / "stream.pv"
    pz.write(carrier, str(path))

    # Append a header array once, then stream columns as they are produced.
    append_arrays(path, {"header": header})
    for j in range(n_cols):
        append_arrays(path, {f"col_{j:04d}": full[:, j].copy()})

    # Partial read: pull back individual columns, bit-exact, no full load.
    r = AppendReader(path)
    assert r.read_array("header").shape == (n_cols,)
    assert np.array_equal(r.read_array("header"), header)
    for j in (0, 5, n_cols - 1):
        col = r.read_array(f"col_{j:04d}")
        assert col.shape == (n_rows,)
        assert np.array_equal(col, full[:, j])

    # All columns reconstruct the full block exactly.
    recon = np.column_stack([r.read_array(f"col_{j:04d}") for j in range(n_cols)])
    assert np.array_equal(recon, full)


def test_field_array_info_reports_shape_dtype(tmp_path, base_grid) -> None:
    """AppendReader reports the shape and dtype of each field array."""
    path = _write_base(tmp_path / "i.pv", base_grid)
    append_arrays(path, {"x": np.ones((3, 7), dtype=np.float32)})
    info = AppendReader(path).field_array_info["x"]
    assert tuple(info.shape) == (3, 7)
    assert info.dtype == "float32"


# --------------------------------------------------------------------------
# 3. Backward / forward compatibility with the upstream reader.
# --------------------------------------------------------------------------
def test_non_appended_file_reads_identically(tmp_path, base_grid) -> None:
    """
    A never-appended file reads identically to the unmodified writer.

    The append module must not perturb the normal write/read path.
    """
    p1 = _write_base(tmp_path / "ref.pv", base_grid)
    g = pz.read(str(p1))
    assert g.n_points == base_grid.n_points
    assert g.n_cells == base_grid.n_cells
    assert np.array_equal(np.asarray(g.point_data["pdata"]), np.arange(base_grid.n_points))
    assert np.array_equal(np.asarray(g.cell_data["cdata"]), np.arange(base_grid.n_cells))


def test_upstream_reader_sees_appended_blocks_and_original_data(tmp_path, base_grid) -> None:
    """
    The unmodified reader still reads the original data after append.

    Appended blocks are surfaced as field_data; the mesh, point, and cell
    data are untouched.
    """
    path = _write_base(tmp_path / "u.pv", base_grid)
    rng = np.random.default_rng(3)
    extra = rng.standard_normal(2048).astype(np.float64)
    append_arrays(path, {"extra": extra})

    reader = pz.Reader(str(path))
    assert "extra" in reader.available_field_arrays
    assert "meta" in reader.available_field_arrays

    g = reader.read()
    # original mesh + point/cell data intact
    assert g.n_points == base_grid.n_points
    assert g.n_cells == base_grid.n_cells
    assert np.array_equal(np.asarray(g.point_data["pdata"]), np.arange(base_grid.n_points))
    assert np.array_equal(np.asarray(g.cell_data["cdata"]), np.arange(base_grid.n_cells))
    assert "Normals" in g.point_data
    # original field array preserved
    assert np.array_equal(np.asarray(g.field_data["meta"]).ravel(), np.array([1.0, 2.0, 3.0]))
    # appended block visible + bit-exact
    assert np.array_equal(np.asarray(g.field_data["extra"]).ravel(), extra)


def test_kept_frames_are_byte_identical_after_append(tmp_path, base_grid) -> None:
    """
    Append does not re-compress untouched arrays.

    Every compressed byte before the regenerated dataset-metadata frame is
    identical between the original and appended files.
    """
    path = _write_base(tmp_path / "b.pv", base_grid)
    orig = path.read_bytes()

    # Where the first regenerated frame starts. The index comes from the bytes
    # (a harness may parse the container the product no longer does in Python)
    # and the frame names from the core, which is what decides them.
    nf = struct.unpack("<Q", orig[-8:])[0]
    meta = orig[-(8 + nf * 16) : -8]
    ends = [struct.unpack("<QQ", meta[i * 16 : (i + 1) * 16])[0] for i in range(nf)]
    starts = [0, *ends[:-1]]
    with _capi.CoreReader(path) as reader:
        frame_names = json.loads(reader.file_metadata_json)["frame_names"]
    ds_meta_idx = next(i for i, n in enumerate(frame_names) if n.endswith(DS_METADATA_KEY))
    kept_prefix_len = starts[ds_meta_idx * 2]

    append_arrays(path, {"z": np.arange(100, dtype=np.float64)})
    new = path.read_bytes()
    assert orig[:kept_prefix_len] == new[:kept_prefix_len]
    assert kept_prefix_len > 0


# --------------------------------------------------------------------------
# 4. Crash / partial-write safety.
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="denying file creation in a directory needs POSIX permissions and a non-root user",
)
def test_failed_append_does_not_destroy_prior_blocks(tmp_path, base_grid) -> None:
    """
    An append that cannot complete leaves the original file fully readable.

    The failure is staged by taking away the right to create the staging file:
    an append names it through the operating system now, so there is no name to
    occupy in advance the way there was when every append used the same one.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    path = _write_base(vault / "c.pv", base_grid)
    append_arrays(path, {"first": np.arange(50, dtype=np.float64)})
    committed = path.read_bytes()

    vault.chmod(0o500)  # readable and searchable, but nothing new may be created
    try:
        with pytest.raises(_capi.PvzstdError, match="I/O error"):
            append_arrays(path, {"second": np.arange(99, dtype=np.float64)})

        assert path.read_bytes() == committed
        with AppendReader(path) as r:
            assert "first" in r.field_array_names
            assert "second" not in r.field_array_names
            assert np.array_equal(r.read_array("first"), np.arange(50, dtype=np.float64))
    finally:
        vault.chmod(0o700)

    append_arrays(path, {"second": np.arange(99, dtype=np.float64)})
    assert np.array_equal(read_array(path, "second"), np.arange(99, dtype=np.float64))
    assert list(vault.glob("c.pv.append.*")) == []


def test_reader_close_releases_the_file_and_reads_reopen(tmp_path, base_grid) -> None:
    """close() drops the open file; a later read reopens it."""
    path = _write_base(tmp_path / "r.pv", base_grid)
    append_arrays(path, {"a": np.arange(20, dtype=np.float64)})

    r = AppendReader(path)
    assert np.array_equal(r.read_array("a"), np.arange(20, dtype=np.float64))
    r.close()
    r.close()
    append_arrays(path, {"b": np.arange(7, dtype=np.float64)})
    assert np.array_equal(r.read_array("b"), np.arange(7, dtype=np.float64))


def test_manually_truncated_file_does_not_corrupt_committed(tmp_path, base_grid) -> None:
    """
    A stale, half-written temp file is irrelevant.

    The committed .pv is the source of truth and reads fine; a fresh
    append still works, and stages somewhere else rather than picking up
    whatever a crashed one left.
    """
    path = _write_base(tmp_path / "t.pv", base_grid)
    append_arrays(path, {"good": np.arange(33, dtype=np.float64)})
    good_bytes = path.read_bytes()

    # A stale, truncated temp file from a prior crashed append, under a name
    # that append staged into before the name came from the operating system.
    tmp = path.with_suffix(path.suffix + ".append.tmp")
    tmp.write_bytes(good_bytes[: len(good_bytes) // 2])

    # Committed file reads fine; a fresh append still works and cleans up.
    assert np.array_equal(read_array(path, "good"), np.arange(33, dtype=np.float64))
    append_arrays(path, {"good2": np.arange(7, dtype=np.float64)})
    assert np.array_equal(read_array(path, "good2"), np.arange(7, dtype=np.float64))
    assert np.array_equal(read_array(path, "good"), np.arange(33, dtype=np.float64))


# --------------------------------------------------------------------------
# 5. Error handling.
# --------------------------------------------------------------------------
def test_append_duplicate_name_rejected(tmp_path, base_grid) -> None:
    """
    Appending a name that already exists is rejected, and the error names it.

    The clashing name is offered second. The core reports which of the call's
    own arrays it refused; naming the first would pass a one-array test.
    """
    path = _write_base(tmp_path / "e.pv", base_grid)
    append_arrays(path, {"dup": np.arange(5, dtype=np.float64)})
    with pytest.raises(ValueError, match="field array 'dup' already exists"):
        append_arrays(
            path,
            {"fresh": np.arange(3, dtype=np.float64), "dup": np.arange(5, dtype=np.float64)},
        )


def test_append_bad_suffix_rejected(tmp_path) -> None:
    """Appending to a non-.pv file is rejected."""
    with pytest.raises(ValueError, match="must end in"):
        append_arrays(tmp_path / "x.txt", {"a": np.arange(3)})


def test_append_to_multiblock_rejected(tmp_path) -> None:
    """
    Appending to a MultiBlock .pv file is rejected with a clear error.

    A MultiBlock has no single root dataset to attach field arrays to; without
    this guard the over-greedy ``__ds_metadata`` suffix match picked up the
    ``__multiblock__ds_metadata`` frame and crashed with an opaque TypeError.
    """
    multi = pv.MultiBlock()
    multi["a"] = pv.Sphere()
    multi["b"] = pv.Cube()
    path = tmp_path / "multi.pv"
    pz.write(multi, str(path))
    with pytest.raises(NotImplementedError, match="MultiBlock"):
        append_arrays(path, {"extra": np.arange(5, dtype=np.float64)})


def test_read_missing_array_raises(tmp_path, base_grid) -> None:
    """Reading a missing array raises KeyError."""
    path = _write_base(tmp_path / "f.pv", base_grid)
    with pytest.raises(KeyError, match="not found"):
        read_array(path, "nope")


def test_empty_arrays_dict_is_noop(tmp_path, base_grid) -> None:
    """Appending an empty dict leaves the file unchanged."""
    path = _write_base(tmp_path / "n.pv", base_grid)
    before = path.read_bytes()
    append_arrays(path, {})
    assert path.read_bytes() == before


def test_append_then_top_level_api(tmp_path, base_grid) -> None:
    """The convenience names are re-exported at package top level."""
    path = _write_base(tmp_path / "g.pv", base_grid)
    pz.append_arrays(path, {"q": np.arange(4, dtype=np.float64)})
    assert np.array_equal(pz.read_array(path, "q"), np.arange(4, dtype=np.float64))
    assert "q" in pz.AppendReader(path)
