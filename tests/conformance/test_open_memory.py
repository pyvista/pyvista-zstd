"""
Reading a container out of memory must be the same read as reading the file.

The memory entry point exists for callers with no path to open -- an archive
member, a response body, a build with no filesystem -- so what matters is that
it is the *same* reader, not a second one: identical bytes out of identical
containers, and identical refusals for containers that are damaged or crafted.

The safety cases are written against the memory entry point on purpose. Every
one of them is a check the file path already makes, and a buffer reaches the
parse by a different door; a guard that only the file door passes through is
an out-of-bounds read waiting for a caller who has the bytes rather than a
path.

Requires the core, and does not skip without it -- as
``test_cpp_reader_roundtrip`` does, and for the same reason.
"""

from __future__ import annotations

import ctypes
import gc
import struct
from typing import TYPE_CHECKING
import weakref

import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd import _capi

if TYPE_CHECKING:
    from pathlib import Path

# Mirrors PVZSTD_E_INVALID. Nothing else in this module needs a status by name.
STATUS_INVALID = 7

# Even (frames pair as header/payload) and large enough that n_frames * 16
# wraps a uint64 back to 32 -- so a guard that multiplies before comparing
# sees a tiny index and accepts the file.
POISON_FRAME_COUNT = 2**60 + 2

# Written into an out-parameter before a call that must refuse, so the assert
# that it came back NULL is about the callee and not about ctypes' zeroing.
POISON_HANDLE = 0xDEADBEEF


def _dataset() -> pv.DataSet:
    rng = np.random.default_rng(11)
    ds = pv.Sphere(theta_resolution=14, phi_resolution=14)
    ds.point_data["scal_f64"] = rng.random(ds.n_points)
    ds.point_data["vec_f32"] = rng.random((ds.n_points, 3)).astype(np.float32)
    ds.cell_data["ids_i64"] = np.arange(ds.n_cells, dtype=np.int64)
    ds.field_data["note_i32"] = np.arange(4, dtype=np.int32)
    return ds


@pytest.fixture
def container(tmp_path: Path) -> Path:
    """Write a single-dataset container and return its path."""
    path = tmp_path / "memory.pv"
    pz.write(_dataset(), path, shuffle=True, progress_bar=False)
    return path


def _open_memory(raw: bytes) -> int:
    """Call pvzstd_open_memory on *raw* and return the status, closing on success."""
    lib = _capi._load()  # noqa: SLF001
    handle = ctypes.c_void_p()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    status = lib.pvzstd_open_memory(
        ctypes.c_void_p(buffer.ctypes.data),
        ctypes.c_uint64(buffer.nbytes),
        ctypes.byref(handle),
    )
    if status == _capi._STATUS_OK:  # noqa: SLF001
        lib.pvzstd_close(handle)
    assert handle.value is None or status == _capi._STATUS_OK, (  # noqa: SLF001
        "a refused open must leave *out at NULL"
    )
    return int(status)


def test_the_same_container_reads_identically_from_a_path_and_from_memory(container: Path) -> None:
    """Every array's bytes and every metadata document agree between the two."""
    raw = container.read_bytes()

    with _capi.CoreReader(container) as by_path, _capi.CoreReader(buffer=raw) as by_memory:
        assert by_path.names() == by_memory.names()
        assert by_path.names(), "a container with no arrays would pass this vacuously"

        for index in range(len(by_path)):
            left = by_path.read_at(index)
            right = by_memory.read_at(index)
            assert left.dtype == right.dtype, by_path.names()[index]
            assert left.shape == right.shape, by_path.names()[index]
            assert left.tobytes() == right.tobytes(), by_path.names()[index]

        assert by_path.metadata_documents() == by_memory.metadata_documents()
        assert by_path.ds_metadata_json == by_memory.ds_metadata_json
        assert by_path.file_metadata_json == by_memory.file_metadata_json
        assert by_path.field_array_names() == by_memory.field_array_names()

        left_sizes, left_compressed = by_path.frame_sizes()
        right_sizes, right_compressed = by_memory.frame_sizes()
        assert np.array_equal(left_sizes, right_sizes)
        assert np.array_equal(left_compressed, right_compressed)


def test_the_dataset_read_out_of_memory_matches_the_one_written(container: Path) -> None:
    """The public read reaches the same dataset through bytes as through a path."""
    from_path = pz.read(container)
    from_memory = pz.read_buffer(container.read_bytes())

    assert type(from_path) is type(from_memory)
    assert np.array_equal(from_path.points, from_memory.points)
    for attr in ("point_data", "cell_data", "field_data"):
        left, right = getattr(from_path, attr), getattr(from_memory, attr)
        assert set(left.keys()) == set(right.keys()), attr
        for key in left:
            assert left[key].dtype == right[key].dtype, (attr, key)
            assert np.array_equal(left[key], right[key]), (attr, key)


def test_the_reader_survives_the_bytes_object_going_out_of_scope(container: Path) -> None:
    """The reader holds the buffer, so a caller dropping its own name is safe."""
    reader = _capi.CoreReader(buffer=container.read_bytes())
    names = reader.names()
    assert names

    # Nothing else refers to those bytes now. The core borrows rather than
    # owning, so a reader that did not keep them would read freed memory here.
    for index in range(len(reader)):
        assert reader.read_at(index).nbytes >= 0
    reader.close()


def test_a_truncated_buffer_is_refused(container: Path) -> None:
    """Cutting the trailer off leaves bytes that must not parse."""
    raw = container.read_bytes()
    ok = _capi._STATUS_OK  # noqa: SLF001
    assert _open_memory(raw) == ok, "the control case must open, or the prefixes prove nothing"

    # Shorter than the trailer count, cut inside the index, and cut just short
    # of the trailer: the three ways the bytes can run out.
    for keep in (4, len(raw) - 32, len(raw) - 8):
        assert _open_memory(raw[:keep]) != ok, f"a {keep}-byte prefix of a container was accepted"


def test_an_absurd_frame_count_is_refused_rather_than_multiplied(container: Path) -> None:
    """
    The frame-count guard divides, so a count that overflows the multiply fails.

    ``2**60 + 2`` entries of 16 bytes wrap a uint64 to 32, which a guard
    written as ``n_frames * 16 > available`` would wave through and then index
    a megabyte-sized buffer with.
    """
    raw = bytearray(container.read_bytes())
    raw[-8:] = struct.pack("<Q", POISON_FRAME_COUNT)

    assert _open_memory(bytes(raw)) == _capi._STATUS_FORMAT, (  # noqa: SLF001
        "a frame count whose index would not fit in the container was accepted"
    )


def test_a_header_whose_declared_size_disagrees_with_its_shape_is_refused(tmp_path: Path) -> None:
    """
    A payload size that does not follow from shape and dtype is a buffer overrun.

    Reads honour the declared size while callers size destinations from shape
    and dtype, so the two disagreeing is the file telling the reader to write
    past the end of the caller's array.
    """
    path = tmp_path / "declared.pv"
    ds = pv.Sphere(theta_resolution=8, phi_resolution=8)
    ds.point_data["scal_f64"] = np.arange(ds.n_points, dtype=np.float64)
    pz.write(ds, path, progress_bar=False)

    raw = bytearray(path.read_bytes())
    # The trailer's index entries are (end offset, decompressed size) pairs; the
    # payload frames are the odd ones. Doubling one payload's declared size
    # leaves it disagreeing with the shape and dtype in its header.
    (n_frames,) = struct.unpack("<Q", raw[-8:])
    index_off = len(raw) - 8 - n_frames * 16
    size_off = index_off + 16 + 8  # frame 1 is the first payload
    (declared,) = struct.unpack("<Q", raw[size_off : size_off + 8])
    raw[size_off : size_off + 8] = struct.pack("<Q", declared * 2)

    assert _open_memory(bytes(raw)) != _capi._STATUS_OK, (  # noqa: SLF001
        "a payload size disagreeing with its header's shape and dtype was accepted"
    )


def test_a_zero_size_buffer_is_invalid() -> None:
    """Zero bytes is not a container, and is a misuse rather than a bad file."""
    assert _open_memory(b"") == STATUS_INVALID


def test_a_null_pointer_is_invalid() -> None:
    """NULL with a plausible size must be refused, not dereferenced."""
    lib = _capi._load()  # noqa: SLF001
    # Poisoned, not zeroed: ctypes zero-initialises, so a handle left at its
    # default would pass whether or not the refusal cleared it.
    handle = ctypes.c_void_p(POISON_HANDLE)
    status = lib.pvzstd_open_memory(None, ctypes.c_uint64(1024), ctypes.byref(handle))

    assert int(status) == STATUS_INVALID
    assert handle.value is None, "a refused open left *out holding what the caller had there"


def test_every_refusal_leaves_the_out_pointer_at_null() -> None:
    """
    Both doors clear *out before deciding, so no refusal hands back a stale value.

    A caller who checks the handle rather than the status -- or who reuses one
    across two opens -- otherwise reads whatever was in that slot.
    """
    lib = _capi._load()  # noqa: SLF001
    raw = np.frombuffer(b"not a container", dtype=np.uint8)
    data = ctypes.c_void_p(raw.ctypes.data)

    # A NULL pointer, a zero size, and bytes that are not a container, through
    # the memory door; a NULL path and a missing file through the file door.
    refusals = {
        "null data": (lib.pvzstd_open_memory, (None, ctypes.c_uint64(1024))),
        "zero size": (lib.pvzstd_open_memory, (data, ctypes.c_uint64(0))),
        "not a container": (lib.pvzstd_open_memory, (data, ctypes.c_uint64(raw.nbytes))),
        "null path": (lib.pvzstd_open, (None,)),
        "missing file": (lib.pvzstd_open, (b"no/such/container.pv",)),
    }
    for case, (entry, args) in refusals.items():
        handle = ctypes.c_void_p(POISON_HANDLE)
        assert int(entry(*args, ctypes.byref(handle))) != _capi._STATUS_OK, case  # noqa: SLF001
        assert handle.value is None, case


def test_a_null_out_pointer_is_invalid() -> None:
    """Nowhere to put the reader is a misuse too, and must not be written to."""
    lib = _capi._load()  # noqa: SLF001
    raw = np.frombuffer(b"not a container", dtype=np.uint8)
    status = lib.pvzstd_open_memory(ctypes.c_void_p(raw.ctypes.data), ctypes.c_uint64(raw.nbytes), None)

    assert int(status) == STATUS_INVALID


def test_the_versioned_form_reports_the_file_version(container: Path) -> None:
    """A memory open names the container's version, as the file open does."""
    lib = _capi._load()  # noqa: SLF001
    raw = np.frombuffer(container.read_bytes(), dtype=np.uint8)

    from_memory = ctypes.c_uint32(0)
    handle = ctypes.c_void_p()
    status = lib.pvzstd_open_memory_versioned(
        ctypes.c_void_p(raw.ctypes.data),
        ctypes.c_uint64(raw.nbytes),
        ctypes.byref(handle),
        ctypes.byref(from_memory),
    )
    assert status == _capi._STATUS_OK  # noqa: SLF001
    lib.pvzstd_close(handle)

    from_path = ctypes.c_uint32(0)
    handle = ctypes.c_void_p()
    status = lib.pvzstd_open_versioned(str(container).encode("utf-8"), ctypes.byref(handle), ctypes.byref(from_path))
    assert status == _capi._STATUS_OK  # noqa: SLF001
    lib.pvzstd_close(handle)

    assert from_memory.value == from_path.value


def test_the_core_reader_takes_exactly_one_source(container: Path) -> None:
    """Neither source, or both, is a misuse rather than a silent preference."""
    with pytest.raises(TypeError):
        _capi.CoreReader()
    with pytest.raises(TypeError):
        _capi.CoreReader(container, buffer=container.read_bytes())


def test_the_public_reader_takes_exactly_one_source(container: Path) -> None:
    """The same rule at the public surface, so neither layer guesses."""
    with pytest.raises(TypeError):
        pz.Reader()
    with pytest.raises(TypeError):
        pz.Reader(container, buffer=container.read_bytes())


def test_a_buffer_is_judged_by_its_bytes_and_not_by_a_suffix(container: Path) -> None:
    """
    The suffix check belongs to the path, not to the container.

    A path ending in something else is refused before anything is read; a
    buffer has no name to refuse, so what it holds has to decide.
    """
    with pytest.raises(ValueError, match="Filename must end in"):
        pz.Reader(container.with_suffix(".txt"))

    assert pz.Reader(buffer=container.read_bytes()) is not None
    with pytest.raises(RuntimeError, match="File may be corrupted"):
        pz.Reader(buffer=b"not a container at all, but it is bytes")


def test_a_reader_closes_and_stays_closed(container: Path) -> None:
    """Both doors take a close, and a second one is not an error."""
    for reader in (pz.Reader(container), pz.Reader(buffer=container.read_bytes())):
        assert reader.read().n_points
        reader.close()
        reader.close()

        with pytest.raises(ValueError, match="closed Reader"):
            reader.read()


def test_the_reader_is_a_context_manager(container: Path) -> None:
    """``with`` closes on the way out, through either door."""
    with pz.Reader(container) as by_path:
        from_path = by_path.read()
    with pz.Reader(buffer=container.read_bytes()) as by_memory:
        from_memory = by_memory.read()

    assert np.array_equal(from_path.points, from_memory.points)
    for reader in (by_path, by_memory):
        with pytest.raises(ValueError, match="closed Reader"):
            reader.read()


def test_closing_lets_go_of_the_buffer_the_caller_handed_over(container: Path) -> None:
    """
    The point of closing a memory-backed reader: the caller's bytes are freed.

    A reader kept in a list -- one per response body -- otherwise pins every
    one of those bodies for as long as the list lives.
    """
    raw = np.frombuffer(container.read_bytes(), dtype=np.uint8).copy()
    witness = weakref.ref(raw)

    reader = pz.Reader(buffer=raw)
    assert reader.read().n_points
    del raw
    assert witness() is not None, "the reader must hold the bytes while it is open"

    reader.close()
    gc.collect()
    assert witness() is None, "a closed reader still holds the caller's buffer"


def test_a_buffer_cannot_be_resized_under_an_open_reader(container: Path) -> None:
    """The borrow pins the buffer, so a resize is refused rather than followed."""
    raw = bytearray(container.read_bytes())

    with pz.Reader(buffer=raw) as reader:
        assert reader.read().n_points
        with pytest.raises(BufferError):
            raw.extend(b"\0" * 64)

    # Closing gives the bytes back, resizable again.
    raw.extend(b"\0" * 64)
