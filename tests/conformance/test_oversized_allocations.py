"""
An allocation the append and stream paths cannot make comes back as a status.

Both paths size a buffer from a 64-bit length out of the container's trailer,
and both used to wrap that resize in ``catch (const std::bad_alloc &)``. A
resize past the container type's ``max_size()`` does not throw that -- it throws
``std::length_error``, which is not a ``bad_alloc``, so the narrow handler never
caught it on any target. The reader was widened first; these are the same
handler in the other translation units.

``POISON_DECOMPRESSED_SIZE`` is answered by a different mechanism on each
target, and the assertions accept either, because the crafted file is the same
file on both:

* On a 64-bit host it is past every standard library's ``max_size()``, so the
  resize throws ``std::length_error`` and the widened handler is what turns it
  into ``PVZSTD_E_NOMEM`` instead of letting it out of the library.
* On a 32-bit target -- a WebAssembly build is one -- it is also past
  ``SIZE_MAX``, so the guard now in front of the resize refuses it as
  ``PVZSTD_E_FORMAT`` and no allocation is attempted at all.

The writer's three sites are not driven from here. Two of them size themselves
from a ``std::string`` the writer just built, so there is no length to poison;
the third takes ``nbytes`` from the caller, and a caller able to pass a length
past ``SIZE_MAX`` is one that could not have allocated the buffer it describes.
Their handlers were widened for the same reason, but a public entry point
cannot reach them, and a seam added to make it possible would be testing the
seam.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from container_surgery import rebuild
from container_surgery import split_frames
import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd import _capi

if TYPE_CHECKING:
    from pathlib import Path

POISON_DECOMPRESSED_SIZE = 2**64 - 2**32 + 2**31
"""
A declared frame size no allocator on any target can serve.

Above ``PTRDIFF_MAX``, so a 64-bit ``resize`` refuses it by throwing
``std::length_error`` rather than by asking the allocator for it -- which is the
throw the narrow handler could not catch. Its low 32 bits are ``2**31``, so a
32-bit build that narrowed it instead of refusing it would still ask for 2 GiB
rather than truncating to something small enough to succeed quietly.
"""

# PVZSTD_E_FORMAT and PVZSTD_E_NOMEM. Which one comes back is a property of the
# target's size_t, not of the file; both mean the allocation was refused.
REFUSALS = (2, 5)

APPENDED = {"step_1_ids": np.arange(16, dtype=np.int64)}


def _dataset() -> pv.DataSet:
    rng = np.random.default_rng(5)
    ds = pv.Sphere(theta_resolution=10, phi_resolution=10)
    ds.point_data["scal_f64"] = rng.random(ds.n_points)
    return ds


@pytest.fixture
def container(tmp_path: Path) -> bytes:
    """Write a small single-dataset container and return its bytes."""
    path = tmp_path / "surgery_source.pv"
    pz.write(_dataset(), path, progress_bar=False)
    return path.read_bytes()


def _poison_file_metadata_size(raw: bytes) -> bytes:
    """
    Declare an impossible decompressed size for the file-metadata payload.

    That frame is the last one, and it is the first thing both an append and a
    stream open decompress -- so the poisoned length reaches the resize before
    anything else in either path can refuse the file for another reason.
    """
    frames = split_frames(raw)
    frames[-1][1] = POISON_DECOMPRESSED_SIZE
    return rebuild(frames)


def _stream_open(path: Path) -> int:
    """Open *path* for streaming and return the status, freeing on success."""
    lib = _capi._load()  # noqa: SLF001
    lib.pvzstd_stream_open.restype = ctypes.c_int
    lib.pvzstd_stream_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.pvzstd_stream_free.restype = None
    lib.pvzstd_stream_free.argtypes = [ctypes.c_void_p]

    handle = ctypes.c_void_p()
    status = int(lib.pvzstd_stream_open(str(path).encode("utf-8"), ctypes.byref(handle)))
    if status == _capi._STATUS_OK:  # noqa: SLF001
        lib.pvzstd_stream_free(handle)
    else:
        assert not handle, "a refused open must not hand back a stream"
    return status


def test_the_control_container_still_appends_and_streams(container: bytes, tmp_path: Path) -> None:
    """Rebuilt but unpoisoned, the container must still work, or nothing below means anything."""
    path = tmp_path / "control.pv"
    path.write_bytes(rebuild(split_frames(container)))

    assert _stream_open(path) == _capi._STATUS_OK  # noqa: SLF001
    pz.append_arrays(path, APPENDED)
    assert np.array_equal(pz.read_array(path, "step_1_ids"), APPENDED["step_1_ids"])


def test_append_refuses_a_frame_size_no_allocator_can_serve(container: bytes, tmp_path: Path) -> None:
    """
    The append reports the refusal rather than letting the throw out.

    Before the handler was widened, ``std::length_error`` walked past it to the
    entry point's ``catch (...)``. That is a status too, so what this pins is
    the property rather than the route: no exception crosses the C ABI, and no
    build that turns the throw into an abort meets one here.
    """
    path = tmp_path / "poisoned.pv"
    path.write_bytes(_poison_file_metadata_size(container))

    with pytest.raises(_capi.PvzstdError) as caught:
        pz.append_arrays(path, APPENDED)
    assert caught.value.status in REFUSALS, (
        f"a frame declaring {POISON_DECOMPRESSED_SIZE} decompressed bytes was refused with "
        f"status {caught.value.status}, which is neither a format error nor an allocation failure"
    )


def test_append_leaves_the_container_alone_when_it_refuses(container: bytes, tmp_path: Path) -> None:
    """The refused append writes nothing: it fails while reading, before it stages anything."""
    path = tmp_path / "poisoned.pv"
    poisoned = _poison_file_metadata_size(container)
    path.write_bytes(poisoned)

    with pytest.raises(_capi.PvzstdError):
        pz.append_arrays(path, APPENDED)

    assert path.read_bytes() == poisoned
    assert list(tmp_path.glob("poisoned.pv.*")) == [], "the refused append left a staging file behind"


def test_stream_open_refuses_a_frame_size_no_allocator_can_serve(container: bytes, tmp_path: Path) -> None:
    """
    The streaming open reports it too, which is the one site the widening changes.

    Every other handler here sits under an entry point whose ``catch (...)``
    already produced the same status. This one sits under an open that closes
    the file and deletes the half-built stream on every failure it *returns*, so
    a throw walking past the handler leaked both.
    """
    path = tmp_path / "poisoned.pv"
    path.write_bytes(_poison_file_metadata_size(container))

    assert _stream_open(path) in REFUSALS
