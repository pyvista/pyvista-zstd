"""
Two appends against one container, running at the same time.

One append runs against a container at a time, and
:func:`pyvista_zstd.append_arrays` enforces it rather than asking callers to.
An append is a read-modify-write, so two of them at once each commit "what was
there plus mine" and the second to land replaces the first's arrays -- with both
callers told they succeeded. Checking for that at the end does not recover it:
before the lock went in, four appends doing equal work reached their commits so
close together that not one of them saw another's result, and three of the four
sets of arrays vanished with every caller reporting success.

The exclusion is a lock file created exclusively beside the container, because
exclusive creation is the one primitive every target this builds for implements
the same way -- ``flock`` returns success on the WebAssembly target without
locking anything. A second append meanwhile is refused at once rather than
queued.

Two things are asserted here, and the second is the one that matters: appends
that run at the same time are serialised rather than interleaved, and a caller
told its append succeeded finds its arrays in the file afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv

import pyvista_zstd as pz
from pyvista_zstd import _capi

if TYPE_CHECKING:
    from pathlib import Path

# Big enough that copying its body takes long enough to replace the file
# underneath the copy, and small enough to be cheap to write. Random doubles
# barely compress, so the container is about this size on disk too.
BULK_DIMENSION = 160

WRITERS = 4
"""Concurrent appends in the racing test. One per array, all released together."""

RACE_TIMEOUT_S = 120.0
"""Ceiling on any wait here; every one of them is normally over in well under a second."""

_CHILD = f"""
import sys
import time
from pathlib import Path

import numpy as np

import pyvista_zstd as pz
from pyvista_zstd import _capi

container, ready, go, name, seed = sys.argv[1:6]
payload = np.random.default_rng(int(seed)).random(1 << 18)

Path(ready).touch()
deadline = time.monotonic() + {RACE_TIMEOUT_S}
while not Path(go).exists():
    if time.monotonic() > deadline:
        print("timeout", flush=True)
        raise SystemExit(1)

# Refused while another append holds the container, so retry until it lets go:
# the point of the lock is that the work is serialised, not that it is dropped.
deadline = time.monotonic() + {RACE_TIMEOUT_S}
while True:
    try:
        pz.append_arrays(container, {{name: payload}})
    except _capi.ContainerBusyError:
        if time.monotonic() > deadline:
            print("busy", flush=True)
            break
        continue
    except _capi.ContainerChangedError:
        print("changed", flush=True)
        break
    except _capi.PvzstdError as err:
        print(f"status {{err.status}}", flush=True)
        break
    else:
        print("ok", flush=True)
        break
"""


def _bulk(seed: int) -> pv.DataSet:
    ds = pv.ImageData(dimensions=(BULK_DIMENSION,) * 3)
    ds.point_data["bulk_f64"] = np.random.default_rng(seed).random(ds.n_points)
    return ds


def _wait_for(predicate, what: str) -> object:
    """Spin until *predicate* returns something truthy, and return it."""
    deadline = time.monotonic() + RACE_TIMEOUT_S
    while True:
        found = predicate()
        if found:
            return found
        if time.monotonic() > deadline:
            pytest.fail(f"waited {RACE_TIMEOUT_S}s for {what}")


def _staging_files(container: Path) -> list[Path]:
    """Whatever an in-flight or abandoned append has left beside *container*."""
    return sorted(container.parent.glob(f"{container.name}.append.*"))


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows will not replace a file another process holds open, so the collision cannot be staged there",
)
def test_a_container_replaced_mid_append_is_refused_rather_than_overwritten(tmp_path: Path) -> None:
    """
    An append whose container is replaced while it works reports it and writes nothing.

    The collision is driven rather than raced for. The child's staging file
    cannot exist before the child has opened and parsed the container, so its
    appearance is a point strictly after the read and strictly before the
    commit -- which is exactly the interval the check exists to cover. The
    replacement is a single ``os.replace``, against a body copy of some tens of
    milliseconds, so what is timing-dependent is only the margin and not the
    ordering.
    """
    container = tmp_path / "shared.pv"
    pz.write(_bulk(11), container, progress_bar=False)

    # What the other writer commits: the same container with its own array.
    theirs = tmp_path / "theirs.pv"
    shutil.copyfile(container, theirs)
    pz.append_arrays(theirs, {"step_1_theirs": np.arange(8, dtype=np.int64)})
    replacement = theirs.read_bytes()

    ready = tmp_path / "ready"
    go = tmp_path / "go"
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _CHILD, str(container), str(ready), str(go), "step_1_ours", "5"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(ready.exists, "the child to finish importing")
        go.touch()
        _wait_for(lambda: _staging_files(container), "the child's append to open its staging file")
        theirs.replace(container)
        verdict = child.communicate(timeout=RACE_TIMEOUT_S)[0].strip()
    finally:
        child.kill()

    assert verdict == "changed", (
        f"the append reported {verdict!r} for a container that was replaced while it was staging"
    )
    assert container.read_bytes() == replacement, "the refused append overwrote the other writer's result"
    assert _staging_files(container) == [], "the refused append left its staging file behind"


def test_concurrent_appends_are_serialised_and_lose_nothing(tmp_path: Path) -> None:
    """
    Genuinely concurrent, not sequential: every writer is released at once.

    Each child imports and builds its array first, touches a file to say it is
    ready, and then spins on a shared release file, so the appends themselves
    overlap rather than the interpreter startups. A child refused because
    another append holds the container retries until it is let in, which is what
    the refusal is for: the work is serialised, not dropped.

    Without the lock this fails on its last assertion and not its first -- every
    child reports success and most of their arrays are not in the file.
    """
    container = tmp_path / "shared.pv"
    pz.write(_bulk(3), container, progress_bar=False)
    before = pz.read(container).point_data["bulk_f64"]

    go = tmp_path / "go"
    names = [f"step_1_writer{i}" for i in range(WRITERS)]
    children = []
    for i, name in enumerate(names):
        ready = tmp_path / f"ready{i}"
        children.append(
            (
                name,
                ready,
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-c", _CHILD, str(container), str(ready), str(go), name, str(i)],
                    stdout=subprocess.PIPE,
                    text=True,
                ),
            )
        )
    try:
        for _, ready, _proc in children:
            _wait_for(ready.exists, "every child to finish importing")
        go.touch()
        verdicts = {name: proc.communicate(timeout=RACE_TIMEOUT_S)[0].strip() for name, _, proc in children}
    finally:
        for _, _, proc in children:
            proc.kill()

    assert set(verdicts.values()) == {"ok"}, (
        f"appends that retry until the container is free should all get in eventually: {verdicts}"
    )

    final = pz.read(container)
    assert np.array_equal(final.point_data["bulk_f64"], before), "a concurrent append damaged the container"
    landed = set(pz.AppendReader(container).field_array_names)
    assert set(names) <= landed, (
        f"{set(names) - landed} were reported as appended and are not in the container; "
        "one append committed over another's result"
    )
    assert _staging_files(container) == [], "a finished append left its staging file behind"
    assert not (tmp_path / f"{container.name}.append.lock").exists(), "an append kept its lock"


def test_a_lock_left_by_a_killed_append_refuses_the_next_one(tmp_path: Path) -> None:
    """
    The cost of locking with a file, asserted rather than left to be discovered.

    A process killed mid-append leaves its lock behind and every later append is
    refused until the file is deleted. That is the trade the lock makes, so the
    refusal names the file and clears the moment it goes.
    """
    container = tmp_path / "locked.pv"
    pz.write(_bulk(9), container, progress_bar=False)
    committed = container.read_bytes()

    stale = tmp_path / f"{container.name}.append.lock"
    stale.touch()
    with pytest.raises(_capi.ContainerBusyError, match=r"locked\.pv\.append\.lock"):
        pz.append_arrays(container, {"step_1_blocked": np.arange(8, dtype=np.int64)})
    assert container.read_bytes() == committed, "a refused append touched the container"

    stale.unlink()
    pz.append_arrays(container, {"step_1_blocked": np.arange(8, dtype=np.int64)})
    assert "step_1_blocked" in pz.AppendReader(container).field_array_names
