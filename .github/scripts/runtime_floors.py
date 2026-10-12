# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///
"""Print ``name==floor`` for every runtime dependency declared in pyproject.toml."""

from __future__ import annotations

from pathlib import Path
import sys

from packaging.requirements import Requirement
import tomllib

pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
pins = []
for dep in pyproject["project"]["dependencies"]:
    req = Requirement(dep)
    floors = [s.version for s in req.specifier if s.operator == ">="]
    if len(floors) != 1:
        msg = f"{req.name} needs exactly one >= floor, got {dep!r}"
        raise SystemExit(msg)
    pins.append(f"{req.name}=={floors[0]}")
sys.stdout.write(" ".join(pins) + "\n")
