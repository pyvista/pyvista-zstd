"""VTK compression library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("zvtk")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from zvtk.zvtk import FILE_VERSION
from zvtk.zvtk import Reader
from zvtk.zvtk import read
from zvtk.zvtk import write

__all__ = ["FILE_VERSION", "Reader", "read", "write"]
