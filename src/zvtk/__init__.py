"""VTK compression library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("zvtk")
except PackageNotFoundError:
    __version__ = "unknown"

from zvtk.zvtk import compress
from zvtk.zvtk import decompress

__all__ = ["compress", "decompress"]
