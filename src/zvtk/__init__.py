"""VTK compression library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zvtk")
except PackageNotFoundError:
    __version__ = "unknown"

from zvtk.zvtk import compress, decompress


__all__ = ["compress", "decompress"]
