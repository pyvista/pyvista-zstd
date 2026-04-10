"""VTK zstandard compression library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("pyvista-zstd")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from pyvista_zstd.pyvista_zstd import FILE_VERSION
from pyvista_zstd.pyvista_zstd import Reader
from pyvista_zstd.pyvista_zstd import Writer
from pyvista_zstd.pyvista_zstd import read
from pyvista_zstd.pyvista_zstd import write

__all__ = ["FILE_VERSION", "Reader", "Writer", "read", "write"]
