"""VTK zstandard compression library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("pyvista-zstd")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from pyvista_zstd.append import AppendReader
from pyvista_zstd.append import append_arrays
from pyvista_zstd.append import read_array
from pyvista_zstd.pyvista_zstd import FILE_VERSION
from pyvista_zstd.pyvista_zstd import ArrayInfo
from pyvista_zstd.pyvista_zstd import DataSetMetadata
from pyvista_zstd.pyvista_zstd import MultiBlockMetadata
from pyvista_zstd.pyvista_zstd import Reader
from pyvista_zstd.pyvista_zstd import Writer
from pyvista_zstd.pyvista_zstd import ZstdFileMetadata
from pyvista_zstd.pyvista_zstd import read
from pyvista_zstd.pyvista_zstd import read_buffer
from pyvista_zstd.pyvista_zstd import write
from pyvista_zstd.thumbnail import ThumbnailInfo
from pyvista_zstd.thumbnail import read_thumbnail

__all__ = [
    "FILE_VERSION",
    "AppendReader",
    "ArrayInfo",
    "DataSetMetadata",
    "MultiBlockMetadata",
    "Reader",
    "ThumbnailInfo",
    "Writer",
    "ZstdFileMetadata",
    "append_arrays",
    "read",
    "read_array",
    "read_buffer",
    "read_thumbnail",
    "write",
]
