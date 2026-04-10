"""
Compare multiple VTK file compression approaches with pyvista-zstd.

Size in memory: 1993.89 MB

Saved using legacy (.vtk) file type in 4.06 seconds
  File size:            2256.94 MB
  Compression Ratio:    0.883450863826514

Compressed using None in 7.98 seconds
  File size:            2858.94 MB
  Compression Ratio:    0.6974239255525742

Compressed using zlib in 18.85 seconds
  File size:            791.05 MB
  Compression Ratio:    2.5205590295735663

Compressed using lz4 in 4.89 seconds
  File size:            1351.84 MB
  Compression Ratio:    1.4749437646792847

Compressed using lzma in 200.62 seconds
  File size:            643.25 MB
  Compression Ratio:    3.099705445021067

Saved using VTK HDF (.vtkhdf) file type in 1.08 seconds
  Compression level:    0
  File size:            2145.25 MB
  Compression Ratio:    0.9294451359140696

Saved using VTK HDF (.vtkhdf) file type in 14.45 seconds
  Compression level:    4
  File size:            841.36 MB
  Compression Ratio:    2.3698297330779705

Saved using pyvista-zstd (.pv) file type in 2.74 seconds
  Compression level:    3
  Threads:              0
  File size:            660.47 MB
  Compression Ratio:    3.018891167953701

Saved using pyvista-zstd (.pv) file type in 1.08 seconds
  Compression level:    3
  Threads:              4
  File size:            660.41 MB
  Compression Ratio:    3.019175318642047

"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pyvista as pv
import vtk

import pyvista_zstd

tmp_dir = Path("/tmp/pyvista_zstd_test")
tmp_dir.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
results = []

# Generate a single unstructured grid
n_dim = 200
imdata = pv.ImageData(dimensions=(n_dim, n_dim, n_dim))
ugrid = imdata.to_tetrahedra()

ugrid["pdata"] = rng.random(ugrid.n_points)
ugrid["cdata"] = rng.random(ugrid.n_cells)

nbytes = (
    ugrid.points.nbytes
    + ugrid.cell_connectivity.nbytes
    + ugrid.offset.nbytes
    + ugrid.celltypes.nbytes
    + ugrid["pdata"].nbytes
    + ugrid["cdata"].nbytes
)
print(f"Size in memory: {nbytes / 1024**2:.2f} MB")
print()

# save using legacy VTK format
tstart = time.time()
tmp_path = Path("/tmp/ds.vtk")
ugrid.save(tmp_path)
print(f"Saved using legacy (.vtk) file type in {time.time() - tstart:.2f} seconds")
nbytes_disk = tmp_path.stat().st_size
print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
print()


# save using VTK XML format
tmp_path = Path("/tmp/ds.vtu")
compression_type = [None, "zlib", "lz4"]
for compression in compression_type:
    tstart = time.time()
    ugrid.save(tmp_path, compression=compression)
    print(f"Compressed using {compression} in {time.time() - tstart:.2f} seconds")
    nbytes_disk = tmp_path.stat().st_size
    print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
    print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
    print()

###############################################################################
# VTK HDF with no compression

tstart = time.time()
tmp_path = Path("/tmp/ds-no-comp.vtkhdf")
ugrid.save(tmp_path)
print(f"Saved using VTK HDF (.vtkhdf) file type in {time.time() - tstart:.2f} seconds")
nbytes_disk = tmp_path.stat().st_size
print("  Compression level:    0")
print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
print()


###############################################################################
# VTK HDF with compression (level 4, default)

tstart = time.time()
lvl = 4
tmp_path = Path("/tmp/ds-level-4.vtkhdf")
hdf_writer = vtk.vtkHDFWriter()
hdf_writer.SetCompressionLevel(lvl)
hdf_writer.SetInputData(ugrid)
hdf_writer.SetFileName(str(tmp_path))
hdf_writer.Write()
print(f"Saved using VTK HDF (.vtkhdf) file type in {time.time() - tstart:.2f} seconds")
nbytes_disk = tmp_path.stat().st_size
print(f"  Compression level:    {lvl}")
print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
print()


###############################################################################
# pyvista-zstd (no threads, default compression)

level = 3
n_threads = 0
tstart = time.time()
tmp_path = Path("/tmp/ds.pv")
pyvista_zstd.write(ugrid, tmp_path, n_threads=n_threads, level=level)
print(f"Saved using pyvista-zstd (.pv) file type in {time.time() - tstart:.2f} seconds")
nbytes_disk = tmp_path.stat().st_size
print(f"  Compression level:    {level}")
print(f"  Threads:              {n_threads}")
print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
print()

###############################################################################
# pyvista-zstd (4 threads, default compression)
level = 3
n_threads = 4
tstart = time.time()
tmp_path = Path("/tmp/ds.pv")
pyvista_zstd.write(ugrid, tmp_path, n_threads=n_threads, level=level)
print(f"Saved using pyvista-zstd (.pv) file type in {time.time() - tstart:.2f} seconds")
nbytes_disk = tmp_path.stat().st_size
print(f"  Compression level:    {level}")
print(f"  Threads:              {n_threads}")
print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
print()

###############################################################################
# pyvista-zstd (All threads, maximum compression)
level = 22
n_threads = -1
tstart = time.time()
tmp_path = Path("/tmp/ds.pv")
pyvista_zstd.write(ugrid, tmp_path, n_threads=n_threads, level=level)
print(f"Saved using pyvista-zstd (.pv) file type in {time.time() - tstart:.2f} seconds")
nbytes_disk = tmp_path.stat().st_size
print(f"  Compression level:    {level}")
print(f"  Threads:              {n_threads}")
print(f"  File size:            {nbytes_disk / 1024**2:.2f} MB")
print(f"  Compression Ratio:    {nbytes / nbytes_disk}")
print()
