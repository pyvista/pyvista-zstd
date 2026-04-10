"""
Compare Saving a Surface DataSet
--------------------------------

Compress a PyVista example dataset using three approaches.

"""

import timeit
from pathlib import Path

from pyvista import examples
import pyvista as pv

import pyvista_zstd

###############################################################################
# Get the file size of the dataset

print("Original PLY file")
filename = Path(examples.download_nefertiti(load=False))
ds = pv.read(filename)
print(f"File size:            {filename.stat().st_size / 1024**2}\n")


###############################################################################
# Write the dataset out using VTK's legacy format

print("Legacy VTK")
vtk_filename = Path("nefertiti.vtk")
ttot = timeit.timeit(lambda: ds.save(vtk_filename), number=10)
print(f"  Time to save:   {ttot / 10:.3f} s")
ttot = timeit.timeit(lambda: pv.read(vtk_filename), number=10)
print(f"  Time to read:   {ttot / 10:.3f} s")
print(f"  File size:      {vtk_filename.stat().st_size / 1024**2:.2f} MB\n")


###############################################################################
# Write the dataset out using VTK's XML format

print("VTK XML")
vtp_filename = Path("nefertiti.vtp")
ttot = timeit.timeit(lambda: ds.save(vtp_filename), number=5)
print(f"  Time to save:   {ttot / 5:.3f} s")
ttot = timeit.timeit(lambda: pv.read(vtp_filename), number=10)
print(f"  Time to read:   {ttot / 10:.3f} s")
print(f"  File size:      {vtp_filename.stat().st_size / 1024**2:.2f} MB\n")


###############################################################################
# Write the dataset out using pyvista-zstd

print("pyvista-zstd")
pyvista_zstd_filename = Path("nefertiti.pv")
ttot = timeit.timeit(lambda: pyvista_zstd.write(ds, pyvista_zstd_filename), number=10)
print(f"  Time to save:   {ttot / 10:.3f} s")
ttot = timeit.timeit(lambda: pyvista_zstd.read(pyvista_zstd_filename), number=10)
print(f"  Time to read:   {ttot / 10:.3f} s")
print(f"  File size:      {pyvista_zstd_filename.stat().st_size / 1024**2:.2f} MB\n")


###############################################################################
# Show the dataset is preserved with pyvista-zstd

ds_in = pyvista_zstd.read(pyvista_zstd_filename)
print("Dataset identical:", ds == ds_in)


###############################################################################
# Show the dataset is preserved with pyvista-zstd
ds_in.plot(cpos="yz", zoom=1.3)
