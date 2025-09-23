"""
Compare Saving a Surface DataSet
--------------------------------

Compress a PyVista example dataset using three approaches.

"""

import timeit
from pathlib import Path

from pyvista import examples
import pyvista as pv

import zvtk

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
# Write the dataset out using zvtk

print("zvtk")
zvtk_filename = Path("nefertiti.zvtk")
ttot = timeit.timeit(lambda: zvtk.write(ds, zvtk_filename), number=10)
print(f"  Time to save:   {ttot / 10:.3f} s")
ttot = timeit.timeit(lambda: zvtk.read(zvtk_filename), number=10)
print(f"  Time to read:   {ttot / 10:.3f} s")
print(f"  File size:      {zvtk_filename.stat().st_size / 1024**2:.2f} MB\n")


###############################################################################
# Show the dataset is preserved with zvtk

ds_in = zvtk.read(zvtk_filename)
print("Dataset identical:", ds == ds_in)


###############################################################################
# Show the dataset is preserved with zvtk
ds_in.plot(cpos="yz", zoom=1.3)
