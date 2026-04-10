"""
Selectively Load a MultiBlock
-----------------------------

Selectively or progressively load a :class:`pyvista.MultiBlock`.

"""

from pathlib import Path

from pyvista import examples
import pyvista as pv

import pyvista_zstd

###############################################################################
# First, download a multiblock dataset from pyvista and save it as a pyvista-zstd file.
#
# .. note::
#    Unlike VTK, ``pyvista-zstd`` saves composite datasets as a single file.

ds = examples.download_whole_body_ct_male()

pyvista_zstd_filename = "whole_body_ct_male.pv"
pyvista_zstd.write(ds, pyvista_zstd_filename)


###############################################################################
# Next, create a :class:`pyvista-zstd.Reader` and show the global hierarchy

reader = pyvista_zstd.Reader(pyvista_zstd_filename)
reader


###############################################################################
# Read in a single dataset from the multi-block. Note how we can perform nested
# indexing.

block = reader[1][50].read()
block


###############################################################################
# Alternatively, progressively read in the dataset and apply a filter on each
# block.

blocks = pv.MultiBlock([reader[1][ii].read().contour_labels() for ii in range(56, 76)])

# Plot the rib cage
pl = pv.Plotter()
pl.add_mesh(blocks, multi_colors=True)
pl.view_zx()
pl.camera.up = (0, 0, 1)
pl.show()
