"""
Selectively Load a MultiBlock
-----------------------------

Selectively or progressively load a :class:`pyvista.MultiBlock`.

"""

from pathlib import Path

from pyvista import examples
import pyvista as pv

import zvtk

###############################################################################
# First, download a multiblock dataset from pyvista and save it as a zvtk file.
#
# .. note::
#    Unlike VTK, ``zvtk`` saves composite datasets as a single file.

ds = examples.download_whole_body_ct_male()

zvtk_filename = "whole_body_ct_male.zvtk"
zvtk.write(ds, zvtk_filename)


###############################################################################
# Next, create a :class:`zvtk.Reader` and show the global hierarchy

reader = zvtk.Reader(zvtk_filename)
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
