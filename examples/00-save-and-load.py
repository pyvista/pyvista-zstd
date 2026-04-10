"""
Save and Load DataSet using pyvista-zstd
--------------------------------

"""

import timeit
from pathlib import Path

from pyvista import examples
import pyvista as pv

import pyvista_zstd

###############################################################################
# Download the pyvista carotid dataset and save it to disk using pyvista-zstd's
# convenience function :func:`pyvista-zstd.write`:

ds = examples.download_carotid()
pyvista_zstd.write(ds, "carotid.pv")
print(f"Wrote {Path('carotid.pv').stat().st_size / 1024**2:.3} MB")


###############################################################################
# Read the dataset back in using the convenience function :func:`pyvista-zstd.read`:

ds_in = pyvista_zstd.read("carotid.pv")
ds_in


###############################################################################
# Alternatively, control the arrays which are read in using the
# :class:`pyvista-zstd.Reader` class. First, create the reader. Note how we can view
# the content of the dataset without reading it in.

reader = pyvista_zstd.Reader("carotid.pv")
reader


###############################################################################
# Next, select which arrays to read in. For example, only read in the scalar
# arrays. Then read in the dataset. Note how only one array was read in.
#
# This approach can vastly speed up the time it takes to large datasets with
# arrays you are not interested in.

reader.selected_point_arrays = {"scalars"}
ds_in = reader.read()
ds_in


###############################################################################
# Frame Compression
# -----------------
# Data written to a Zstandard file is typically written in "frames", and
# ``pyvista-zstd`` writes out each array within a :class:`pyvista.DataSet` as an
# individual frame. It's possible to see the compressed and decompressed sizes
# of each individual frame written to a ``pyvista-zstd`` file using
# :meth:`pyvista-zstd.Reader.show_frame_compression`. This can be helpful when
# determining the compressability of the internal arrays of a
# ``pyvista.DataSet``.
#
# In the case of a :class:`pyvista.ImageData`, the points and cells are both
# implicit arrays and therefore only the dimensionality, spacing, and origin
# need to be saved to disk as metadata and are not shown as a frame. Only the
# point, cell, and field arrays will be shown.
#
# For additional reading, see `Zstandard - Concepts
# <https://python-zstandard.readthedocs.io/en/latest/concepts.html>`_
#

print(reader.show_frame_compression())
