API Reference
=============

API reference for ``pyvista-zstd``.

Convenience Functions
---------------------
``pyvista-zstd`` exposes three convenience functions to easily read and write compressed
datasets using `Zstandard <https://github.com/facebook/zstd>`_

:func:`pyvista_zstd.read` and :func:`pyvista_zstd.write` work on files.
A container that is already in memory, rather than on disk, is read with
:func:`pyvista_zstd.read_buffer`. :func:`pyvista_zstd.read_metadata` describes a
container without decompressing its arrays, and :func:`pyvista_zstd.read_array`
reads a single array.


.. autosummary::
   :toctree: _autosummary

   pyvista_zstd.read
   pyvista_zstd.read_buffer
   pyvista_zstd.read_metadata
   pyvista_zstd.read_array
   pyvista_zstd.write


Classes
-------
``pyvista-zstd`` also exposes two classes to fine tune how datasets are read and
written. For example, using the :class:`pyvista_zstd.Reader`, you can select the arrays
you wish to read in or even progressively read in individual datasets from a
:class:`pyvista.MultiBlock`.

.. autosummary::
   :toctree: _autosummary

   pyvista_zstd.Reader
   pyvista_zstd.Writer

The reader also describes a file without decompressing its arrays, and reads
one array at a time:

.. code-block:: python

    with pyvista_zstd.Reader("mesh.pv") as reader:
        meta = reader.dataset_metadata  # type, sizes, array shapes and dtypes
        level = reader.file_metadata.compression_level
        stress = reader.read_array("stress", association="cell")

For a :class:`pyvista.MultiBlock`, index the reader to a block first, as in
``reader[0][1].read_array("Normals", association="point")``.


Metadata
--------
Returned by :func:`pyvista_zstd.read_metadata`,
:attr:`pyvista_zstd.Reader.dataset_metadata` and
:attr:`pyvista_zstd.Reader.file_metadata`.

.. autosummary::
   :toctree: _autosummary

   pyvista_zstd.ContainerMetadata
   pyvista_zstd.DataSetMetadata
   pyvista_zstd.MultiBlockMetadata
   pyvista_zstd.ArrayInfo
   pyvista_zstd.ZstdFileMetadata
