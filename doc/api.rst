API Reference
=============

API reference for ``pyvista-zstd``.

Convenience Functions
---------------------
``pyvista-zstd`` exposes three convenience functions to easily read and write compressed
datasets using `Zstandard <https://github.com/facebook/zstd>`_

:func:`pyvista_zstd.read` and :func:`pyvista_zstd.write` work on files.
A container that is already in memory, rather than on disk, is read with
:func:`pyvista_zstd.read_buffer`.


.. autosummary::
   :toctree: _autosummary

   pyvista_zstd.read
   pyvista_zstd.read_buffer
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
