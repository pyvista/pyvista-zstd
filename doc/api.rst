API Reference
=============

API reference for ``pyvista-zstd``.

Convenience Functions
---------------------
``pyvista-zstd`` exposes two convenience functions to easily read and write compressed
datasets to disk using `Zstandard <https://github.com/facebook/zstd>`_


.. autosummary::
   :toctree: _autosummary

   pyvista_zstd.read
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
