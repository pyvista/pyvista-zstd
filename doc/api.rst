API Reference
=============

API reference for ``zvtk``.

Convenience Functions
---------------------
``zvtk`` exposes two convenience functions to easily read and write compressed
datasets to disk using `Zstandard <https://github.com/facebook/zstd>`_


.. autosummary::
   :toctree: _autosummary

   zvtk.read
   zvtk.write


Classes
-------
``zvtk`` also exposes two classes to fine tune how datasets are read and
written. For example, using the :class:`zvtk.Reader`, you can select the arrays
you wish to read in or even progressively read in individual datasets from a
:class:`pyvista.MultiBlock`.

.. autosummary::
   :toctree: _autosummary

   zvtk.Reader
   zvtk.Writer
