Synthetic Dataset Benchmarks
============================

These benchmarks evaluate ``pyvista-zstd`` performance on synthetic
:class:`pyvista.UnstructuredGrid`s generated from
:class:`pyvista.ImageData`. Dataset sizes range from a handful of KB to over 5
GB.

File Size Comparison
--------------------

.. figure:: figures/synthetic-fig0.png
   :alt: File size comparison: pyvista-zstd vs VTK XML
   :align: center

   File size comparison for synthetic unstructured grids

``pyvista-zstd`` consistently produces smaller files than VTK XML.  The red line indicates
the linear fit ratio between ``pyvista-zstd`` and VTK file sizes, showing a 26% reduction
in file size for ``pyvista-zstd`` files vs. VTK XML using zlib (default).


Write Time Comparison
---------------------

.. figure:: figures/synthetic-fig1.png
   :alt: Write performance comparison: pyvista-zstd vs VTK XML
   :align: center

   Write time comparison for synthetic unstructured grids

``pyvista-zstd`` write times are consistently lower than VTK XML, about 37 times faster
for this dataset.


Read Time Comparison
--------------------

.. figure:: figures/synthetic-fig2.png
   :alt: Read performance comparison: pyvista-zstd vs VTK XML
   :align: center

   Read time comparison for synthetic unstructured grids

Reading ``pyvista-zstd`` files is substantially faster than VTK XML across all dataset
sizes, about 14 times faster than VTK.


Speedup vs Dataset Size
-----------------------

.. figure:: figures/synthetic-fig3.png
   :alt: Read/Write speedup vs dataset size
   :align: center

   Read/Write speedup (pyvista-zstd / VTK XML) versus dataset size

Both read and write operations achieve multiple-fold speedups with ``pyvista-zstd``.
Larger datasets show the most pronounced improvements.

Compression Ratios vs Dataset Size
----------------------------------

.. figure:: figures/synthetic-fig4.png
   :alt: Compression ratios vs dataset size
   :align: center

   Compression ratios (pyvista-zstd vs VTK XML) versus dataset size

pyvista-zstd maintains higher compression than VTK XML for all synthetic dataset sizes.

Benchmark Script
----------------

The benchmarks were executed using the following Python script:

.. literalinclude:: ../../benchmarks/benchmark-synthetic.py
   :language: python
