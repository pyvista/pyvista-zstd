PyVista Examples Benchmarks
===========================

The following benchmarks evaluate `pyvista-zstd` performance against VTK's native XML writer
using datasets from `pyvista.examples <https://docs.pyvista.org/api/examples/dataset_gallery#dataset-gallery>`__.

Datasets tested include:

- :class:`pyvista.MultiBlock`
- :class:`pyvista.PolyData`
- :class:`pyvista.ImageData`
- :class:`pyvista.UnstructuredGrid` 
- :class:`pyvista.RectilinearGrid`
- :class:`pyvista.StructuredGrid`
- :class:`pyvista.ExplicitStructuredGrid`.


File Size Comparison
--------------------

.. figure:: figures/examples-fig0.png
   :alt: File size comparison: pyvista-zstd vs VTK XML
   :align: center

   File size: pyvista-zstd vs. VTK XML (default compression)

Most datasets show smaller file sizes when using ``pyvista-zstd`` compared to VTK XML.
Points fall below the unity line (dashed), indicating reduced disk usage with
Zstandard compression.


Write Time Comparison
---------------------

.. figure:: figures/examples-fig1.png
   :alt: Write performance comparison: pyvista-zstd vs VTK XML
   :align: center

   Write performance: pyvista-zstd vs. VTK XML (default compression)

Write operations with ``pyvista-zstd`` are several times faster across all dataset
types.  Larger datasets benefit more from multi-threaded compression.


Read Time Comparison
--------------------

.. figure:: figures/examples-fig2.png
   :alt: Read performance comparison: pyvista-zstd vs VTK XML
   :align: center

   Read performance: pyvista-zstd vs. VTK XML (default compression)

Reading ``pyvista-zstd`` files is significantly faster than VTK XML.

.. note::
   Performance gains persist even without multi-threading.


Top 10 Largest Datasets: Speedup and Compression Ratios
-------------------------------------------------------

.. figure:: figures/examples-fig3.png
   :alt: Read/write speedup and compression ratios for top 10 largest datasets
   :align: center

   Read/Write Speedup and Compression Ratios for Top 10 pyvista.examples Datasets

The left panel shows read/write speedups: all top datasets achieve multiple-fold
speed increases when using ``pyvista-zstd``.  
The right panel shows compression ratios (log scale): ``pyvista-zstd`` achieves higher
compression than VTK XML in all cases while retaining full dataset fidelity.


Benchmark Script
----------------

The benchmarks were executed using the following Python script:

.. literalinclude:: ../../benchmarks/benchmark-examples.py
   :language: python
