pyvista-zstd Compression and Threading Benchmarks
==========================================

These benchmarks evaluate ``pyvista-zstd`` performance across multiple compression
levels and varying numbers of threads. A 500 MB
:class:`pyvista.UnstructuredGrid` was used for compression-level tests, and a
10 GB :class:`pyvista.UnstructuredGrid` was used for multi-threading
benchmarks.

Compression Level Comparison
----------------------------

.. figure:: figures/pyvista-zstd-single-ds-fig0.png
   :alt: Write time (log) and compression ratio vs compression level
   :align: center

   pyvista-zstd write time (log) and compression ratio vs compression level

Write times increase with higher compression levels, while compression ratios
also improve. The plot shows the trade-off between speed and compression.

.. figure:: figures/pyvista-zstd-single-ds-fig4.png
   :alt: Read/Write Performance vs. Compression Levels 
   :align: center

   Read/Write Speed vs. Compression Levels

Write performance peaks around the default compression level (3), while the
read performance is highly dependent on the compression level, approaching the
SSD drive speed (6 GB/s).

.. note::
   Note the severely low write speeds using high (15+) levels of compression.


Threading Performance Comparison
--------------------------------

.. figure:: figures/pyvista-zstd-single-ds-fig2.png
   :alt: Write/read time vs number of threads (log scale)
   :align: center

   pyvista-zstd write/read time vs number of threads

Increasing threads significantly reduces write/read times, with diminishing
returns beyond 8 threads.

.. figure:: figures/pyvista-zstd-single-ds-fig3.png
   :alt: Write/read speed (MB/s) vs number of threads
   :align: center

   pyvista-zstd write/read speed vs number of threads

Read and write speed scales with threads; read speeds are generally higher than
write speeds for the same number of threads.


Benchmark Script
----------------

The benchmarks were executed using the following Python script:

.. literalinclude:: ../../benchmarks/benchmark-single-ds.py
   :language: python
