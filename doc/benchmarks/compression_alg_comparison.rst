Compression Algorithm Comparison
================================

The table below summarizes the performance and compression achieved across all
available VTK compression algorithms and `Zstandard
<https://github.com/facebook/zstd>`_ using ``pyvista-zstd``.

``pyvista-zstd`` performed the best with a write speed of 1.8 GB/s with a compression
ratio of over 3x.

**Write Performance**

+------------------------------+-------------------+-------------------+----------------------+
| File Type / Method           | Write Speed       | Compression Ratio | Notes                |
+==============================+===================+===================+======================+
| Legacy VTK (.vtk)            | 465 MB/s          | 0.88              | Significant overhead |
+------------------------------+-------------------+-------------------+----------------------+
| VTK XML, none                | 256 MB/s          | 0.70              | Significant overhead |
+------------------------------+-------------------+-------------------+----------------------+
| VTK XML, zlib                | 105 MB/s          | 2.52              | VTK Default          |
+------------------------------+-------------------+-------------------+----------------------+
| VTK XML, lz4                 | 401 MB/s          | 1.47              |                      |
+------------------------------+-------------------+-------------------+----------------------+
| VTK XML, lzma                | 9.93 MB/s         | 3.10              |                      |
+------------------------------+-------------------+-------------------+----------------------+
| VTK HDF (.vtkhdf), lvl0      | 1733 MB/s         | 0.93              | No compression       |
+------------------------------+-------------------+-------------------+----------------------+
| VTK HDF (.vtkhdf), lvl4      | 137 MB/s          | 2.37              | Default compression  |
+------------------------------+-------------------+-------------------+----------------------+
| pyvista-zstd (.pv), lvl3     | 711 MB/s          | 3.02              | Threads = 0          |
+------------------------------+-------------------+-------------------+----------------------+
| **pyvista-zstd (.pv), lvl3** | **1845 MB/s**     | **3.02**          | **Threads = 4**      |
+------------------------------+-------------------+-------------------+----------------------+
| pyvista-zstd (.pv), lvl22    | 15.8 MB/s         | 3.79              | All threads (-1)     |
+------------------------------+-------------------+-------------------+----------------------+


Benchmarks performed on the following environment:

.. code:: bash

   --------------------------------------------------------------------------------
                   OS : Linux (NixOS 25.05)
               CPU(s) : 24
              Machine : x86_64
         Architecture : 64bit
                  RAM : 188.5 GiB
          Environment : IPython
          File system : ext4
           GPU Vendor : NVIDIA Corporation
         GPU Renderer : NVIDIA GeForce RTX 4090/PCIe/SSE2
          GPU Version : 4.5.0 NVIDIA 570.153.02
        Render Window : vtkXOpenGLRenderWindow
     MathText Support : True

     Python 3.12.11 (main, Jun  3 2025, 15:41:47) [GCC 14.2.1 20250322]

              pyvista : 0.47.dev0
                  vtk : 9.4.2
