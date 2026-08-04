### `pyvista-zstd` Benchmarks

This directory contains benchmark scripts for `pyvista-zstd`.

`benchmark-mesh-filters.py` compares format-version-2 storage with forced and
automatically selected format-version-3 mesh filters. It measures direct
transform/probe cost plus end-to-end write time, read time, and file size on
ordered, scrambled, and incompressible random synthetic triangle meshes and
real meshes from `pyvista.examples`.

The example download functions use PyVista's normal data cache and may fetch a
mesh on the first run. Dataset loading is completed before timing starts.

### Measured transform cost

The table below reports the direct CPU cost of the current implementation. The
values are medians in milliseconds after an untimed warmup on an Intel Xeon
E-2288G, using the default benchmark sizes, single-threaded zstd level 3,
Python 3.13.11, PyVista 0.48.4, VTK 9.6.2, and NumPy 2.5.1. The point and
connectivity probe columns are added together by automatic mode. Forced mode
skips both probes.

| Dataset              | Point encode | Point decode | Triangle encode | Triangle decode | Auto probes |
| -------------------- | -----------: | -----------: | --------------: | --------------: | ----------: |
| Synthetic ordered    |      4.94 ms |      3.26 ms |         8.18 ms |         9.66 ms |     8.61 ms |
| Synthetic scrambled  |      4.78 ms |      3.28 ms |         6.92 ms |        10.44 ms |     8.56 ms |
| Synthetic random     |      3.22 ms |      2.77 ms |         6.52 ms |         9.98 ms |     4.86 ms |
| PyVista bunny        |      0.30 ms |      0.26 ms |         0.57 ms |         0.74 ms |     6.70 ms |
| PyVista horse        |      0.28 ms |      0.27 ms |         0.54 ms |         0.74 ms |    10.16 ms |
| PyVista woman        |      1.03 ms |      1.00 ms |         2.32 ms |         2.95 ms |    11.28 ms |
| PyVista Louis Louvre |      1.47 ms |      1.41 ms |         3.11 ms |         4.18 ms |     7.67 ms |

For these datasets, forced writes spend 0.82 to 13.12 ms in the two encoders.
Reads spend 1.00 to 12.93 ms in the corresponding decoders. Automatic writes
add another 4.86 to 11.28 ms to decide whether the filters are worthwhile. The
fixed probe cost dominates the two smallest examples; forcing the transforms
is therefore faster when the caller already knows that storage is the priority.
