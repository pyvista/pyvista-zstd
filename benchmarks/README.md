### `pyvista-zstd` Benchmarks

This directory contains benchmark scripts for `pyvista-zstd`.

`benchmark-mesh-filters.py` compares format-version-2 storage with forced and
automatically selected format-version-3 mesh filters. It measures direct
transform/probe cost plus end-to-end write time, read time, and file size on
ordered, scrambled, and incompressible random synthetic triangle meshes and
real meshes from `pyvista.examples`.

The example download functions use PyVista's normal data cache and may fetch a
mesh on the first run. Dataset loading is completed before timing starts.
