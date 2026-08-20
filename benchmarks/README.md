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

### Relative CPU cost

These percentages normalize the direct costs against the corresponding
format-version-2 end-to-end operation. They come from an 11-repeat run pinned
to one CPU core; the three synthetic datasets use five timed repeats under the
benchmark's large-mesh policy.

| Dataset              | Encoders / v2 write | Auto probes / v2 write | Decoders / v2 read |
| -------------------- | ------------------: | ---------------------: | -----------------: |
| Synthetic ordered    |                5.6% |                   4.7% |              22.3% |
| Synthetic scrambled  |                5.4% |                   4.3% |              36.3% |
| Synthetic random     |                7.1% |                   3.5% |              57.4% |
| PyVista bunny        |                6.6% |                  51.6% |              29.8% |
| PyVista horse        |                6.5% |                  74.7% |              27.8% |
| PyVista woman        |                7.7% |                  24.6% |              29.1% |
| PyVista Louis Louvre |                5.9% |                   9.9% |              32.5% |

The encoders add 5.4 to 7.7 percent of the previous total write time. The
decoders add 22.3 to 57.4 percent of the previous total read time; the real
example meshes cluster between 27.8 and 32.5 percent. Automatic selection adds
3.5 to 74.7 percent of the previous write time. That range is size-dependent:
the fixed probe is 3.5 to 4.7 percent on the million-triangle synthetic meshes
but 51.6 to 74.7 percent on bunny and horse.

These are gross CPU costs before accounting for the reduced zstd workload. The
end-to-end results can still be faster because the transformed representation
compresses more quickly and produces fewer bytes.
