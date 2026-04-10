"""
Synthetic DataSet benchmarks.

Create a synatic unstructured grid from imagedata.

"""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from tqdm import tqdm

import pyvista_zstd

tmp_dir = Path("/tmp/pyvista_zstd_test")
tmp_dir.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
results = []

# Synthetic ImageData -> UnstructuredGrid
for n_dim in tqdm(range(10, 300, 20)):  # adjust step for speed
    name = f"synthetic_{n_dim}^3"
    imdata = pv.ImageData(dimensions=(n_dim, n_dim, n_dim))
    ugrid = imdata.to_tetrahedra()

    # Add point/cell data
    ugrid["sample_point_data"] = rng.random((ugrid.n_points, 3))
    ugrid["sample_cell_data"] = rng.random((ugrid.n_cells, 3))

    # Save/read timing for VTK
    filename = tmp_dir / f"{name}.vtu"
    tstart = time.time()
    ugrid.save(filename)
    vtk_write_time = time.time() - tstart

    tstart = time.time()
    pv.read(filename)
    vtk_read_time = time.time() - tstart

    vtk_fs = filename.stat().st_size

    # Save/read timing for pyvista-zstd
    out_file = tmp_dir / f"{name}.pv"
    tstart = time.time()
    writer = pyvista_zstd.Writer(ugrid, out_file)
    writer.write(n_threads=4)
    pyvista_zstd_write_time = time.time() - tstart
    pyvista_zstd_fs = out_file.stat().st_size

    tstart = time.time()
    reader = pyvista_zstd.Reader(out_file)
    reader.read()
    pyvista_zstd_read_time = time.time() - tstart

    print(reader.nbytes / 1024**2)
    results.append(
        {
            "name": name,
            "ds_type": "UnstructuredGrid",
            "raw_nbytes": reader.nbytes,
            "vtk_nbytes": vtk_fs,
            "pyvista_zstd_nbytes": pyvista_zstd_fs,
            "vtk_write_time": vtk_write_time,
            "pyvista_zstd_write_time": pyvista_zstd_write_time,
            "vtk_read_time": vtk_read_time,
            "pyvista_zstd_read_time": pyvista_zstd_read_time,
        }
    )

# Build DataFrame and compute ratios
df = pd.DataFrame(results)
df["write_speedup"] = df["vtk_write_time"] / df["pyvista_zstd_write_time"]
df["read_speedup"] = df["vtk_read_time"] / df["pyvista_zstd_read_time"]
df["compression_ratio_vtk"] = df["vtk_nbytes"] / df["raw_nbytes"]
df["compression_ratio_pyvista_zstd"] = df["pyvista_zstd_nbytes"] / df["raw_nbytes"]

summary = df.sort_values("raw_nbytes", ascending=False)


summary = df.copy()  # synthetic benchmark results

sns.set(style="whitegrid")

# Convert bytes to MB for better readability
summary["raw_MB"] = summary["raw_nbytes"] / 1e6
summary["vtk_MB"] = summary["vtk_nbytes"] / 1e6
summary["pyvista_zstd_MB"] = summary["pyvista_zstd_nbytes"] / 1e6


###############################################################################
# 1. File size comparison with ratio fit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=summary, x="vtk_MB", y="pyvista_zstd_MB", s=50)

# 1:1 reference line
max_val = max(summary["vtk_MB"].max(), summary["pyvista_zstd_MB"].max())
plt.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 line")

# Linear fit forced through origin to get ratio
ratio_coef = 1 / float(np.sum(summary["vtk_MB"] * summary["pyvista_zstd_MB"]) / np.sum(summary["vtk_MB"] ** 2))
fit_line = 1 / ratio_coef * summary["vtk_MB"]
plt.plot(summary["vtk_MB"], fit_line, "r-", lw=2, label=f"Fit ratio: {ratio_coef:.2f}× VTK")

plt.xlabel("VTK XML file size (MB)")
plt.ylabel("pyvista-zstd file size (MB)")
plt.title("File size comparison: pyvista-zstd vs VTK XML")
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# 2. Write time comparison with ratio fit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=summary, x="vtk_write_time", y="pyvista_zstd_write_time", s=50)

max_val = max(summary["vtk_write_time"].max(), summary["pyvista_zstd_write_time"].max())
plt.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 line")

ratio_coef = 1 / float(
    np.sum(summary["vtk_write_time"] * summary["pyvista_zstd_write_time"]) / np.sum(summary["vtk_write_time"] ** 2)
)
fit_line = 1 / ratio_coef * summary["vtk_write_time"]
plt.plot(summary["vtk_write_time"], fit_line, "r-", lw=2, label=f"Fit ratio: {ratio_coef:.2f}× VTK")

plt.xlabel("VTK write time (s)")
plt.ylabel("pyvista-zstd write time (s)")
plt.title("Write performance comparison: pyvista-zstd vs VTK XML")
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# 3. Read time comparison with ratio fit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=summary, x="vtk_read_time", y="pyvista_zstd_read_time", s=50)

max_val = max(summary["vtk_read_time"].max(), summary["pyvista_zstd_read_time"].max())
plt.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 line")

ratio_coef = 1 / float(
    np.sum(summary["vtk_read_time"] * summary["pyvista_zstd_read_time"]) / np.sum(summary["vtk_read_time"] ** 2)
)
fit_line = 1 / ratio_coef * summary["vtk_read_time"]
plt.plot(summary["vtk_read_time"], fit_line, "r-", lw=2, label=f"Fit ratio: {ratio_coef:.2f}× VTK")

plt.xlabel("VTK read time (s)")
plt.ylabel("pyvista-zstd read time (s)")
plt.title("Read performance comparison: pyvista-zstd vs VTK XML")
plt.legend()
plt.tight_layout()
plt.show()


###############################################################################
# 4. Ratios summary (speedup and compression)
# 4a. Speedup summary
summary_speedup = summary.melt(
    id_vars=["raw_MB"],
    value_vars=["write_speedup", "read_speedup"],
    var_name="metric",
    value_name="value",
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=summary_speedup, x="raw_MB", y="value", hue="metric", marker="o")
plt.xlabel("Raw dataset size (MB)")
plt.ylabel("Speedup (pyvista-zstd / VTK XML)")
plt.title("Read/Write Speedup vs Dataset Size")
plt.tight_layout()
plt.show()

# 4b. Compression ratio summary
summary_compression = summary.melt(
    id_vars=["raw_MB"],
    value_vars=["compression_ratio_vtk", "compression_ratio_pyvista_zstd"],
    var_name="metric",
    value_name="value",
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=summary_compression, x="raw_MB", y="value", hue="metric", marker="o")
plt.xlabel("Raw dataset size (MB)")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratios vs Dataset Size")
plt.tight_layout()
plt.show()
