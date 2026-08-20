"""Benchmark format-version-3 mesh filters on synthetic and example meshes."""

# ruff: noqa: INP001, S101, T201

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import pyvista as pv
from pyvista import examples
from vtkmodules.util.numpy_support import vtk_to_numpy

import pyvista_zstd as pvzstd
from pyvista_zstd.pyvista_zstd import _FILTER_COMPONENT_SHUFFLE
from pyvista_zstd.pyvista_zstd import _FILTER_NONE
from pyvista_zstd.pyvista_zstd import _FILTER_TRIANGLE_DELTA_SHUFFLE
from pyvista_zstd.pyvista_zstd import _component_shuffle_bytes
from pyvista_zstd.pyvista_zstd import _special_filter_beneficial
from pyvista_zstd.pyvista_zstd import _triangle_delta_shuffle_bytes
from pyvista_zstd.pyvista_zstd import _uncomponent_shuffle_bytes
from pyvista_zstd.pyvista_zstd import _untriangle_delta_shuffle_bytes

if TYPE_CHECKING:
    from collections.abc import Callable

Mode = Literal["off", "force", "auto"]
MAX_FULL_REPEATS_CELLS = 500_000


@dataclass(frozen=True)
class Result:
    """One end-to-end benchmark result."""

    dataset: str
    source: str
    points: int
    triangles: int
    mode: Mode
    file_bytes: int
    write_seconds: float
    read_seconds: float


@dataclass(frozen=True)
class TransformResult:
    """Direct transform and probe costs for one dataset."""

    dataset: str
    point_encode_seconds: float
    point_decode_seconds: float
    point_probe_seconds: float
    connectivity_encode_seconds: float
    connectivity_decode_seconds: float
    connectivity_probe_seconds: float


def _median_seconds(func: Callable[[], object], repeats: int) -> float:
    func()
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = perf_counter()
        func()
        samples.append(perf_counter() - start)
    return median(samples)


def _synthetic_grid(n_side: int, *, scramble: bool = False) -> pv.PolyData:
    """Return a smooth triangulated height field with optional cell scrambling."""
    x, y = np.meshgrid(
        np.linspace(-1, 1, n_side, dtype=np.float32),
        np.linspace(-1, 1, n_side, dtype=np.float32),
        indexing="ij",
    )
    z = (0.1 * np.sin(5 * x) * np.cos(7 * y)).astype(np.float32)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    ids = np.arange(n_side * n_side, dtype=np.int32).reshape(n_side, n_side)
    lower_left = ids[:-1, :-1].ravel()
    lower_right = ids[1:, :-1].ravel()
    upper_left = ids[:-1, 1:].ravel()
    upper_right = ids[1:, 1:].ravel()
    triangles = np.concatenate(
        (
            np.column_stack((lower_left, lower_right, upper_right)),
            np.column_stack((lower_left, upper_right, upper_left)),
        )
    )
    if scramble:
        triangles = triangles[np.random.default_rng(12).permutation(len(triangles))]
    faces = np.empty((len(triangles), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles
    return pv.PolyData(points, faces)


def _synthetic_random(n_side: int) -> pv.PolyData:
    """Return valid but incompressible points and triangle connectivity."""
    rng = np.random.default_rng(23)
    n_points = n_side * n_side
    n_triangles = 2 * (n_side - 1) ** 2
    points = rng.random((n_points, 3), dtype=np.float32)
    triangles = rng.integers(0, n_points, size=(n_triangles, 3), dtype=np.int32)
    faces = np.empty((n_triangles, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles
    return pv.PolyData(points, faces)


def _example_meshes() -> dict[str, pv.PolyData]:
    """Load real triangle meshes published through ``pyvista.examples``."""
    return {
        "bunny": examples.download_bunny(),
        "horse": examples.download_horse(),
        "woman": examples.download_woman(),
        "louis_louvre": examples.download_louis_louvre(),
    }


def _connectivity(mesh: pv.PolyData) -> np.ndarray:
    return vtk_to_numpy(mesh.GetPolys().GetConnectivityArray())


def _transform_benchmark(name: str, mesh: pv.PolyData, repeats: int) -> TransformResult:
    points = np.asarray(mesh.points)
    connectivity = _connectivity(mesh).astype(np.int32, copy=False)
    encoded_points = _component_shuffle_bytes(points)
    encoded_connectivity = _triangle_delta_shuffle_bytes(connectivity)
    result = TransformResult(
        dataset=name,
        point_encode_seconds=_median_seconds(lambda: _component_shuffle_bytes(points), repeats),
        point_decode_seconds=_median_seconds(
            lambda: _uncomponent_shuffle_bytes(encoded_points, points.dtype, points.shape),
            repeats,
        ),
        point_probe_seconds=_median_seconds(
            lambda: _special_filter_beneficial(points, _FILTER_COMPONENT_SHUFFLE, _FILTER_NONE, 3),
            repeats,
        ),
        connectivity_encode_seconds=_median_seconds(
            lambda: _triangle_delta_shuffle_bytes(connectivity),
            repeats,
        ),
        connectivity_decode_seconds=_median_seconds(
            lambda: _untriangle_delta_shuffle_bytes(
                encoded_connectivity,
                connectivity.dtype,
                connectivity.shape,
            ),
            repeats,
        ),
        connectivity_probe_seconds=_median_seconds(
            lambda: _special_filter_beneficial(
                connectivity,
                _FILTER_TRIANGLE_DELTA_SHUFFLE,
                _FILTER_NONE,
                3,
            ),
            repeats,
        ),
    )
    assert _uncomponent_shuffle_bytes(encoded_points, points.dtype, points.shape).tobytes() == points.tobytes()
    assert (
        _untriangle_delta_shuffle_bytes(encoded_connectivity, connectivity.dtype, connectivity.shape).tobytes()
        == connectivity.tobytes()
    )
    return result


def _end_to_end_benchmark(  # noqa: PLR0913
    name: str,
    source: str,
    mesh: pv.PolyData,
    mode: Mode,
    output: Path,
    repeats: int,
) -> Result:
    mesh_filter_setting: bool | Literal["auto"] = {"off": False, "force": True, "auto": "auto"}[mode]
    write_seconds = _median_seconds(
        lambda: pvzstd.write(
            mesh,
            output,
            level=3,
            n_threads=0,
            mesh_filters=mesh_filter_setting,
        ),
        repeats,
    )
    read_seconds = _median_seconds(lambda: pvzstd.read(output, n_threads=0), repeats)
    recovered = pvzstd.read(output, n_threads=0)
    expected_connectivity = _connectivity(mesh).astype(np.int32, copy=False)
    assert recovered.points.tobytes() == mesh.points.tobytes()
    assert _connectivity(recovered).tobytes() == expected_connectivity.tobytes()
    return Result(
        dataset=name,
        source=source,
        points=mesh.n_points,
        triangles=mesh.n_cells,
        mode=mode,
        file_bytes=output.stat().st_size,
        write_seconds=write_seconds,
        read_seconds=read_seconds,
    )


def main() -> None:
    """Run the benchmark and print machine-readable JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-side", type=int, default=700)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repeats = 2 if args.quick else args.repeats
    synthetic_side = 300 if args.quick else args.synthetic_side
    meshes = {
        "synthetic_ordered": ("synthetic", _synthetic_grid(synthetic_side)),
        "synthetic_scrambled": ("synthetic", _synthetic_grid(synthetic_side, scramble=True)),
        "synthetic_random": ("synthetic", _synthetic_random(synthetic_side)),
        **{name: ("pyvista.examples", mesh) for name, mesh in _example_meshes().items()},
    }
    results = []
    transforms = []
    with TemporaryDirectory(prefix="pyvista-zstd-mesh-filter-benchmark-") as tmp:
        tmp_path = Path(tmp)
        for name, (source, mesh) in meshes.items():
            if not mesh.is_all_triangles:
                msg = f"{name} is not an all-triangle mesh"
                raise ValueError(msg)
            dataset_repeats = repeats if mesh.n_cells < MAX_FULL_REPEATS_CELLS else max(2, repeats // 2)
            transforms.append(_transform_benchmark(name, mesh, dataset_repeats))
            results.extend(
                [
                    _end_to_end_benchmark(
                        name,
                        source,
                        mesh,
                        mode,
                        tmp_path / f"{name}-{mode}.pv",
                        dataset_repeats,
                    )
                    for mode in ("off", "force", "auto")
                ]
            )
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "pyvista_version": pv.__version__,
                "vtk_version": pv.vtk_version_info,
                "numpy_version": np.__version__,
                "compression_level": 3,
                "n_threads": 0,
                "repeats": repeats,
                "synthetic_side": synthetic_side,
                "results": [asdict(result) for result in results],
                "transforms": [asdict(result) for result in transforms],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
