"""
Build the pvzstd C++ core and bundle it into the wheel.

The library is the package, so the build is required rather than best-effort.
``PVZSTD_BUILD_CORE``:

  unset  required -- a C++ build failure is a build failure
  1      required -- same; CI and cibuildwheel set it explicitly
  0      skipped  -- never invoke CMake. The install is unusable until a
                     library arrives from elsewhere (``PVZSTD_LIBRARY``, or
                     one dropped into the package's ``lib/``). For docs builds
                     and CI jobs testing a separate CMake build; never for a
                     wheel anyone installs.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from setuptools import setup
from setuptools.dist import Distribution

HERE = Path(__file__).parent.resolve()
CPP_DIR = HERE / "cpp"
PACKAGE_LIB_DIR = HERE / "src" / "pyvista_zstd" / "lib"

_SKIPPED = os.environ.get("PVZSTD_BUILD_CORE", "").strip() == "0"


def _library_names() -> list[str]:
    """
    Shared-library file names CMake may produce, per platform.

    Kept in step with ``pyvista_zstd._capi._candidate_names``; the loader looks
    for these exact names inside the installed package.
    """
    if sys.platform == "win32":
        return ["pvzstd.dll", "libpvzstd.dll"]
    if sys.platform == "darwin":
        return ["libpvzstd.dylib"]
    return ["libpvzstd.so"]


def _find_built_library(build_dir: Path) -> Path | None:
    """
    Locate the shared library inside a finished CMake build tree.

    Searched recursively because multi-config generators (Visual Studio) place
    the artefact under a per-configuration subdirectory, while single-config
    generators put it at the top.
    """
    for name in _library_names():
        matches = sorted(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _macos_architectures() -> str:
    """
    Return the macOS architectures to build for, or an empty string.

    ``CMAKE_OSX_ARCHITECTURES`` wins when set; otherwise the value comes from
    ``ARCHFLAGS``, which is how wheel builders request a cross-build. Ignoring
    it puts a host-architecture library inside a wheel labelled for the other.
    """
    explicit = os.environ.get("CMAKE_OSX_ARCHITECTURES", "").strip()
    if explicit:
        return explicit
    if sys.platform != "darwin":
        return ""
    flags = os.environ.get("ARCHFLAGS", "").split()
    found = [flags[i + 1] for i, flag in enumerate(flags) if flag == "-arch" and i + 1 < len(flags)]
    return ";".join(dict.fromkeys(found))


def _build_cpp() -> Path | None:
    """Configure, build, and return the path to the bundled shared library."""
    if _SKIPPED:
        print("pvzstd: PVZSTD_BUILD_CORE=0, skipping the C++ build")
        return None
    if not (CPP_DIR / "CMakeLists.txt").is_file():
        # An sdist that omitted cpp/, or a source tree in an odd state.
        msg = f"no CMake project at {CPP_DIR}; set PVZSTD_BUILD_CORE=0 to install without a core"
        raise SystemExit(msg)

    build_dir = Path(tempfile.mkdtemp(prefix="pvzstd-build-"))
    configure = [
        "cmake",
        "-S",
        str(CPP_DIR),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPVZSTD_BUILD_SHARED=ON",
        # Developer utilities; building them only lengthens every wheel build.
        "-DPVZSTD_BUILD_TOOLS=OFF",
        # PROVIDER, not just VENDOR_ZSTD: the latter falls back to the pinned
        # source only if nothing is installed, so a build machine with zstd links
        # that one -- fatal on a macOS cross-build, where the host's is arm64.
        "-DPVZSTD_ZSTD_PROVIDER=vendored",
        "-DPVZSTD_VENDOR_ZSTD=ON",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]
    if sys.platform == "darwin" and (target := os.environ.get("MACOSX_DEPLOYMENT_TARGET")):
        configure.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}")
    if archs := _macos_architectures():
        configure.append(f"-DCMAKE_OSX_ARCHITECTURES={archs}")

    try:
        subprocess.run(configure, check=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        msg = f"pvzstd: the C++ core failed to build, and it is what the package is: {exc}"
        raise SystemExit(msg) from exc

    built = _find_built_library(build_dir)
    if built is None:
        msg = f"pvzstd: build produced no shared library in {build_dir}"
        raise SystemExit(msg)

    PACKAGE_LIB_DIR.mkdir(parents=True, exist_ok=True)
    destination = PACKAGE_LIB_DIR / built.name
    shutil.copy2(built, destination)
    print(f"pvzstd: bundled {built} -> {destination}")
    shutil.rmtree(build_dir, ignore_errors=True)
    return destination


# Commands that put files into a wheel or an installed tree. Metadata-only ones
# are absent: compiling zstd to answer a version question would be waste.
_BUILDING_COMMANDS = frozenset(
    {"bdist_wheel", "build", "build_py", "editable_wheel", "install"},
)


def _is_a_build() -> bool:
    """Whether this invocation is going to produce installable artefacts."""
    return any(arg in _BUILDING_COMMANDS for arg in sys.argv[1:])


# At import time, not from build_py: bdist_wheel decides the platform tag in
# finalize_options, before any build command. Deferring it produced a wheel with a
# Linux shared object tagged py3-none-any.
_bundled: Path | None = _build_cpp() if _is_a_build() else None


class CoreDistribution(Distribution):
    """
    Tag the wheel for this platform once a compiled library is inside it.

    Without this, setuptools sees no ``ext_modules`` and stamps the wheel
    ``py3-none-any`` -- which would advertise a Linux shared object as valid
    on Windows.
    """

    def has_ext_modules(self) -> bool:
        """Report whether a compiled artefact is going into the wheel."""
        return _bundled is not None


_cmdclass: dict[str, type] = {}

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - setuptools older than 70.1
    try:
        from wheel.bdist_wheel import bdist_wheel  # type: ignore[no-redef]
    except ImportError:
        bdist_wheel = None  # type: ignore[assignment]

if bdist_wheel is not None:

    class PlatformNotInterpreterWheel(bdist_wheel):
        """
        Tag the wheel by platform only, never by interpreter.

        ``has_ext_modules`` makes setuptools reach for the full
        ``cp312-cp312-linux_x86_64`` tag. This library is a plain C shared
        object loaded through ctypes with no CPython ABI reference, so one
        build serves every supported interpreter.
        """

        def get_tag(self) -> tuple[str, str, str]:
            """Return the wheel tag, widened to every Python 3 interpreter."""
            python, abi, platform = super().get_tag()
            if _bundled is not None:
                return "py3", "none", platform
            return python, abi, platform

    _cmdclass["bdist_wheel"] = PlatformNotInterpreterWheel


setup(
    distclass=CoreDistribution,
    cmdclass=_cmdclass,
    package_data={"pyvista_zstd": ["lib/*"]},
)
