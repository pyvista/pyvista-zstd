cmake_minimum_required(VERSION 3.16)
# Tracks the released version line, so `find_package(pvzstd 0.3)` means what a
# reader expects. It cannot be derived: the Python distribution's version comes
# from git tags through setuptools_scm and has no value at configure time. Bump
# it when the release line moves. The real compatibility contract for a C or C++
# consumer is PVZSTD_ABI_VERSION in the public header, which is checked for
# equality rather than as a floor.
project(pvzstd VERSION 0.3.1 LANGUAGES CXX)

option(PVZSTD_BUILD_TOOLS "Build the pvzstd_dump conformance tool" ON)
option(PVZSTD_BUILD_SHARED "Build a shared library (needed for the ctypes binding)" ON)

# For targets with no thread runtime. Reads are unaffected; writes lose
# byte-identity above the reference rule's 2 MiB threshold, reported as
# PVZSTD_E_ZSTD rather than emitted as frames that would not match.
option(PVZSTD_THREADS "Use threads for batch decompression and zstd workers" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export exactly what PVZSTD_API marks and nothing else.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# --- zstd -------------------------------------------------------------
#
# An installed zstd first, so a distribution build gets its security updates;
# otherwise a pinned source build, which is what makes Windows and WebAssembly
# possible at all -- neither has a system zstd to find.
option(PVZSTD_VENDOR_ZSTD "Build a pinned zstd from source if none is installed" ON)

# auto      prefer an installed zstd, fall back to the pinned source build
# vendored  always build the pinned source, ignoring anything installed
# system    require an installed zstd
#
# "vendored" exists because different zstd versions emit different (equally valid)
# frames for the same input, so the conformance suite has to pin the same source
# the reference writer carries. Measured: linking a system zstd instead failed the
# byte-identity tests on multi-frame data where platforms without one passed.
set(PVZSTD_ZSTD_PROVIDER "auto" CACHE STRING "Where zstd comes from: auto, vendored, or system")
set_property(CACHE PVZSTD_ZSTD_PROVIDER PROPERTY STRINGS auto vendored system)

if(NOT PVZSTD_ZSTD_PROVIDER STREQUAL "vendored")
  find_package(zstd CONFIG QUIET)
  if(NOT zstd_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
    endif()
  endif()
endif()

if(TARGET zstd::libzstd_shared)
  set(PVZSTD_ZSTD_TARGET zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
  set(PVZSTD_ZSTD_TARGET zstd::libzstd_static)
elseif(TARGET PkgConfig::ZSTD)
  set(PVZSTD_ZSTD_TARGET PkgConfig::ZSTD)
else()
  # Try the plain library/header pair before downloading: cross-toolchains often
  # carry zstd without a CMake or pkg-config file.
  find_path(PVZSTD_ZSTD_INCLUDE_DIR zstd.h)
  find_library(PVZSTD_ZSTD_LIBRARY NAMES zstd libzstd zstd_static)
  if(PVZSTD_ZSTD_PROVIDER STREQUAL "system")
    message(FATAL_ERROR
      "PVZSTD_ZSTD_PROVIDER=system but no installed zstd was found.")
  endif()
  if(PVZSTD_ZSTD_INCLUDE_DIR AND PVZSTD_ZSTD_LIBRARY AND NOT PVZSTD_ZSTD_PROVIDER STREQUAL "vendored")
    add_library(pvzstd_zstd_found UNKNOWN IMPORTED)
    set_target_properties(pvzstd_zstd_found PROPERTIES
      IMPORTED_LOCATION "${PVZSTD_ZSTD_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${PVZSTD_ZSTD_INCLUDE_DIR}")
    set(PVZSTD_ZSTD_TARGET pvzstd_zstd_found)
  elseif(PVZSTD_VENDOR_ZSTD)
    include(FetchContent)
    # By hash, not tag: a tag can be moved, and these bytes decide what we ship.
    FetchContent_Declare(zstd_vendored
      URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
      URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
      SOURCE_SUBDIR build/cmake)
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
    # Load-bearing: the writer sets ZSTD_c_nbWorkers to reproduce the reference
    # bytes, and a zstd built without threads rejects that parameter. Follows
    # PVZSTD_THREADS so the two answers cannot disagree.
    set(ZSTD_MULTITHREAD_SUPPORT ${PVZSTD_THREADS} CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zstd_vendored)
    set(PVZSTD_ZSTD_TARGET libzstd_static)

    # Upstream default is ON; measured slower than zstd's own C path here.
    # Pinned, byte-identical output: decode kernel 1.166x slower on a P-core and
    # 0.919x faster on an E-core, 1.03-1.05x faster end to end on P-cores. Two
    # microarchitectures on one machine, so this is a default and not a claim
    # about every CPU -- set it ON for zstd's own.
    option(PVZSTD_ZSTD_ASM "Use zstd's assembly Huffman decoder" OFF)
    if(NOT PVZSTD_ZSTD_ASM)
      target_compile_definitions(libzstd_static PRIVATE ZSTD_DISABLE_ASM=1)
    endif()
    message(STATUS "pvzstd: zstd assembly decoder ${PVZSTD_ZSTD_ASM}")
  else()
    message(FATAL_ERROR
      "No zstd found and PVZSTD_VENDOR_ZSTD is OFF. Install zstd development "
      "files or re-run with -DPVZSTD_VENDOR_ZSTD=ON.")
  endif()
endif()
message(STATUS "pvzstd: linking zstd via ${PVZSTD_ZSTD_TARGET}")

if(PVZSTD_BUILD_SHARED)
  add_library(pvzstd SHARED src/reader.cpp src/writer.cpp src/append.cpp src/stream.cpp)
  target_compile_definitions(pvzstd PRIVATE PVZSTD_BUILDING PUBLIC PVZSTD_SHARED)
else()
  add_library(pvzstd STATIC src/reader.cpp src/writer.cpp src/append.cpp src/stream.cpp)
endif()

# Namespaced alias so a submodule / FetchContent consumer links the same
# target name (pvzstd::pvzstd) that find_package(pvzstd) yields below.
add_library(pvzstd::pvzstd ALIAS pvzstd)

target_include_directories(pvzstd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(pvzstd PRIVATE ${PVZSTD_ZSTD_TARGET})

if(MSVC)
  target_compile_options(pvzstd PRIVATE /W4)
else()
  # Containers exceed 2 GiB; without this a 32-bit libc gives off_t 32 bits and
  # fseeko is no wider than fseek. See cpp/src/fileio.h.
  target_compile_definitions(pvzstd PRIVATE _FILE_OFFSET_BITS=64)
  target_compile_options(pvzstd PRIVATE -Wall -Wextra)
endif()

# --- Exceptions on WebAssembly ------------------------------------------
#
# Every extern "C" entry point is a function-try-block ending in catch(...),
# which is how the header can promise a status code rather than an exception
# unwinding into a C caller. Emscripten defaults to
# -sDISABLE_EXCEPTION_CATCHING=1, which compiles every one of those handlers
# away and turns a throw into abort() -- so on that target the promise the
# header makes is not kept, and a caller parsing untrusted input gets a dead
# process instead of PVZSTD_E_NOMEM.
#
# -fwasm-exceptions rather than -sNO_DISABLE_EXCEPTION_CATCHING: the latter is
# the legacy JavaScript-based unwinder, which is slower, larger, and taxes
# every call site whether or not anything throws, while the former lowers to
# the WebAssembly exception-handling instructions that every engine emsdk
# 4.0.20 targets already implements. It is needed at compile and at link.
#
# PUBLIC, not PRIVATE: a wasm link cannot mix exception models, so a consumer
# compiling its own translation units against this library and then linking it
# has to be on the same one. Scoped to the target rather than set globally, so
# a project that embeds this library keeps its own flags everywhere else.
if(EMSCRIPTEN)
  target_compile_options(pvzstd PUBLIC -fwasm-exceptions)
  target_link_options(pvzstd PUBLIC -fwasm-exceptions)
endif()

if(PVZSTD_THREADS)
  find_package(Threads REQUIRED)
  target_link_libraries(pvzstd PRIVATE Threads::Threads)
else()
  target_compile_definitions(pvzstd PRIVATE PVZSTD_NO_THREADS)
endif()

if(PVZSTD_BUILD_TOOLS)
  add_executable(pvzstd_dump tools/pvzstd_dump.cpp)
  target_link_libraries(pvzstd_dump PRIVATE pvzstd)
  add_executable(pvzstd_rewrite tools/pvzstd_rewrite.cpp)
  target_link_libraries(pvzstd_rewrite PRIVATE pvzstd)
  add_executable(pvzstd_append tools/pvzstd_append.cpp)
  target_link_libraries(pvzstd_append PRIVATE pvzstd)

  add_executable(pvzstd_stream tools/pvzstd_stream.cpp)
  target_link_libraries(pvzstd_stream PRIVATE pvzstd)
endif()

include(GNUInstallDirs)

# --- Install / export ---------------------------------------------------
#
# zstd is linked PRIVATE (see target_link_libraries(pvzstd ...) above), and
# the public headers are a pure C ABI that never names a zstd type, so zstd
# is purely an implementation detail here, not something a consumer needs
# to find. That holds cleanly for the default shared build
# (PVZSTD_BUILD_SHARED=ON): a PRIVATE link into a shared library folds
# zstd's object code into libpvzstd itself, so the installed library is
# self-contained regardless of whether zstd was vendored or found on the
# system.
#
# It does NOT hold for a static build (PVZSTD_BUILD_SHARED=OFF): CMake does
# not fold a PRIVATE static link dependency's object code into the
# resulting archive, so libpvzstd.a still needs zstd's symbols resolved at
# the downstream link step, and this export does not declare that
# dependency for a consumer -- zstd (vendored or system) is treated purely
# as this build's own implementation detail, not a re-exported one. With a
# system zstd a consumer can supply that themselves. With
# PVZSTD_ZSTD_PROVIDER=vendored, zstd is pulled in via FetchContent as an
# internal build-tree dependency; its own CMakeLists happens to install its
# own findable "zstd" package alongside ours when install() runs (a side
# effect of FetchContent adding it as a subdirectory, not something this
# project arranges or guarantees), so it is not strictly unfindable, but
# relying on that is fragile and not something to design around. Prefer the
# default shared build for anything that will be installed and linked
# downstream.
set(PVZSTD_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/pvzstd")

install(TARGETS pvzstd
        EXPORT pvzstdTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT pvzstdTargets
        FILE pvzstdTargets.cmake
        NAMESPACE pvzstd::
        DESTINATION ${PVZSTD_INSTALL_CMAKEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pvzstdConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/pvzstdConfig.cmake"
  INSTALL_DESTINATION "${PVZSTD_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/pvzstdConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/pvzstdConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/pvzstdConfigVersion.cmake"
  DESTINATION "${PVZSTD_INSTALL_CMAKEDIR}")
