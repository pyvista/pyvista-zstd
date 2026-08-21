cmake_minimum_required(VERSION 3.16)
project(pvzstd LANGUAGES CXX)

option(PVZSTD_BUILD_TOOLS "Build the pvzstd_dump conformance tool" ON)
option(PVZSTD_BUILD_SHARED "Build a shared library (needed for the ctypes binding)" ON)

# For targets with no thread runtime, where the alternative is not building.
# Reads are unaffected (same bytes, one core). Writes lose byte-identity above
# the reference rule's 2 MiB threshold: a zstd without ZSTD_MULTITHREAD
# rejects a non-zero ZSTD_c_nbWorkers, so pvzstd_writer_write returns
# PVZSTD_E_ZSTD rather than emitting frames that would not match.
option(PVZSTD_THREADS "Use threads for batch decompression and zstd workers" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hide everything except what PVZSTD_API marks, so the shared object exports
# exactly the documented C ABI and nothing else.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# --- zstd -------------------------------------------------------------
#
# Look for an installed zstd first, so a distribution build links the one the
# system already ships and gets its security updates. Fall back to building a
# pinned copy from source, which is what makes the Windows and WebAssembly
# builds possible at all -- neither has a system zstd to find.
option(PVZSTD_VENDOR_ZSTD "Build a pinned zstd from source if none is installed" ON)

# auto      prefer an installed zstd, fall back to the pinned source build
# vendored  always build the pinned source, ignoring anything installed
# system    require an installed zstd
#
# "vendored" exists because byte-for-byte reproduction of the reference
# writer's output depends on the zstd build, not only on this writer's logic:
# different zstd versions emit different (equally valid) compressed frames for
# the same input. The conformance suite compares against a reference writer
# carrying its own copy of zstd, so it has to pin the same source to compare
# like with like. Measured: linking a distribution's system zstd instead of the
# pinned one made the byte-identity tests fail on multi-frame data while the
# same commit passed on platforms that had no system zstd to find.
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
  # Nothing installed. Try the plain library/header pair before downloading,
  # so a toolchain that carries zstd without a CMake or pkg-config file (the
  # usual case for cross-compilers) still links against what it already has.
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
    # Pinned by hash, not by tag: a tag can be moved, and these bytes decide
    # what every wheel we publish contains.
    FetchContent_Declare(zstd_vendored
      URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
      URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
      SOURCE_SUBDIR build/cmake)
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
    # Load-bearing, not a preference. The writer sets ZSTD_c_nbWorkers to
    # reproduce the reference writer's bytes; a zstd built without threads
    # rejects that parameter, and the writer reports the failure rather than
    # silently emitting single-threaded frames that would not match. It follows
    # PVZSTD_THREADS so the two answers cannot disagree.
    set(ZSTD_MULTITHREAD_SUPPORT ${PVZSTD_THREADS} CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zstd_vendored)
    set(PVZSTD_ZSTD_TARGET libzstd_static)

    # zstd's assembly Huffman decoder is on by default upstream; on this
    # project's reference machine it is slower than zstd's own C path, so it
    # is off here. Measured on an i9-14900KF, taskset-pinned, output
    # byte-identical either way -- decode kernel 1.166x slower on a P-core and
    # 0.919x faster on an E-core; end to end through Reader.read() this build
    # is 1.03-1.05x faster on P-cores and neutral on E-cores.
    #
    # Two microarchitectures on one machine, so this is a default rather than
    # a claim about every CPU: set it ON for zstd's own. Only reaches the
    # vendored build; a system zstd was compiled before we got here.
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

target_include_directories(pvzstd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(pvzstd PRIVATE ${PVZSTD_ZSTD_TARGET})

if(MSVC)
  target_compile_options(pvzstd PRIVATE /W4)
else()
  # A container can be larger than 2 GiB, and the append and stream paths seek
  # around it. Without this a 32-bit libc gives off_t 32 bits and fseeko is no
  # wider than fseek. See cpp/src/fileio.h.
  target_compile_definitions(pvzstd PRIVATE _FILE_OFFSET_BITS=64)
  target_compile_options(pvzstd PRIVATE -Wall -Wextra)
endif()

if(PVZSTD_THREADS)
  # The batch reader spreads frames over std::thread workers, and the writer
  # sets ZSTD_c_nbWorkers; both want a thread runtime.
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
install(TARGETS pvzstd
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
