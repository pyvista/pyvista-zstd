cmake_minimum_required(VERSION 3.16)
project(pvzstd LANGUAGES CXX)

option(PVZSTD_BUILD_TOOLS "Build the pvz_dump conformance tool" ON)
option(PVZSTD_BUILD_SHARED "Build a shared library (needed for the ctypes binding)" ON)

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

find_package(zstd CONFIG QUIET)
if(NOT zstd_FOUND)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
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
  if(PVZSTD_ZSTD_INCLUDE_DIR AND PVZSTD_ZSTD_LIBRARY)
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
    # silently emitting single-threaded frames that would not match.
    set(ZSTD_MULTITHREAD_SUPPORT ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zstd_vendored)
    set(PVZSTD_ZSTD_TARGET libzstd_static)
  else()
    message(FATAL_ERROR
      "No zstd found and PVZSTD_VENDOR_ZSTD is OFF. Install zstd development "
      "files or re-run with -DPVZSTD_VENDOR_ZSTD=ON.")
  endif()
endif()
message(STATUS "pvzstd: linking zstd via ${PVZSTD_ZSTD_TARGET}")

if(PVZSTD_BUILD_SHARED)
  add_library(pvzstd SHARED src/reader.cpp src/writer.cpp)
  target_compile_definitions(pvzstd PRIVATE PVZSTD_BUILDING PUBLIC PVZSTD_SHARED)
else()
  add_library(pvzstd STATIC src/reader.cpp src/writer.cpp)
endif()

target_include_directories(pvzstd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(pvzstd PRIVATE ${PVZSTD_ZSTD_TARGET})

if(MSVC)
  target_compile_options(pvzstd PRIVATE /W4)
else()
  target_compile_options(pvzstd PRIVATE -Wall -Wextra)
endif()

# The writer sets ZSTD_c_nbWorkers, so the library needs a thread runtime.
find_package(Threads REQUIRED)
target_link_libraries(pvzstd PRIVATE Threads::Threads)

if(PVZSTD_BUILD_TOOLS)
  add_executable(pvz_dump tools/pvz_dump.cpp)
  target_link_libraries(pvz_dump PRIVATE pvzstd)
  add_executable(pvz_rewrite tools/pvz_rewrite.cpp)
  target_link_libraries(pvz_rewrite PRIVATE pvzstd)
endif()

include(GNUInstallDirs)
install(TARGETS pvzstd
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
