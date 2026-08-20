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

find_package(zstd CONFIG QUIET)
if(NOT zstd_FOUND)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD IMPORTED_TARGET libzstd)
  endif()
endif()

if(PVZSTD_BUILD_SHARED)
  add_library(pvzstd SHARED src/reader.cpp)
  target_compile_definitions(pvzstd PRIVATE PVZSTD_BUILDING PUBLIC PVZSTD_SHARED)
else()
  add_library(pvzstd STATIC src/reader.cpp)
endif()

target_include_directories(pvzstd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(TARGET zstd::libzstd_shared)
  target_link_libraries(pvzstd PRIVATE zstd::libzstd_shared)
elseif(TARGET zstd::libzstd_static)
  target_link_libraries(pvzstd PRIVATE zstd::libzstd_static)
elseif(TARGET PkgConfig::ZSTD)
  target_link_libraries(pvzstd PRIVATE PkgConfig::ZSTD)
else()
  # Last resort: the bare library name. Emscripten resolves this through its
  # own ports, and a plain Unix toolchain finds it on the default link path.
  target_link_libraries(pvzstd PRIVATE zstd)
endif()

if(MSVC)
  target_compile_options(pvzstd PRIVATE /W4)
else()
  target_compile_options(pvzstd PRIVATE -Wall -Wextra)
endif()

if(PVZSTD_BUILD_TOOLS)
  add_executable(pvz_dump tools/pvz_dump.cpp)
  target_link_libraries(pvz_dump PRIVATE pvzstd)
endif()

include(GNUInstallDirs)
install(TARGETS pvzstd
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
