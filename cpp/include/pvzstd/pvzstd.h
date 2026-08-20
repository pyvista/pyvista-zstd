/* pvzstd -- C ABI for the .pv / .zvtk trailer-indexed zstd container.
 *
 * This header is the whole public surface. It is plain C89-compatible C with
 * no C++ types crossing the boundary, so it binds from ctypes, from another
 * C++ project consuming this as a submodule, or from WASM, without a Python
 * extension module anywhere in the picture.
 *
 * The on-disk format is specified in doc/format/container-v2.md. Two rules a
 * caller should know:
 *
 *   - Arrays are addressed by index in frame order, or looked up by name.
 *     Frame order is significant: it is how the container maps names to
 *     frames.
 *   - Opening a file parses only the trailer index and the per-array headers.
 *     Payloads are decompressed on demand, so opening a large container and
 *     reading one array does not pay for the rest.
 */

#ifndef PVZSTD_PVZSTD_H
#define PVZSTD_PVZSTD_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(PVZSTD_SHARED)
#  if defined(PVZSTD_BUILDING)
#    define PVZSTD_API __declspec(dllexport)
#  else
#    define PVZSTD_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define PVZSTD_API __attribute__((visibility("default")))
#else
#  define PVZSTD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped only on an incompatible change to the declarations below. A caller
 * that binds this ABI dynamically should check it before anything else. */
#define PVZSTD_ABI_VERSION 1u

/* Width of the dtype field in an array header, and of the dataset-UID name
 * prefix. These coincide in the format and the coincidence is load-bearing. */
#define PVZSTD_DTYPE_LEN 16

typedef enum pvz_status {
  PVZ_OK = 0,
  PVZ_E_IO = 1,          /* file missing, unreadable, or truncated */
  PVZ_E_FORMAT = 2,      /* trailer or header did not parse */
  PVZ_E_ZSTD = 3,        /* a frame failed to decompress */
  PVZ_E_RANGE = 4,       /* index out of range, or destination too small */
  PVZ_E_NOMEM = 5,       /* allocation failed */
  PVZ_E_FILTER = 6,      /* per-array filter id this build cannot reverse */
  PVZ_E_INVALID = 7      /* NULL argument or misuse */
} pvz_status;

/* Per-array filter ids. An unknown id is an error, never a passthrough:
 * returning filtered bytes as-is would silently corrupt the array. */
#define PVZ_FILTER_NONE 0
#define PVZ_FILTER_SHUFFLE 1

typedef struct pvz_reader pvz_reader;

/* A view onto one array's header. All pointers are owned by the reader and
 * remain valid until pvz_close(); do not free them. */
typedef struct pvz_array_info {
  const char *name;        /* NUL-terminated, UTF-8, includes the UID prefix */
  const uint64_t *shape;   /* ndim entries; NULL when ndim == 0 */
  uint32_t ndim;
  uint8_t filter_id;
  char dtype[PVZSTD_DTYPE_LEN + 1]; /* e.g. "<f8", "|u1"; NUL-terminated */
  uint64_t nbytes;         /* decompressed payload size */
} pvz_array_info;

/* Open a container. On success *out receives a reader that must be released
 * with pvz_close(). On failure *out is set to NULL. */
PVZSTD_API pvz_status pvz_open(const char *path, pvz_reader **out);

/* Release a reader. Safe to call with NULL. */
PVZSTD_API void pvz_close(pvz_reader *reader);

/* Number of arrays, excluding the two JSON metadata frames. */
PVZSTD_API uint64_t pvz_array_count(const pvz_reader *reader);

/* Describe array `index`. */
PVZSTD_API pvz_status pvz_array_info_at(const pvz_reader *reader, uint64_t index,
                                        pvz_array_info *out);

/* Index of the array called `name`, or -1 if there is none. */
PVZSTD_API int64_t pvz_find_array(const pvz_reader *reader, const char *name);

/* Decompress array `index` into `dst`, reversing any filter. `dst_size` must
 * be at least the `nbytes` reported by pvz_array_info_at, or PVZ_E_RANGE is
 * returned and nothing is written. */
PVZSTD_API pvz_status pvz_read_array_at(const pvz_reader *reader, uint64_t index,
                                        void *dst, uint64_t dst_size);

/* The two JSON metadata documents, NUL-terminated and owned by the reader.
 * Either may be NULL if the container did not carry it. */
PVZSTD_API const char *pvz_ds_metadata_json(const pvz_reader *reader);
PVZSTD_API const char *pvz_file_metadata_json(const pvz_reader *reader);

/* A static, human-readable string for a status code. Never NULL. */
PVZSTD_API const char *pvz_status_message(pvz_status status);

/* The ABI version this library was built with. */
PVZSTD_API uint32_t pvz_abi_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PVZSTD_PVZSTD_H */
