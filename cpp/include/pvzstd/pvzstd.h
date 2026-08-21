/* pvzstd -- C ABI for the .pv / .zvtk trailer-indexed zstd container.
 *
 * The whole public surface, in C89-compatible C. Format: doc/format/container-v2.md.
 *
 * Arrays are addressed by index in frame order, or by name. Frame order is
 * significant: it is how the container maps names to frames. Opening parses only
 * the trailer and array headers; payloads are decompressed on demand.
 */

#ifndef PVZSTD_PVZSTD_H
#define PVZSTD_PVZSTD_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(PVZSTD_SHARED)
#if defined(PVZSTD_BUILDING)
#define PVZSTD_API __declspec(dllexport)
#else
#define PVZSTD_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define PVZSTD_API __attribute__((visibility("default")))
#else
#define PVZSTD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any change below, additions included -- callers check equality, not
 * a floor, because they bind every symbol up front. */
#define PVZSTD_ABI_VERSION 2u

/* Dtype-field width and dataset-UID prefix width. They coincide in the format. */
#define PVZSTD_DTYPE_LEN 16

typedef enum pvzstd_status {
  PVZSTD_OK = 0,
  PVZSTD_E_IO = 1,     /* file missing, unreadable, or truncated */
  PVZSTD_E_FORMAT = 2, /* trailer or header did not parse */
  PVZSTD_E_ZSTD = 3,   /* zstd rejected a frame or a compression parameter */
  PVZSTD_E_RANGE = 4,  /* index or count out of range, or destination too small */
  PVZSTD_E_NOMEM = 5,  /* allocation failed */
  PVZSTD_E_FILTER = 6, /* per-array filter id this build cannot reverse */
  PVZSTD_E_INVALID = 7 /* NULL argument or misuse */
} pvzstd_status;

/* Per-array filter ids. An unknown id is an error, never a passthrough. */
#define PVZSTD_FILTER_NONE 0
#define PVZSTD_FILTER_SHUFFLE 1

typedef struct pvzstd_reader pvzstd_reader;

/* One array's header. Pointers are owned by the reader until pvzstd_close(). */
typedef struct pvzstd_array_info {
  const char *name;      /* NUL-terminated, UTF-8, includes the UID prefix */
  const uint64_t *shape; /* ndim entries; NULL when ndim == 0 */
  uint32_t ndim;
  uint8_t filter_id;
  char dtype[PVZSTD_DTYPE_LEN + 1]; /* e.g. "<f8", "|u1"; NUL-terminated */
  uint64_t nbytes;                  /* decompressed payload size */
} pvzstd_array_info;

/* Open a container. On success *out receives a reader that must be released
 * with pvzstd_close(). On failure *out is set to NULL. */
PVZSTD_API pvzstd_status pvzstd_open(const char *path, pvzstd_reader **out);

/* Release a reader. Safe to call with NULL. */
PVZSTD_API void pvzstd_close(pvzstd_reader *reader);

/* Number of arrays, excluding the two JSON metadata frames. */
PVZSTD_API uint64_t pvzstd_array_count(const pvzstd_reader *reader);

/* Describe array `index`. */
PVZSTD_API pvzstd_status pvzstd_array_info_at(const pvzstd_reader *reader, uint64_t index,
                                              pvzstd_array_info *out);

/* Describe `count` arrays from `first` in one boundary crossing rather than one
 * per array. PVZSTD_E_RANGE (writing nothing) if it runs past the end; a count of
 * zero succeeds. */
PVZSTD_API pvzstd_status pvzstd_array_info_range(const pvzstd_reader *reader, uint64_t first,
                                                 uint64_t count, pvzstd_array_info *out);

/* Index of the array called `name`, or -1 if there is none. */
PVZSTD_API int64_t pvzstd_find_array(const pvzstd_reader *reader, const char *name);

/* Decompress array `index` into `dst`, reversing any filter. `dst_size` must
 * be at least the `nbytes` reported by pvzstd_array_info_at, or PVZSTD_E_RANGE is
 * returned and nothing is written. */
PVZSTD_API pvzstd_status pvzstd_read_array_at(const pvzstd_reader *reader, uint64_t index,
                                              void *dst, uint64_t dst_size);

/* Decompress `count` arrays over `n_threads` workers; `indices[i]` goes to
 * `dsts[i]`, at least `dst_sizes[i]` bytes. Frames are independent, so this is a
 * pure fan-out. `n_threads` <= 1 runs inline; PVZSTD_THREADS_AUTO uses hardware
 * concurrency capped at `count`. Returns the first non-OK status. */
PVZSTD_API pvzstd_status pvzstd_read_arrays(const pvzstd_reader *reader, const uint64_t *indices,
                                            uint64_t count, void *const *dsts,
                                            const uint64_t *dst_sizes, int n_threads);

/* ---- Field arrays ----
 *
 * The blocks an append adds, named without the dataset-UID prefix or the
 * "__field_data" suffix the frame carries. The set and its order come from the
 * dataset metadata; scanning frame names for the suffix would also match an
 * ordinary array ending that way. MultiBlock containers report zero. */

/* Number of field arrays on the root dataset. */
PVZSTD_API uint64_t pvzstd_field_array_count(const pvzstd_reader *reader);

/* Bare name of field array `index`, owned by the reader, or NULL. */
PVZSTD_API const char *pvzstd_field_array_name_at(const pvzstd_reader *reader, uint64_t index);

/* Array index of field array `name`, or -1 -- including when the metadata names
 * it but the frame is absent, which is a damaged file. */
PVZSTD_API int64_t pvzstd_find_field_array(const pvzstd_reader *reader, const char *name);

/* The two JSON metadata documents, NUL-terminated and owned by the reader.
 * Either may be NULL if the container did not carry it. */
PVZSTD_API const char *pvzstd_ds_metadata_json(const pvzstd_reader *reader);
PVZSTD_API const char *pvzstd_file_metadata_json(const pvzstd_reader *reader);

/* Every entry point below reports failure as a status code and never lets an
 * exception cross this boundary: a caller reaching it through ctypes or another
 * language has no way to catch one. Allocation failure surfaces as
 * PVZSTD_E_NOMEM; the accessors returning a pointer or a count report it as NULL
 * or as zero, which are the values they already use for "cannot answer". */

/* A static, human-readable string for a status code. Never NULL. */
PVZSTD_API const char *pvzstd_status_message(pvzstd_status status);

/* The ABI version this library was built with. */
PVZSTD_API uint32_t pvzstd_abi_version(void);

/* ------------------------------------------------------------------ *
 * Writer
 *
 * The container layer only: it takes arrays the caller already produced and
 * emits the file. Turning a dataset into arrays stays with the caller, which is
 * what keeps VTK out of this library and lets it build as WASM.
 *
 * Reproducing the reference writer byte for byte needs the same compression
 * level *and* worker count -- zstd's threaded mode emits different (equally
 * valid) bytes. See pvzstd_writer_set_threads.
 * ------------------------------------------------------------------ */

typedef struct pvzstd_writer pvzstd_writer;

/* Mirrors the reference writer's ``shuffle=False | True | "auto"``. */
typedef enum pvzstd_shuffle_mode {
  PVZSTD_SHUFFLE_NEVER = 0,  /* default */
  PVZSTD_SHUFFLE_ALWAYS = 1, /* still skipped for itemsize <= 1 */
  PVZSTD_SHUFFLE_AUTO = 2    /* float/complex only, and only if a trial compress shrinks */
} pvzstd_shuffle_mode;

/* Derive the worker count from payload size, as the reference writer does. */
#define PVZSTD_THREADS_AUTO (-2)

PVZSTD_API pvzstd_status pvzstd_writer_create(pvzstd_writer **out);
PVZSTD_API void pvzstd_writer_free(pvzstd_writer *writer);

/* Compression level; default 3, matching the reference writer. */
PVZSTD_API pvzstd_status pvzstd_writer_set_level(pvzstd_writer *writer, int level);

/* Worker count per frame. PVZSTD_THREADS_AUTO (default) reproduces the reference
 * rule: floor(total_MiB / 2), or -1 once that exceeds 8. 0 is single-threaded;
 * negative is one worker per logical CPU -- so such a file only reproduces on a
 * machine with the same CPU count, which is a property of the existing format.
 *
 * A zstd built without multithreading rejects any non-zero count, so
 * pvzstd_writer_write returns PVZSTD_E_ZSTD rather than emitting non-matching
 * single-threaded frames. 0 always writes a valid file, byte-identical only
 * below the reference rule's threshold. */
PVZSTD_API pvzstd_status pvzstd_writer_set_threads(pvzstd_writer *writer, int n_threads);

PVZSTD_API pvzstd_status pvzstd_writer_set_shuffle(pvzstd_writer *writer, pvzstd_shuffle_mode mode);

/* Cell topology stored without an offsets array; promotes the file version to 2. */
PVZSTD_API pvzstd_status pvzstd_writer_set_fixed_width_cells(pvzstd_writer *writer, int enabled);

/* Append one array; call order is frame order, which is how names map to frames.
 * ``dtype`` is a numpy dtype string such as "<f8". The data is copied. */
PVZSTD_API pvzstd_status pvzstd_writer_add_array(pvzstd_writer *writer, const char *name,
                                                 const char *dtype, const uint64_t *shape,
                                                 uint32_t ndim, const void *data, uint64_t nbytes);

/* Dataset-metadata JSON, stored under "<uid>__ds_metadata". Dataset readers
 * expect it. */
PVZSTD_API pvzstd_status pvzstd_writer_set_ds_metadata(pvzstd_writer *writer, const char *uid,
                                                       const char *json);

/* Emit the file. The trailing file-metadata frame is generated here, so it always
 * agrees with what was written. */
PVZSTD_API pvzstd_status pvzstd_writer_write(pvzstd_writer *writer, const char *path);

/* ------------------------------------------------------------------ *
 * Append
 *
 * Add field arrays without rewriting the container. Existing frames are copied
 * by offset, never decompressed, so the cost is what is added rather than the
 * file size; only the two metadata frames are regenerated. Each append commits
 * by rename, so an interrupted one cannot damage what was there.
 * ------------------------------------------------------------------ */

/* One array to append. The format records two dtype spellings for the same array
 * -- a string ("<f8") in the frame header, a name ("float64") in the dataset
 * metadata -- and both come from the caller: deriving one from the other would
 * mean carrying a copy of numpy's dtype-name table, and copies drift. */
typedef struct pvzstd_append_array {
  const char *name;       /* bare name; the UID prefix and suffix are added */
  const char *dtype;      /* header spelling, e.g. "<f8" */
  const char *dtype_name; /* metadata spelling, e.g. "float64" */
  const uint64_t *shape;
  uint32_t ndim;
  const void *data;
  uint64_t nbytes;
} pvzstd_append_array;

/* Pass as `level` to reuse the level recorded in the file. */
#define PVZSTD_LEVEL_FROM_FILE (-1000)

/* Append `count` arrays to the container at `path`. A name colliding with an
 * existing field array returns PVZSTD_E_INVALID rather than overwriting it.
 * MultiBlock is refused (PVZSTD_E_FORMAT): no single root dataset to append to.
 * Appending nothing is a no-op. */
PVZSTD_API pvzstd_status pvzstd_append_arrays(const char *path, const pvzstd_append_array *arrays,
                                              uint64_t count, int level,
                                              pvzstd_shuffle_mode shuffle);

/* ------------------------------------------------------------------ *
 * Streaming append
 *
 * The same edit as pvzstd_append_arrays, with the state kept open rather than
 * rediscovered per call -- that one re-reads the trailer and copies the body
 * every time, so its per-commit cost grows with the container (measured over 40
 * commits: 4.24x and 10.30x from the first five to the last five, dominated by
 * page cache). A stream of N commits produces the same bytes as N separate
 * pvzstd_append_arrays calls.
 *
 * The trade is crash behaviour, which is why both exist: append_arrays commits
 * by rename, whereas an interrupted stream commit leaves a trailer describing
 * frames that were not fully written. Use pvzstd_append_arrays when every commit
 * must leave a valid file.
 * ------------------------------------------------------------------ */

typedef struct pvzstd_stream pvzstd_stream;

/* Take over a container for streaming, parsing it once. Its two metadata arrays
 * must be the final two, as this library and the reference writer emit.
 * MultiBlock is refused (PVZSTD_E_FORMAT). */
PVZSTD_API pvzstd_status pvzstd_stream_open(const char *path, pvzstd_stream **out);

/* Commit `count` arrays as one group at the level recorded in the file. Names
 * must not collide with an existing field array. Committing nothing is a no-op. */
PVZSTD_API pvzstd_status pvzstd_stream_append(pvzstd_stream *stream,
                                              const pvzstd_append_array *arrays, uint64_t count,
                                              pvzstd_shuffle_mode shuffle);

/* Commits made so far. */
PVZSTD_API uint64_t pvzstd_stream_commit_count(const pvzstd_stream *stream);

/* Finalise. A mismatch against the caller's declared `expected_commits` returns
 * PVZSTD_E_RANGE, so a short stream is not mistaken for a complete result. Does
 * not release the stream. */
PVZSTD_API pvzstd_status pvzstd_stream_close(pvzstd_stream *stream, uint64_t expected_commits);

/* Release. Safe with NULL, and without closing -- on disk is whatever the last
 * completed commit left. */
PVZSTD_API void pvzstd_stream_free(pvzstd_stream *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PVZSTD_PVZSTD_H */
