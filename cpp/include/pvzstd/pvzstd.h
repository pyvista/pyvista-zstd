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

/* Bumped on any change to the set of declarations below, additions included.
 * A caller that binds this ABI dynamically should check it before anything
 * else.
 *
 * Additions are bumped too because the check is an equality, not a floor: the
 * Python package bundled with this library binds every symbol it knows about
 * up front, so meeting an older library that is missing one would fail at bind
 * time with an error naming the symbol rather than the version. Bumping keeps
 * that mismatch reported as what it is. */
#define PVZSTD_ABI_VERSION 2u

/* Width of the dtype field in an array header, and of the dataset-UID name
 * prefix. These coincide in the format and the coincidence is load-bearing. */
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

/* Per-array filter ids. An unknown id is an error, never a passthrough:
 * returning filtered bytes as-is would silently corrupt the array. */
#define PVZSTD_FILTER_NONE 0
#define PVZSTD_FILTER_SHUFFLE 1

typedef struct pvzstd_reader pvzstd_reader;

/* A view onto one array's header. All pointers are owned by the reader and
 * remain valid until pvzstd_close(); do not free them. */
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

/* Describe `count` arrays starting at `first`, writing out[0] .. out[count-1].
 *
 * Same information as calling pvzstd_array_info_at in a loop; the point is that
 * it is one crossing of the boundary rather than one per array. A caller that
 * reaches this library through a foreign-function layer pays a fixed cost per
 * call that can exceed the work being asked for, and a container holds one
 * array per data field, so describing a whole file was paying that cost a
 * dozen times to copy a few hundred bytes.
 *
 * Returns PVZSTD_E_RANGE and writes nothing if first + count runs past the end.
 * A count of zero succeeds and writes nothing. */
PVZSTD_API pvzstd_status pvzstd_array_info_range(const pvzstd_reader *reader, uint64_t first,
                                                 uint64_t count, pvzstd_array_info *out);

/* Index of the array called `name`, or -1 if there is none. */
PVZSTD_API int64_t pvzstd_find_array(const pvzstd_reader *reader, const char *name);

/* Decompress array `index` into `dst`, reversing any filter. `dst_size` must
 * be at least the `nbytes` reported by pvzstd_array_info_at, or PVZSTD_E_RANGE is
 * returned and nothing is written. */
PVZSTD_API pvzstd_status pvzstd_read_array_at(const pvzstd_reader *reader, uint64_t index,
                                              void *dst, uint64_t dst_size);

/* Decompress `count` arrays at once, spreading them over `n_threads` workers.
 *
 * `indices[i]` is decompressed into `dsts[i]`, which must be at least
 * `dst_sizes[i]` bytes. Frames are independent -- each has its own source
 * range and its own destination -- so this is a pure fan-out with no shared
 * mutable state and no ordering between arrays.
 *
 * This exists because it is the shape the work actually has. Reading arrays
 * one at a time leaves every core but one idle, and measured against the
 * reference reader's threaded batch decompressor that is a net loss, not a
 * win. `n_threads` <= 1 runs inline; PVZSTD_THREADS_AUTO uses the hardware
 * concurrency, capped at `count`.
 *
 * Returns the first non-OK status any array produced, so a single bad frame
 * still reports its own reason rather than a generic failure. */
PVZSTD_API pvzstd_status pvzstd_read_arrays(const pvzstd_reader *reader, const uint64_t *indices,
                                            uint64_t count, void *const *dsts,
                                            const uint64_t *dst_sizes, int n_threads);

/* ---- Field arrays ----
 *
 * The blocks an append adds, addressed the way a caller names them: by the
 * bare name, without the dataset-UID prefix or the "__field_data" suffix that
 * the frame carries. The list comes from the dataset metadata, which is what
 * defines the set and its order -- scanning frame names for the suffix would
 * also match an ordinary array whose name happens to end that way.
 *
 * Reading one costs its two frames and nothing else, because that is all
 * pvzstd_read_array_at touches. Seeding a container with a mesh and committing
 * each result as its own append therefore stays readable a block at a time,
 * however large the file grows.
 *
 * A MultiBlock container reports zero field arrays: it has no single root
 * dataset whose field data these would be. */

/* Number of field arrays on the root dataset. */
PVZSTD_API uint64_t pvzstd_field_array_count(const pvzstd_reader *reader);

/* Bare name of field array `index`, owned by the reader, or NULL if out of
 * range. */
PVZSTD_API const char *pvzstd_field_array_name_at(const pvzstd_reader *reader, uint64_t index);

/* Array index of the field array called `name` -- suitable for
 * pvzstd_array_info_at and pvzstd_read_array_at -- or -1 if the container has no
 * such field array. Also -1 when the metadata names it but its frame is
 * absent, which is a damaged file rather than a missing name. */
PVZSTD_API int64_t pvzstd_find_field_array(const pvzstd_reader *reader, const char *name);

/* The two JSON metadata documents, NUL-terminated and owned by the reader.
 * Either may be NULL if the container did not carry it. */
PVZSTD_API const char *pvzstd_ds_metadata_json(const pvzstd_reader *reader);
PVZSTD_API const char *pvzstd_file_metadata_json(const pvzstd_reader *reader);

/* A static, human-readable string for a status code. Never NULL. */
PVZSTD_API const char *pvzstd_status_message(pvzstd_status status);

/* The ABI version this library was built with. */
PVZSTD_API uint32_t pvzstd_abi_version(void);

/* ------------------------------------------------------------------ *
 * Writer
 *
 * Scope: this is the *container* layer. It takes arrays the caller has
 * already produced and emits the trailer-indexed file. Turning a dataset
 * into arrays (points / cells / celltypes / attribute arrays and the two
 * JSON metadata documents) stays with the caller -- that keeps VTK out of
 * this library entirely, which is what lets it build as a submodule and as
 * WASM.
 *
 * Byte-for-byte reproduction of the reference Python writer requires
 * matching the compression level *and the worker count*, because zstd's
 * multi-threaded mode emits different (equally valid) bytes than its
 * single-threaded mode. See pvzstd_writer_set_threads.
 * ------------------------------------------------------------------ */

typedef struct pvzstd_writer pvzstd_writer;

/* Per-array byte-shuffle policy, mirroring the reference writer's
 * ``shuffle=False | True | "auto"``. */
typedef enum pvzstd_shuffle_mode {
  PVZSTD_SHUFFLE_NEVER = 0,  /* default */
  PVZSTD_SHUFFLE_ALWAYS = 1, /* still skipped for itemsize <= 1 */
  PVZSTD_SHUFFLE_AUTO = 2    /* float/complex only, and only if a trial compress shrinks */
} pvzstd_shuffle_mode;

/* Pass to pvzstd_writer_set_threads to derive the worker count from the total
 * payload size the way the reference writer does. */
#define PVZSTD_THREADS_AUTO (-2)

PVZSTD_API pvzstd_status pvzstd_writer_create(pvzstd_writer **out);
PVZSTD_API void pvzstd_writer_free(pvzstd_writer *writer);

/* Compression level; default 3, matching the reference writer. */
PVZSTD_API pvzstd_status pvzstd_writer_set_level(pvzstd_writer *writer, int level);

/* Worker count per frame. PVZSTD_THREADS_AUTO (the default) reproduces the
 * reference rule: floor(total_MiB / 2), or -1 once that exceeds 8. A value
 * of 0 means single-threaded; negative means one worker per logical CPU.
 *
 * This is load-bearing for byte-identity, and note the consequence: a file
 * written with a negative worker count is only reproducible on a machine
 * with the same CPU count. That is a property of the existing format's
 * writer, not something introduced here.
 *
 * A zstd built without multithreading rejects any non-zero worker count, so
 * on such a build pvzstd_writer_write returns PVZSTD_E_ZSTD rather than quietly
 * emitting single-threaded frames that would not match. Passing 0 writes a
 * valid file on any build; above the reference rule's threshold that file is
 * correct but not byte-identical. */
PVZSTD_API pvzstd_status pvzstd_writer_set_threads(pvzstd_writer *writer, int n_threads);

PVZSTD_API pvzstd_status pvzstd_writer_set_shuffle(pvzstd_writer *writer, pvzstd_shuffle_mode mode);

/* Records that cell topology was stored without an offsets array, which is
 * what promotes the file version to 2. */
PVZSTD_API pvzstd_status pvzstd_writer_set_fixed_width_cells(pvzstd_writer *writer, int enabled);

/* Append one array. Frame order is the order of these calls and it is
 * significant -- it is how names map to frames. ``dtype`` is a numpy dtype
 * string such as "<f8" or "|u1". The data is copied. */
PVZSTD_API pvzstd_status pvzstd_writer_add_array(pvzstd_writer *writer, const char *name,
                                                 const char *dtype, const uint64_t *shape,
                                                 uint32_t ndim, const void *data, uint64_t nbytes);

/* Supply the dataset-metadata JSON document, stored under a
 * "<uid>__ds_metadata" frame. Optional but expected by dataset readers. */
PVZSTD_API pvzstd_status pvzstd_writer_set_ds_metadata(pvzstd_writer *writer, const char *uid,
                                                       const char *json);

/* Emit the file. The trailing file-metadata frame (frame names, level,
 * resolved file version) is generated here, so it always agrees with what
 * was actually written. */
PVZSTD_API pvzstd_status pvzstd_writer_write(pvzstd_writer *writer, const char *path);

/* ------------------------------------------------------------------ *
 * Append
 *
 * Add field arrays to a container that already exists, without rewriting
 * it. Frames already on disk are copied byte-for-byte by offset -- never
 * decompressed, never recompressed -- so the cost is the size of what is
 * being added, not the size of the file. Only the dataset-metadata frame
 * and the trailing file-metadata frame are regenerated, because both grow.
 *
 * This is what makes a container writable incrementally: seed it once with
 * the mesh, then commit each result (a load step, a mode, a frequency) as
 * its own append. Peak memory is one array rather than the whole result
 * set, and each append is committed by rename, so an interrupted one cannot
 * damage what was already there.
 * ------------------------------------------------------------------ */

/* One array to append. Note the two dtype spellings: the format records a
 * numpy dtype *string* ("<f8") in the frame header and a dtype *name*
 * ("float64") in the dataset metadata, for the same array. Both are taken
 * from the caller rather than derived here -- deriving one from the other
 * would mean embedding a copy of numpy's dtype-name table in this library,
 * and a copy is a thing that drifts. */
typedef struct pvzstd_append_array {
  const char *name;       /* bare name; the UID prefix and suffix are added */
  const char *dtype;      /* header spelling, e.g. "<f8" */
  const char *dtype_name; /* metadata spelling, e.g. "float64" */
  const uint64_t *shape;
  uint32_t ndim;
  const void *data;
  uint64_t nbytes;
} pvzstd_append_array;

/* Pass as `level` to reuse the level recorded in the file, so appended
 * blocks are compressed the same way the original ones were. */
#define PVZSTD_LEVEL_FROM_FILE (-1000)

/* Append `count` arrays to the container at `path`.
 *
 * Names must not collide with a field array already in the file; this
 * returns PVZSTD_E_INVALID rather than overwriting one. MultiBlock containers
 * are refused (PVZSTD_E_FORMAT): they have no single root dataset to append
 * to, and misreading their metadata as a dataset's would corrupt the file.
 *
 * Appending nothing is a no-op, not an error. */
PVZSTD_API pvzstd_status pvzstd_append_arrays(const char *path, const pvzstd_append_array *arrays,
                                              uint64_t count, int level,
                                              pvzstd_shuffle_mode shuffle);

/* ------------------------------------------------------------------ *
 * Streaming append
 *
 * The same edit pvzstd_append_arrays performs, with the state it needs kept
 * open instead of rediscovered. That function is handed a path, so every
 * call re-reads the trailer, decompresses both metadata frames, and copies
 * the body to a temporary file: cost proportional to the whole container.
 * Measured over 40 single-block commits, the per-commit cost rose from the
 * first five to the last five by 4.24x on one run and 10.30x on another --
 * how much of the container is still in page cache dominates the figure, so
 * treat the growth as unbounded rather than as a constant. The direction is
 * the reproducible part.
 *
 * A stream holds the trailer, the offset where the metadata tail begins,
 * and the dataset-metadata document, so committing costs what is being
 * added and not what is already there. The bytes are the same: a stream of
 * N commits produces the file N separate pvzstd_append_arrays calls would.
 *
 * The trade is crash behaviour, and it is why both exist.
 * pvzstd_append_arrays commits by rename, so an interrupted call leaves the
 * previous file intact. A stream writes in place, so a commit interrupted
 * partway leaves a trailer describing frames that were not fully written --
 * earlier commits' bytes are never revisited, but recovering them means
 * rebuilding the trailer, which this library does not do. Use
 * pvzstd_append_arrays when every single commit must leave a valid file.
 * ------------------------------------------------------------------ */

typedef struct pvzstd_stream pvzstd_stream;

/* Take over an existing container for streaming. Parses it once; nothing
 * after this re-reads the file.
 *
 * The container must have its two metadata arrays as the final two, which
 * is what this library and the reference writer emit. MultiBlock files are
 * refused (PVZSTD_E_FORMAT): no single root dataset to stream into. */
PVZSTD_API pvzstd_status pvzstd_stream_open(const char *path, pvzstd_stream **out);

/* Commit `count` arrays as one group, under the compression level and
 * settings recorded in the file. Names must not collide with a field array
 * already present. Committing nothing is a no-op, not an error. */
PVZSTD_API pvzstd_status pvzstd_stream_append(pvzstd_stream *stream,
                                              const pvzstd_append_array *arrays, uint64_t count,
                                              pvzstd_shuffle_mode shuffle);

/* Commits made so far. */
PVZSTD_API uint64_t pvzstd_stream_commit_count(const pvzstd_stream *stream);

/* Finalise. `expected_commits` is the count the caller declared up front;
 * a mismatch returns PVZSTD_E_RANGE rather than letting a short stream be
 * mistaken for a complete result. Does not release the stream. */
PVZSTD_API pvzstd_status pvzstd_stream_close(pvzstd_stream *stream, uint64_t expected_commits);

/* Release. Safe with NULL, and safe without closing -- what is on disk is
 * whatever the last completed commit left. */
PVZSTD_API void pvzstd_stream_free(pvzstd_stream *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PVZSTD_PVZSTD_H */
