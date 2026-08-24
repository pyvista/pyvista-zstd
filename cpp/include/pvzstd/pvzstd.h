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
#define PVZSTD_ABI_VERSION 8u

/* Dtype-field width and dataset-UID prefix width. They coincide in the format. */
#define PVZSTD_DTYPE_LEN 16

typedef enum pvz_status {
  PVZ_OK = 0,
  PVZ_E_IO = 1,      /* file missing, unreadable, or truncated */
  PVZ_E_FORMAT = 2,  /* trailer or header did not parse */
  PVZ_E_ZSTD = 3,    /* zstd rejected a frame or a compression parameter */
  PVZ_E_RANGE = 4,   /* index or count out of range, or destination too small */
  PVZ_E_NOMEM = 5,   /* allocation failed */
  PVZ_E_FILTER = 6,  /* per-array filter id this build cannot reverse */
  PVZ_E_INVALID = 7, /* NULL argument or misuse */
  /* The next two are refusals, not damage: the file parsed, and the operation
   * is the thing being declined. A caller that cannot tell them from
   * PVZ_E_FORMAT has to report a well-formed container as corrupt. */
  PVZ_E_UNSUPPORTED = 8, /* the container is a shape this operation cannot serve */
  PVZ_E_EXISTS = 9,      /* the name is already taken, and would be overwritten */
  PVZ_E_VERSION = 10     /* the container's file_version is newer than this build decodes */
} pvz_status;

/* An out-parameter that names which of a call's own arrays a status is about
 * carries this when the failure is not about one of them. */
#define PVZ_SLOT_NONE UINT64_MAX

/* Per-array filter ids. An unknown id is an error, never a passthrough. */
#define PVZ_FILTER_NONE 0
#define PVZ_FILTER_SHUFFLE 1

/* Highest container file_version this build can decode.
 *
 * A container stamped higher is refused rather than read: a newer format may
 * transform payloads in a way this build cannot invert, so reading one would
 * hand back corrupt values instead of failing. The ceiling lives here, beside
 * the decoder it describes, so every caller of this ABI gets the same answer --
 * a language binding that kept its own copy would refuse files the library can
 * read, or accept files it cannot. */
#define PVZSTD_FILE_VERSION_MAX 2u
PVZSTD_API uint32_t pvz_max_file_version(void);

typedef struct pvz_reader pvz_reader;

/* One array's header. Pointers are owned by the reader until pvz_close(). */
typedef struct pvz_array_info {
  const char *name;      /* NUL-terminated, UTF-8, includes the UID prefix */
  const uint64_t *shape; /* ndim entries; NULL when ndim == 0 */
  uint32_t ndim;
  uint8_t filter_id;
  char dtype[PVZSTD_DTYPE_LEN + 1]; /* e.g. "<f8", "|u1"; NUL-terminated */
  uint64_t nbytes;                  /* decompressed payload size */
} pvz_array_info;

/* Open a container. On success *out receives a reader that must be released
 * with pvz_close(). On failure *out is set to NULL. A container stamped newer
 * than PVZSTD_FILE_VERSION_MAX is refused with PVZ_E_VERSION. */
PVZSTD_API pvz_status pvz_open(const char *path, pvz_reader **out);

/* As pvz_open, additionally reporting the container's own file_version --
 * including when the open is refused for being too new, which is the case a
 * caller needs the number for. `file_version` may be NULL, and is left untouched
 * when the container carried no readable version. */
PVZSTD_API pvz_status pvz_open_versioned(const char *path, pvz_reader **out,
                                         uint32_t *file_version);

/* Release a reader. Safe to call with NULL. */
PVZSTD_API void pvz_close(pvz_reader *reader);

/* Number of arrays, excluding the two JSON metadata frames. */
PVZSTD_API uint64_t pvz_array_count(const pvz_reader *reader);

/* Describe array `index`. */
PVZSTD_API pvz_status pvz_array_info_at(const pvz_reader *reader, uint64_t index,
                                        pvz_array_info *out);

/* Describe `count` arrays from `first` in one boundary crossing rather than one
 * per array. PVZ_E_RANGE (writing nothing) if it runs past the end; a count of
 * zero succeeds. */
PVZSTD_API pvz_status pvz_array_info_range(const pvz_reader *reader, uint64_t first, uint64_t count,
                                           pvz_array_info *out);

/* Index of the array called `name`, or -1 if there is none. */
PVZSTD_API int64_t pvz_find_array(const pvz_reader *reader, const char *name);

/* Decompress array `index` into `dst`, reversing any filter. `dst_size` must
 * be at least the `nbytes` reported by pvz_array_info_at, or PVZ_E_RANGE is
 * returned and nothing is written. */
PVZSTD_API pvz_status pvz_read_array_at(const pvz_reader *reader, uint64_t index, void *dst,
                                        uint64_t dst_size);

/* Decompress `count` arrays over `n_threads` workers; `indices[i]` goes to
 * `dsts[i]`, at least `dst_sizes[i]` bytes. Frames are independent, so this is a
 * pure fan-out. 0 or 1 runs inline; negative means one worker per logical CPU,
 * the same sign convention pvz_writer_set_threads uses; PVZ_THREADS_AUTO picks
 * from the total size. Every count is capped at `count`, and the work and its
 * order do not depend on the setting. Returns the first non-OK status, by slot
 * rather than by which worker finished first.
 *
 * `failed_slot` (optional) receives the index into `indices` that status is
 * about, or PVZ_SLOT_NONE. Without it a caller can only learn that one of the
 * batch was refused, and has to re-read them singly to find out which. */
PVZSTD_API pvz_status pvz_read_arrays(const pvz_reader *reader, const uint64_t *indices,
                                      uint64_t count, void *const *dsts, const uint64_t *dst_sizes,
                                      int n_threads, uint64_t *failed_slot);

/* ---- Field arrays ----
 *
 * The blocks an append adds, named without the dataset-UID prefix or the
 * "__field_data" suffix the frame carries. The set and its order come from the
 * dataset metadata; scanning frame names for the suffix would also match an
 * ordinary array ending that way. MultiBlock containers report zero. */

/* Number of field arrays on the root dataset. */
PVZSTD_API uint64_t pvz_field_array_count(const pvz_reader *reader);

/* Bare name of field array `index`, owned by the reader, or NULL. */
PVZSTD_API const char *pvz_field_array_name_at(const pvz_reader *reader, uint64_t index);

/* Array index of field array `name`, or -1 -- including when the metadata names
 * it but the frame is absent, which is a damaged file. */
PVZSTD_API int64_t pvz_find_field_array(const pvz_reader *reader, const char *name);

/* The two JSON metadata documents, NUL-terminated and owned by the reader.
 * Either may be NULL if the container did not carry it.
 *
 * pvz_ds_metadata_json reports the LAST dataset-metadata document met, which is
 * the only one for a single-dataset container and an arbitrary block's for a
 * MultiBlock. Use the by-index family below to assemble a hierarchy. */
PVZSTD_API const char *pvz_ds_metadata_json(const pvz_reader *reader);
PVZSTD_API const char *pvz_file_metadata_json(const pvz_reader *reader);

/* ---- Every metadata document, by index ----
 *
 * A MultiBlock stores one dataset-metadata frame per block plus one per nested
 * MultiBlock, and a caller rebuilding the tree needs all of them together with
 * the frame name each was stored under -- the name carries the block's UID, and
 * the documents reference each other by UID alone.
 *
 * Order is file order. The file-metadata document is included, so a caller can
 * tell a legacy container from a current one by the name rather than by
 * re-reading the file: this build accepts both "__pyvista_zstd_metadata" and
 * the legacy "__zvtk_metadata", and only the name distinguishes them. */

/* Number of metadata documents the container carried. */
PVZSTD_API uint64_t pvz_metadata_count(const pvz_reader *reader);

/* Frame name of metadata document `index`, owned by the reader, or NULL. */
PVZSTD_API const char *pvz_metadata_name_at(const pvz_reader *reader, uint64_t index);

/* JSON of metadata document `index`, owned by the reader, or NULL. */
PVZSTD_API const char *pvz_metadata_json_at(const pvz_reader *reader, uint64_t index);

/* ---- Frame sizes ----
 *
 * Two frames per array, (header, payload), in file order -- including the
 * metadata frames, which are not reported as arrays. These are the trailer's
 * own numbers, so a caller needing them no longer has to parse the trailer
 * itself to answer "how big is this file decompressed". */

/* Total frame count, always even. */
PVZSTD_API uint64_t pvz_frame_count(const pvz_reader *reader);

/* Fill two caller-owned arrays with `capacity` entries each. Either pointer may
 * be NULL to skip that half. `capacity` is checked against pvz_frame_count()
 * and a short one returns PVZ_E_RANGE rather than writing past the end -- the
 * count comes from a separate call, so nothing else here can tell the two
 * apart. */
PVZSTD_API pvz_status pvz_frame_sizes(const pvz_reader *reader, uint64_t *decompressed,
                                      uint64_t *compressed, uint64_t capacity);

/* Every entry point below reports failure as a status code and never lets an
 * exception cross this boundary: a caller reaching it through ctypes or another
 * language has no way to catch one. Allocation failure surfaces as
 * PVZ_E_NOMEM; the accessors returning a pointer or a count report it as NULL
 * or as zero, which are the values they already use for "cannot answer". */

/* A static, human-readable string for a status code. Never NULL. */
PVZSTD_API const char *pvz_status_message(pvz_status status);

/* The ABI version this library was built with. */
PVZSTD_API uint32_t pvz_abi_version(void);

/* ------------------------------------------------------------------ *
 * Writer
 *
 * The container layer only: it takes arrays the caller already produced and
 * emits the file. Turning a dataset into arrays stays with the caller, which is
 * what keeps VTK out of this library and lets it build as WASM.
 *
 * Reproducing the reference writer byte for byte needs the same compression
 * level *and* worker count -- zstd's threaded mode emits different (equally
 * valid) bytes. See pvz_writer_set_threads.
 * ------------------------------------------------------------------ */

typedef struct pvz_writer pvz_writer;

/* Mirrors the reference writer's ``shuffle=False | True | "auto"``. */
typedef enum pvz_shuffle_mode {
  PVZ_SHUFFLE_NEVER = 0,  /* default */
  PVZ_SHUFFLE_ALWAYS = 1, /* still skipped for itemsize <= 1 */
  PVZ_SHUFFLE_AUTO = 2    /* float/complex only, and only if a trial compress shrinks */
} pvz_shuffle_mode;

/* Derive the worker count from payload size, as the reference writer does. */
#define PVZ_THREADS_AUTO (-2)

PVZSTD_API pvz_status pvz_writer_create(pvz_writer **out);
PVZSTD_API void pvz_writer_free(pvz_writer *writer);

/* Compression level; default 3, matching the reference writer. */
PVZSTD_API pvz_status pvz_writer_set_level(pvz_writer *writer, int level);

/* Worker count per frame. PVZ_THREADS_AUTO (default) reproduces the reference
 * rule: floor(total_MiB / 2), or -1 once that exceeds 8. 0 is single-threaded;
 * negative is one worker per logical CPU -- so such a file only reproduces on a
 * machine with the same CPU count, which is a property of the existing format.
 *
 * A zstd built without multithreading rejects any non-zero count, so
 * pvz_writer_write returns PVZ_E_ZSTD rather than emitting non-matching
 * single-threaded frames. 0 always writes a valid file, byte-identical only
 * below the reference rule's threshold. */
PVZSTD_API pvz_status pvz_writer_set_threads(pvz_writer *writer, int n_threads);

PVZSTD_API pvz_status pvz_writer_set_shuffle(pvz_writer *writer, pvz_shuffle_mode mode);

/* Cell topology stored without an offsets array; promotes the file version to 2. */
PVZSTD_API pvz_status pvz_writer_set_fixed_width_cells(pvz_writer *writer, int enabled);

/* Append one array; call order is frame order, which is how names map to frames.
 * ``dtype`` is a numpy dtype string such as "<f8". The data is copied. */
PVZSTD_API pvz_status pvz_writer_add_array(pvz_writer *writer, const char *name, const char *dtype,
                                           const uint64_t *shape, uint32_t ndim, const void *data,
                                           uint64_t nbytes);

/* As pvz_writer_add_array, but ``data`` is borrowed rather than copied: it must
 * stay allocated and unmodified until pvz_writer_write returns, or the writer is
 * freed. For a caller that already holds the buffer this halves the resident
 * bytes; a caller passing a temporary wants pvz_writer_add_array instead. */
PVZSTD_API pvz_status pvz_writer_add_array_borrowed(pvz_writer *writer, const char *name,
                                                    const char *dtype, const uint64_t *shape,
                                                    uint32_t ndim, const void *data,
                                                    uint64_t nbytes);

/* Dataset-metadata JSON, stored under "<uid>__ds_metadata". Dataset readers
 * expect it. */
PVZSTD_API pvz_status pvz_writer_set_ds_metadata(pvz_writer *writer, const char *uid,
                                                 const char *json);

/* Emit the file. The trailing file-metadata frame is generated here, so it always
 * agrees with what was written. */
PVZSTD_API pvz_status pvz_writer_write(pvz_writer *writer, const char *path);

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
typedef struct pvz_append_array {
  const char *name;       /* bare name; the UID prefix and suffix are added */
  const char *dtype;      /* header spelling, e.g. "<f8" */
  const char *dtype_name; /* metadata spelling, e.g. "float64" */
  const uint64_t *shape;
  uint32_t ndim;
  const void *data;
  uint64_t nbytes;
} pvz_append_array;

/* Pass as `level` to reuse the level recorded in the file. */
#define PVZ_LEVEL_FROM_FILE (-1000)

/* Append `count` arrays to the container at `path`. A name colliding with an
 * existing field array -- or repeated within one call -- returns PVZ_E_EXISTS
 * rather than overwriting it. MultiBlock is refused (PVZ_E_UNSUPPORTED): no
 * single root dataset to append to. Appending nothing is a no-op.
 *
 * `level` compresses the new frames only. The file's recorded
 * compression_level is left describing the frames that were already there, so
 * an explicit level different from it makes the field describe part of the
 * file rather than all of it. This is the reference implementation's behaviour
 * and the format has one such field; zstd frames carry their own parameters,
 * so nothing decoding the file depends on it. PVZ_LEVEL_FROM_FILE keeps the
 * two in agreement and is what a caller wanting one level should pass.
 *
 * The two refusals say what they are about rather than only that they happened,
 * both optional. On PVZ_E_EXISTS `clash_slot` receives the index into `arrays`
 * of the offered name that was already taken -- by the file or by an earlier
 * entry in this same call -- and PVZ_SLOT_NONE otherwise. On PVZ_E_VERSION
 * `found_version` receives the container's file_version, which is what makes
 * "too new" reportable against pvz_max_file_version(); it is 0 otherwise. */
PVZSTD_API pvz_status pvz_append_arrays(const char *path, const pvz_append_array *arrays,
                                        uint64_t count, int level, pvz_shuffle_mode shuffle,
                                        uint64_t *clash_slot, uint32_t *found_version);

/* ------------------------------------------------------------------ *
 * Streaming append
 *
 * The same edit as pvz_append_arrays, with the state kept open rather than
 * rediscovered per call -- that one re-reads the trailer and copies the body
 * every time, so its per-commit cost grows with the container. A stream of N
 * commits produces the same bytes as N separate pvz_append_arrays calls.
 *
 * Per-commit cost is flat in container size but still linear in the number of
 * field arrays already committed, since the dataset-metadata document is
 * re-scanned and re-emitted on every commit.
 *
 * The trade is crash behaviour, which is why both exist: append_arrays commits
 * by rename, whereas an interrupted stream commit leaves a trailer describing
 * frames that were not fully written. Use pvz_append_arrays when every commit
 * must leave a valid file.
 * ------------------------------------------------------------------ */

typedef struct pvz_stream pvz_stream;

/* Take over a container for streaming, parsing it once. Its two metadata arrays
 * must be the final two, as this library and the reference writer emit.
 * MultiBlock is refused (PVZ_E_UNSUPPORTED). */
PVZSTD_API pvz_status pvz_stream_open(const char *path, pvz_stream **out);

/* Commit `count` arrays as one group at the level recorded in the file. Names
 * must not collide with an existing field array. Committing nothing is a no-op. */
PVZSTD_API pvz_status pvz_stream_append(pvz_stream *stream, const pvz_append_array *arrays,
                                        uint64_t count, pvz_shuffle_mode shuffle);

/* Commits made so far. */
PVZSTD_API uint64_t pvz_stream_commit_count(const pvz_stream *stream);

/* Finalise. A mismatch against the caller's declared `expected_commits` returns
 * PVZ_E_RANGE, so a short stream is not mistaken for a complete result. Does
 * not release the stream. */
PVZSTD_API pvz_status pvz_stream_close(pvz_stream *stream, uint64_t expected_commits);

/* Release. Safe with NULL, and without closing -- on disk is whatever the last
 * completed commit left. */
PVZSTD_API void pvz_stream_free(pvz_stream *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PVZSTD_PVZSTD_H */
