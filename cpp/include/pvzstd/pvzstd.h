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
#define PVZSTD_ABI_VERSION 11u

/* Dtype-field width and dataset-UID prefix width. They coincide in the format. */
#define PVZSTD_DTYPE_LEN 16

typedef enum pvzstd_status {
  PVZSTD_OK = 0,
  PVZSTD_E_IO = 1,      /* file missing, unreadable, or truncated */
  PVZSTD_E_FORMAT = 2,  /* trailer or header did not parse */
  PVZSTD_E_ZSTD = 3,    /* zstd rejected a frame or a compression parameter */
  PVZSTD_E_RANGE = 4,   /* index or count out of range, or destination too small */
  PVZSTD_E_NOMEM = 5,   /* allocation failed */
  PVZSTD_E_FILTER = 6,  /* per-array filter id this build cannot reverse */
  PVZSTD_E_INVALID = 7, /* NULL argument or misuse */
  /* The next two are refusals, not damage: the file parsed, and the operation
   * is the thing being declined. A caller that cannot tell them from
   * PVZSTD_E_FORMAT has to report a well-formed container as corrupt. */
  PVZSTD_E_UNSUPPORTED = 8, /* the container is a shape this operation cannot serve */
  PVZSTD_E_EXISTS = 9,      /* the name is already taken, and would be overwritten */
  PVZSTD_E_VERSION = 10,    /* the container's file_version is newer than this build decodes */
  /* Another append holds this container. Nothing was written; the container is
   * intact and the same call made again once the other one finishes succeeds.
   * Also what a lock left behind by a killed append reports, which retrying
   * does not clear -- see the Append section for the file to remove. */
  PVZSTD_E_BUSY = 11,
  /* The file changed under the operation, which had staged its result against
   * what it read. Nothing was written; the same call made again reads what is
   * there now and succeeds. Distinct from PVZSTD_E_IO because it is the one
   * failure here that retrying is the right answer to. */
  PVZSTD_E_CHANGED = 12
} pvzstd_status;

/* An out-parameter that names which of a call's own arrays a status is about
 * carries this when the failure is not about one of them. */
#define PVZSTD_SLOT_NONE UINT64_MAX

/* Per-array filter ids. An unknown id is an error, never a passthrough. */
#define PVZSTD_FILTER_NONE 0
#define PVZSTD_FILTER_SHUFFLE 1

/* Highest container file_version this build can decode.
 *
 * A container stamped higher is refused rather than read: a newer format may
 * transform payloads in a way this build cannot invert, so reading one would
 * hand back corrupt values instead of failing. The ceiling lives here, beside
 * the decoder it describes, so every caller of this ABI gets the same answer --
 * a language binding that kept its own copy would refuse files the library can
 * read, or accept files it cannot. */
#define PVZSTD_FILE_VERSION_MAX 2u
PVZSTD_API uint32_t pvzstd_max_file_version(void);

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
 * with pvzstd_close(). On failure *out is set to NULL. A container stamped newer
 * than PVZSTD_FILE_VERSION_MAX is refused with PVZSTD_E_VERSION. */
PVZSTD_API pvzstd_status pvzstd_open(const char *path, pvzstd_reader **out);

/* As pvzstd_open, additionally reporting the container's own file_version --
 * including when the open is refused for being too new, which is the case a
 * caller needs the number for. `file_version` may be NULL, and is left untouched
 * when the container carried no readable version. */
PVZSTD_API pvzstd_status pvzstd_open_versioned(const char *path, pvzstd_reader **out,
                                               uint32_t *file_version);

/* Open a container the caller already holds: `data` must point at `size`
 * contiguous bytes of a whole container. NULL, or a size of zero, is
 * PVZSTD_E_INVALID -- a zero size is a bad argument, where a zero-length file
 * is a property of the thing opened and so reaches pvzstd_open as PVZSTD_E_IO.
 * The two doors differ there on purpose; a caller writing one error handler
 * over both should expect it.
 *
 * The bytes are borrowed, never copied: the caller owns the buffer and must
 * keep it allocated and unmodified until pvzstd_close(), which is what makes
 * this cheaper than staging the container somewhere the file entry point can
 * reach. Modifying it meanwhile is not detected: the offsets and sizes were
 * read at open time, so the arrays read afterwards hold undefined contents and
 * no status reports it. For a caller that already has the bytes -- an archive
 * member, an HTTP response body, or a build with no filesystem to open a path
 * on -- this is pvzstd_open over the same container, and refuses a crafted
 * buffer wherever pvzstd_open would refuse a crafted file. */
PVZSTD_API pvzstd_status pvzstd_open_memory(const void *data, uint64_t size, pvzstd_reader **out);

/* As pvzstd_open_memory, additionally reporting the container's own
 * file_version, on the same terms as pvzstd_open_versioned -- including when
 * the open is refused for being too new. A caller reading from memory needs
 * that number for the same reason a caller reading from a path does. */
PVZSTD_API pvzstd_status pvzstd_open_memory_versioned(const void *data, uint64_t size,
                                                      pvzstd_reader **out, uint32_t *file_version);

/* Release a reader. Safe to call with NULL. A reader opened from memory
 * borrows its bytes, so this releases only what the reader itself acquired --
 * the caller's buffer is left alone, and is theirs to free afterwards. */
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
 * pure fan-out. 0 or 1 runs inline; negative means one worker per logical CPU,
 * the same sign convention pvzstd_writer_set_threads uses; PVZSTD_THREADS_AUTO picks
 * from the total size. Every count is capped at `count`, and the work and its
 * order do not depend on the setting. Returns the first non-OK status, by slot
 * rather than by which worker finished first.
 *
 * `failed_slot` (optional) receives the index into `indices` that status is
 * about, or PVZSTD_SLOT_NONE. Without it a caller can only learn that one of the
 * batch was refused, and has to re-read them singly to find out which. */
PVZSTD_API pvzstd_status pvzstd_read_arrays(const pvzstd_reader *reader, const uint64_t *indices,
                                            uint64_t count, void *const *dsts,
                                            const uint64_t *dst_sizes, int n_threads,
                                            uint64_t *failed_slot);

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
 * Either may be NULL if the container did not carry it.
 *
 * pvzstd_ds_metadata_json reports the LAST dataset-metadata document met, which is
 * the only one for a single-dataset container and an arbitrary block's for a
 * MultiBlock. Use the by-index family below to assemble a hierarchy. */
PVZSTD_API const char *pvzstd_ds_metadata_json(const pvzstd_reader *reader);
PVZSTD_API const char *pvzstd_file_metadata_json(const pvzstd_reader *reader);

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
PVZSTD_API uint64_t pvzstd_metadata_count(const pvzstd_reader *reader);

/* Frame name of metadata document `index`, owned by the reader, or NULL. */
PVZSTD_API const char *pvzstd_metadata_name_at(const pvzstd_reader *reader, uint64_t index);

/* JSON of metadata document `index`, owned by the reader, or NULL. */
PVZSTD_API const char *pvzstd_metadata_json_at(const pvzstd_reader *reader, uint64_t index);

/* ---- Frame sizes ----
 *
 * Two frames per array, (header, payload), in file order -- including the
 * metadata frames, which are not reported as arrays. These are the trailer's
 * own numbers, so a caller needing them no longer has to parse the trailer
 * itself to answer "how big is this file decompressed". */

/* Total frame count, always even. */
PVZSTD_API uint64_t pvzstd_frame_count(const pvzstd_reader *reader);

/* Fill two caller-owned arrays with `capacity` entries each. Either pointer may
 * be NULL to skip that half. `capacity` is checked against pvzstd_frame_count()
 * and a short one returns PVZSTD_E_RANGE rather than writing past the end -- the
 * count comes from a separate call, so nothing else here can tell the two
 * apart. */
PVZSTD_API pvzstd_status pvzstd_frame_sizes(const pvzstd_reader *reader, uint64_t *decompressed,
                                            uint64_t *compressed, uint64_t capacity);

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

/* As pvzstd_writer_add_array, but ``data`` is borrowed rather than copied: it must
 * stay allocated and unmodified until pvzstd_writer_write returns, or the writer is
 * freed. For a caller that already holds the buffer this halves the resident
 * bytes; a caller passing a temporary wants pvzstd_writer_add_array instead. */
PVZSTD_API pvzstd_status pvzstd_writer_add_array_borrowed(pvzstd_writer *writer, const char *name,
                                                          const char *dtype, const uint64_t *shape,
                                                          uint32_t ndim, const void *data,
                                                          uint64_t nbytes);

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
 *
 * One append at a time, per container, and that is enforced rather than
 * documented. An append is a read-modify-write: it reads the container, adds
 * to what it read, and commits the result. Two of them running at once each
 * commit "what was there plus mine", and the second to land replaces the
 * first's arrays with a body copied before those arrays existed -- with both
 * callers told they succeeded. Checking at the end does not recover that: two
 * appends doing equal work reach their commits within microseconds of each
 * other, so neither check sees the other's result yet.
 *
 * The exclusion is an advisory lock file, "<path>.append.lock", created
 * exclusively and held for the call. A second append meanwhile returns
 * PVZSTD_E_BUSY at once, having read and written nothing; it does not wait,
 * because a library has no business choosing how long its caller blocks.
 * Exclusive file creation is the primitive because it is the one every target
 * this builds for implements the same way -- flock() would be the obvious
 * choice and on the WebAssembly target it returns success without locking
 * anything, which is a guarantee stated for a target that does not keep it.
 *
 * The cost is a lock that outlives a killed process: an append that dies
 * between taking the lock and finishing leaves the file behind, and every
 * later append to that container returns PVZSTD_E_BUSY until it is deleted.
 * That is deliberate. It is a visible, named, recoverable failure, where what
 * it replaces was one writer's arrays disappearing silently. Deleting the file
 * is safe whenever no append is running.
 *
 * The lock binds appends and nothing else. A caller that replaces the
 * container by other means -- another tool, or a plain move onto the path --
 * is caught separately: each call stages into a file the operating system
 * names, so no two callers are handed the same staging file, and before
 * committing, a call checks that its path still names the container it read.
 * One replaced during the staging returns PVZSTD_E_CHANGED having written
 * nothing. That one is a check and not a lock, and covers the staging rather
 * than the microseconds between the check and the rename.
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
 * existing field array -- or repeated within one call -- returns PVZSTD_E_EXISTS
 * rather than overwriting it. MultiBlock is refused (PVZSTD_E_UNSUPPORTED): no
 * single root dataset to append to. Appending nothing is a no-op.
 *
 * `level` compresses the new frames only. The file's recorded
 * compression_level is left describing the frames that were already there, so
 * an explicit level different from it makes the field describe part of the
 * file rather than all of it. This is the reference implementation's behaviour
 * and the format has one such field; zstd frames carry their own parameters,
 * so nothing decoding the file depends on it. PVZSTD_LEVEL_FROM_FILE keeps the
 * two in agreement and is what a caller wanting one level should pass.
 *
 * PVZSTD_E_BUSY means another append holds this container, or a killed one left
 * "<path>.append.lock" behind. PVZSTD_E_CHANGED means something that does not
 * take that lock replaced the container while this call was staging its result.
 * Neither wrote anything. See the section comment above.
 *
 * The two refusals say what they are about rather than only that they happened,
 * both optional. On PVZSTD_E_EXISTS `clash_slot` receives the index into `arrays`
 * of the offered name that was already taken -- by the file or by an earlier
 * entry in this same call -- and PVZSTD_SLOT_NONE otherwise. On PVZSTD_E_VERSION
 * `found_version` receives the container's file_version, which is what makes
 * "too new" reportable against pvzstd_max_file_version(); it is 0 otherwise. */
PVZSTD_API pvzstd_status pvzstd_append_arrays(const char *path, const pvzstd_append_array *arrays,
                                              uint64_t count, int level,
                                              pvzstd_shuffle_mode shuffle, uint64_t *clash_slot,
                                              uint32_t *found_version);

/* ------------------------------------------------------------------ *
 * Streaming append
 *
 * The same edit as pvzstd_append_arrays, with the state kept open rather than
 * rediscovered per call -- that one re-reads the trailer and copies the body
 * every time, so its per-commit cost grows with the container. A stream of N
 * commits produces the same bytes as N separate pvzstd_append_arrays calls.
 *
 * Per-commit cost is flat in container size but still linear in the number of
 * field arrays already committed, since the dataset-metadata document is
 * re-scanned and re-emitted on every commit.
 *
 * The trade is crash behaviour, which is why both exist: append_arrays commits
 * by rename, whereas an interrupted stream commit leaves a trailer describing
 * frames that were not fully written. Use pvzstd_append_arrays when every commit
 * must leave a valid file.
 *
 * A stream takes no lock and is covered by none: it writes into the container
 * in place, so it has neither a commit point at which to notice another writer
 * nor a staging file to hold back, and PVZSTD_E_BUSY and PVZSTD_E_CHANGED are
 * not among the statuses it can return. One stream at a time, and no append
 * against the same container while it is open, is the caller's to arrange.
 * ------------------------------------------------------------------ */

typedef struct pvzstd_stream pvzstd_stream;

/* Take over a container for streaming, parsing it once. Its two metadata arrays
 * must be the final two, as this library and the reference writer emit.
 * MultiBlock is refused (PVZSTD_E_UNSUPPORTED). */
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
