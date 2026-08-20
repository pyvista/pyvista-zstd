# The `.pv` / `.zvtk` container — format specification

Status: **DRAFT, derived by reading `src/pyvista_zstd/pyvista_zstd.py` at
`6f1d021`.** Every claim below carries the line range it was read from.
Claims that were *not* verified against the source are marked **UNVERIFIED**
and must be closed before the C++ core is written against them.

This document exists because the Python implementation is currently the only
specification. A second implementation written against a reading rather than
against a spec diverges silently, and the divergence surfaces in a user's
file rather than in CI. The conformance corpus (`tests/conformance/`) is the
executable half of this document; this prose is the reviewable half.

## 1. File layout

The container is **trailer-indexed**. Payload frames come first, the index
second, and the frame count occupies the final eight bytes:

```
+-------------------------------------------------+
| zstd frame 0                                    |
| zstd frame 1                                    |
| ...                                             |
| zstd frame N-1                                  |
+-------------------------------------------------+
| index: (offset:u64, decompressed_size:u64) x N  |   16 bytes per entry
+-------------------------------------------------+
| N : u64                                         |   final 8 bytes
+-------------------------------------------------+
```

All integers are **little-endian** (`struct.pack("<Q", ...)`, `:819-820`).

Reading is therefore a backwards walk from EOF (`:1243-1254`):

1. `seek(-8, SEEK_END)`, read `u64` -> `N`.
2. `seek(-(8 + N * 16), SEEK_END)`, read `N * 16` bytes -> the index.
3. Each entry is `(offset, decompressed_size)`; `offset` locates the frame
   from the start of the file.

### 1.1 Consequences that bind the implementation

- **A writer must be able to seek.** The index and count can only be written
  once every frame length is known. This is why the in-tree native writer
  reopens with `r+b` to rewrite its tail. A write-only, forward-only sink is
  not sufficient for this format.
- **A reader must have the tail before it can read anything.** There is no
  forward-parseable stream: the index is mandatory and it lives at the end.
  A reader over a non-seekable source must buffer the whole file.
- **The index entry width is 16 bytes and this is load-bearing.** It coincides
  with `UID_N_CHAR = 16` (`:120`), and the `EMPTY_DS` sentinel is a 16-character
  string specifically so it aligns with a UID (`:121`). Changing either
  constant desynchronises the other.

## 2. File versions

`FILE_VERSION = 2` is what writers emit today (`:74`). Three versions exist
(`:78-80`) and a 1:1 reader must accept all three:

| Value | Name                        | Meaning                                  |
|-------|-----------------------------|------------------------------------------|
| 0     | `FILE_VERSION_UNFILTERED`   | No byte-shuffle. No `filter_id` in array headers. |
| 1     | `FILE_VERSION_SHUFFLE`      | Byte-shuffle filter available; `filter_id` present. |
| 2     | `FILE_VERSION_FIXED_WIDTH_CELLS` | Adds fixed-width cell topology encoding. |

The version is promoted by the *writer* only when it actually uses an optional
encoding — the `ZstdFileMetadata` default is the legacy
`FILE_VERSION_UNFILTERED` (`:220-222`). **A file's version therefore describes
the features it uses, not the version of the library that wrote it.**

## 3. Per-array frame header

Written at `:641-652`, parsed at `:840-858`:

```
u32   name_len
u8[]  name                (name_len bytes, UTF-8)
u32   ndim
u64   shape[ndim]
u8    filter_id           (file_version >= 1 only)
```

`filter_id` is `_FILTER_NONE = 0` or `_FILTER_SHUFFLE = 1` (`:142-143`).

**UNVERIFIED:** where the array *dtype* is carried. `ArrayInfo` (`:205-211`)
holds `shape` and `dtype`, and `DataSetMetadata` holds `points_dtype` and
`celltypes_dtype` (`:272-284`), so dtype may travel in the JSON metadata
rather than in this binary header. This must be resolved before writing the
C++ parser — guessing here produces a reader that works on the common case
and corrupts the uncommon one.

## 4. Metadata

Three JSON documents, each stored as a frame:

- **`ZstdFileMetadata`** (`:213-238`) under key `__pyvista_zstd_metadata`
  (`:94`), legacy `__zvtk_metadata` (`:95`). Fields: `frame_names`
  (**order is significant** — `:791`), `compression_level`, `compression`
  (default `"zstandard"`), `file_version`. Serialised with
  `separators=(",", ":")`, i.e. no whitespace (`:226`).
- **`DataSetMetadata`** (`:272-300`) under `__ds_metadata` (`:92`).
- **`MultiBlockMetadata`** (`:239-241`) under `__multiblock__ds_metadata` (`:93`).

`DataSetMetadata` carries, besides shape/dtype: per-association key maps
(`point_data_keys`, `cell_data_keys`, `field_data_keys`), `fixed_cell_sizes`,
the eight **active-attribute** names (scalars / vectors / texture coordinates /
normals, for both point and cell data), and the optional ImageData block
(`dimensions`, `origin`, `spacing`, `direction_matrix`, `offset`).

The active-attribute names are easy to overlook and are **observable
behaviour**: dropping them silently changes which array a downstream plot
colours by. They are in scope for 1:1 parity.

## 5. The byte-shuffle filter and its adaptive heuristic

Shuffle transposes an `(n_elem, itemsize)` byte matrix so that like-significance
bytes become adjacent, which usually compresses better for float arrays.

The choice is **adaptive**, not static (`_auto_shuffle_beneficial`, `:159-172`):

1. Take a sample of `min(n_elem, 1 MiB // itemsize)` elements, **centred** in
   the array (`start = (n_elem - n_sample) // 2`) as being representative of
   the bulk.
2. Compress the sample twice at the target level — raw, and shuffled.
3. Keep shuffle **only if the shuffled result is strictly smaller**.

This is the single most important parity hazard in the format. The decision is
recorded in `filter_id` and therefore changes the bytes on disk. A C++
implementation that makes this decision by any cheaper means may pick a
different filter than Python would have for the same input: the file still
round-trips correctly, but it is not byte-identical.

**Parity policy for this port: reproduce the decision exactly.** Optimise the
shuffle kernel and the surrounding copies, not the decision rule.

## 6. Compression threading

`_set_n_threads` (`:407-417`): when the caller passes `None`, threads are
guessed as `n_bytes // 2 MiB`, and if that exceeds `max_manual_threads = 8`
the value becomes `-1`, which hands thread selection to zstd itself.

The existing implementation is therefore **already multithreaded**. A C++ port
does not get a speedup for free by being C++; the headroom is in the shuffle
kernel, the probe, and copy elision.

## 7. Dataset coverage

`ds_type` (`:82`) selects the reconstruction path. In scope for 1:1:
PointSet, PolyData (with `polys` / `lines` / `strips` / `verts`, `:115-118`),
UnstructuredGrid (including `polyhedron` and `polyhedron_locaction` —
note the spelling in the source, `:111-112`), RectilinearGrid (`_x_rgrid` /
`_y_rgrid` / `_z_rgrid`, `:100-102`), ImageData, and MultiBlock.

`EMPTY_DS` (`:121`) is a sentinel for an empty dataset and is 16 characters
to align with the UID width.

## 8. Suffixes

`.pv` is current, `.zvtk` is legacy; both are accepted for reading
(`SUPPORTED_READ_SUFFIXES`, `:98`) and both are registered as PyVista reader
and writer entry points. Legacy files additionally use the
`__zvtk_metadata` metadata key.

## 9. Open questions blocking the C++ core

1. **Where does array dtype live** — binary header or JSON metadata? (§3)
2. **What exactly does `FILE_VERSION_FIXED_WIDTH_CELLS` change** in the cell
   encoding relative to version 1?
3. **Is frame order semantically meaningful** beyond index lookup? `:791`
   says "frame order matters" without saying to whom.
4. **What is the append path's contract** when a file is extended
   (`append.py`, 553 lines) — specifically whether the index is rewritten in
   place or appended to.

None of these are answerable by inspection alone with confidence; each gets a
targeted probe plus a conformance case before any C++ is written against it.
