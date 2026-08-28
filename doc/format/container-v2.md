# The `.pv` / `.zvtk` container — format specification

Status: **Normative.** This is the specification the implementation is held
to, not a description of it. Where the two disagree, this document and the
conformance suite decide and the code is what changes.

It did not start that way. It was derived by reading
`src/pyvista_zstd/pyvista_zstd.py` at `6f1d021`, when that module was the only
implementation, and the line ranges every claim carries are that reading's
provenance: they point into the file **as it stood at `6f1d021`** and are not
current line numbers. The C++ core was then written against this document
(`f88a1b5`), and `pyvista_zstd.py` is now a `ctypes` binding to that core
rather than a second implementation of the format, so the direction has
reversed.

The conformance corpus (`tests/conformance/`) is the executable half of this
document; this prose is the reviewable half. A second implementation written
against a reading rather than against a spec diverges silently, and the
divergence surfaces in a user's file rather than in CI.

Two things this document does not cover, both of which shipped after it was
written: the C ABI itself, which is documented in `doc/c_api.rst`, and the
`pvzstd_` symbol names, which appear below only where the format's own rules
are visible through them. Every question §9 once left open is now closed.

## 1. File layout

The container is **trailer-indexed**. Payload frames come first, the index
second, and the frame count occupies the final eight bytes:

```
+-------------------------------------------------+
| zstd frame 0            <- starts at byte 0     |
| zstd frame 1                                    |
| ...                                             |
| zstd frame N-1                                  |
+-------------------------------------------------+
| index: (end:u64, decompressed_size:u64) x N     |   16 bytes per entry
+-------------------------------------------------+
| N : u64                                         |   final 8 bytes
+-------------------------------------------------+
```

All integers are **little-endian** (`struct.pack("<Q", ...)`, `:819-820`).

**There is no magic number and no file header.** Byte 0 is the first byte of
the first zstd frame. A file is identified as a container only by successfully
parsing its trailer.

**The first index field is the frame's END offset, not its start.** The writer
comment is explicit — `frame_meta` is `(compressed_end, decompressed_size)`
(`:808`) — and `offset` is incremented _after_ the write (`:814-815`). The
reader reconstructs starts by shifting (`:1256`):

```
starts = [0] + ends[:-1]
frame i occupies raw[starts[i] : ends[i]]
```

Reading is therefore a backwards walk from EOF (`:1243-1258`):

1. `seek(-8, SEEK_END)`, read `u64` -> `N`.
2. `seek(-(8 + N * 16), SEEK_END)`, read `N * 16` bytes -> the index.
3. Each entry is `(end, decompressed_size)`; derive starts as above.

Misreading field 1 as a start offset does **not** fail loudly: every frame
still decompresses, but each one yields the _previous_ frame's payload, so
the array data silently pairs with the wrong header. This was caught here only
because the declared `decompressed_size` disagreed with the produced length —
which is why a conforming reader should assert that equality.

### 1.2 Frames come in (header, payload) pairs

Frames are not one-per-array. Each logical array occupies **two consecutive
frames**: a binary metadata header (§3) followed by its raw payload. The reader
relies on this directly (`:1315`):

```python
index = self._metadata.frame_names.index(key) * 2  # times two for metadata
```

So `frame_names[i]` is described by frame `2i` and stored in frame `2i + 1`,
and `N` is always even. Frame order is therefore load-bearing, not merely
advisory: the mapping from name to frame is positional arithmetic, not a
lookup.

One asymmetry matters for a reader that wants to iterate by name:
**`frame_names` lists every frame pair except the last one.** It includes
`__ds_metadata` but omits the trailing `__pyvista_zstd_metadata` pair, which is
written after the list is serialised and is reachable only positionally. So

```
len(frame_names) == N / 2 - 1
```

and a reader must not assume the two sequences are the same length. Verified
across PolyData / UnstructuredGrid / ImageData / StructuredGrid that header
order equals `frame_names` order on every frame but that last one.

Note also that not every dataset writes a `points` frame: ImageData's geometry
is implied by `dimensions` / `origin` / `spacing` in `__ds_metadata`, so an
ImageData file with no attached arrays contains only the two metadata pairs.

### 1.1 Consequences that bind the implementation

- **A writer must know every frame's length before it can emit the index**,
  but it does not have to seek: the index goes last and its offsets are
  cumulative ends, so a writer that accumulates them as it goes can stream the
  body and then append the trailer. `pvzstd_writer_write` does exactly that --
  one `wb` handle, frames written in order, trailer appended, close.
  _Editing_ an existing container is where seeking becomes unavoidable, and
  the two edit paths make opposite trades: `pvzstd_append_arrays` stages a
  whole new file and commits it by rename, while `pvzstd_stream_append` holds
  the original open `r+b`, overwrites the tail in place and truncates -- which
  is why an interrupted stream commit can leave a trailer describing frames
  that were not fully written, and an interrupted `pvzstd_append_arrays`
  cannot.
- **A reader must have the tail before it can read anything.** There is no
  forward-parseable stream: the index is mandatory and it lives at the end.
  A reader over a non-seekable source must buffer the whole file, which is
  what `pvzstd_open_memory` exists to serve: it takes the whole container as
  borrowed bytes the caller already holds and parses it in place, refusing a
  crafted buffer wherever the path entry point would refuse a crafted file.
- **The index entry width is 16 bytes and this is load-bearing.** It coincides
  with `UID_N_CHAR = 16` (`:120`), and the `EMPTY_DS` sentinel is a 16-character
  string specifically so it aligns with a UID (`:121`). Changing either
  constant desynchronises the other.

## 2. File versions

`FILE_VERSION = 2` is what writers emit today (`:74`). Three versions exist
(`:78-80`) and a 1:1 reader must accept all three:

| Value | Name                             | Meaning                                             |
| ----- | -------------------------------- | --------------------------------------------------- |
| 0     | `FILE_VERSION_UNFILTERED`        | Neither optional encoding used.                     |
| 1     | `FILE_VERSION_SHUFFLE`           | At least one array carries the byte-shuffle filter. |
| 2     | `FILE_VERSION_FIXED_WIDTH_CELLS` | Cell topology stored without an offsets array.      |

A decoder must also have a ceiling. The C core carries one,
`PVZSTD_FILE_VERSION_MAX = 2`, readable at runtime as
`pvzstd_max_file_version()`, and a container stamped higher is refused with
`PVZSTD_E_VERSION` rather than read: a later version may transform payloads in
a way this build cannot invert, so reading one would hand back
plausible-looking corrupt values instead of failing. The ceiling belongs beside
the decoder it describes -- a binding that kept its own copy would refuse files
the library can read, or accept files it cannot.

The version is promoted by the _writer_ only when it actually uses an optional
encoding — the `ZstdFileMetadata` default is the legacy
`FILE_VERSION_UNFILTERED` (`:220-222`). **A file's version therefore describes
the features it uses, not the version of the library that wrote it.**

**It is a maximum-feature tier, not a generation counter, and the tiers are not
cumulative.** Measured on this build:

| Dataset         | `shuffle=` | version | `fixed_cell_sizes` | filters present |
| --------------- | ---------- | ------- | ------------------ | --------------- |
| mixed hex + tet | `False`    | 0       | `{}`               | no              |
| mixed hex + tet | `True`     | 1       | `{}`               | yes             |
| homogeneous hex | `False`    | 2       | `{"cells": 8}`     | no              |
| homogeneous hex | `True`     | **2**   | `{"cells": 8}`     | **yes**         |

The last row is the hazard: a **version-2 file may or may not carry shuffled
arrays**, because 2 outranks 1 and overwrites it. Any reader that decides
whether to expect a `filter_id` byte by testing `file_version >= 1` will
mis-parse a shuffled version-2 file. **Filter presence must be determined
per-frame from the header length (§3), never from the file version.**

## 3. Per-array frame header

Written by `_pack_array_metadata` (`:639-652`), parsed at `:836-860`:

```
u32   name_len
u8[]  name                (name_len bytes, UTF-8)
u32   ndim
u64   shape[ndim]
u8[16] dtype              numpy dtype.str, space-padded to 16 bytes
u8    filter_id           OPTIONAL -- present only when != 0
```

**Array dtype lives here, in the binary header.** It is `arr.dtype.str`
UTF-8-encoded and `ljust`-padded with spaces to `UID_N_CHAR` (16) bytes
(`:645`), read back with `.strip()` (`:855`). Values are numpy type strings
carrying their own byte order: `<f4`, `<f8`, `<i8`, `<i4`, `|u1`. The JSON
`DataSetMetadata.points_dtype` / `celltypes_dtype` fields are _not_ the
authority for per-array dtype — this field is.

This resolves the dtype question the previous revision of this document left
open. Note how it would have gone wrong: the byte immediately after `shape`
is the first character of the dtype string, and `<` is `0x3C` while `|` is
`0x7C` — both are plausible-looking small integers. A parser that assumed
`filter_id` sat there would read filter ids of 60 and 124, fail the
`0 | 1` check, and _appear_ to detect a corrupt file rather than a wrong
parser.

`filter_id` is `_FILTER_NONE = 0` or `_FILTER_SHUFFLE = 1` (`:142-143`).
**The byte is omitted entirely when the filter is `_FILTER_NONE`** — a
deliberate choice so unfiltered frames stay byte-identical to the legacy
layout (`:646-651`). The reader therefore tests for remaining bytes rather
than the file version (`:857-859`):

```python
filter_id = _FILTER_NONE
if offset < len(meta_buf):
    filter_id = struct.unpack_from("<B", meta_buf, offset)[0]
```

A C++ parser must reproduce exactly this: **header length is the signal**.
After consuming the optional byte, offset should equal the frame length; a
conforming reader should assert that and reject trailing bytes.

An unknown `filter_id` must be a hard error, not a passthrough — reading
filtered bytes as-is silently corrupts the array (`:864-868`).

### 3.1 Name encoding

Array names are prefixed by a 16-character dataset UID (`_make_ds_id`, `:633`,
`f"{id(ds):016x}"`), so a frame name looks like
`00007fb61b52bb80scal_f32__point_data`. Association is carried as a **name
suffix**, not a separate field: `__point_data`, `__cell_data`. Topology and
geometry frames use bare suffix-free names under the same UID prefix —
`points`, `celltypes`, `cells_offset`, `cells_connectivity`, and for PolyData
the `verts_ / lines_ / strips_ / polys_` `_offset` and `_connectivity` pairs.
The two JSON frames are `__ds_metadata` (UID-prefixed) and
`__pyvista_zstd_metadata` (**not** UID-prefixed).

Because the UID derives from a Python object address, it is **not stable across
processes** and must be treated as an opaque token, never parsed for meaning.
Empty topology arrays are still written as full (header, payload) pairs with
`shape=(0,)`.

### 3.2 Every length in this header is file-supplied

The bounds checks a parser writes around §3 are part of conforming to the
format, not an optional hardening pass. `name_len` and `ndim` are 32-bit
fields read out of the file, and a container can declare any value for either.
Three rules, all implemented in `cpp/src/reader.cpp` and exercised by
`tests/conformance/test_crafted_containers.py`:

- **Check against the bytes that remain, by subtracting and dividing** rather
  than by adding to an offset and multiplying. The offset is a `size_t`, 32
  bits on a WebAssembly target, so `off + ndim * 8 + 16` wraps for an `ndim`
  near 2^29 and turns the guard into a pass -- after which the shape loop reads
  eight bytes per dimension past the end of the container.
- **Refuse any field a `size_t` cannot hold** before it reaches an allocator.
  Narrowing a 64-bit declared length on a 32-bit target asks for its low bits
  and leaves the rest of the parse reasoning about the full value.
- **Bound a declared decompressed size by what the frame could possibly
  produce.** The index is file-supplied and unrelated to what the frame holds,
  so a crafted entry can name any 64-bit number. A frame occupies only bytes
  the container has, and zstd can expand each of those by at most 32768 (a
  128 KiB block spelled as a 4-byte RLE block); anything above that product is
  refused. Relying on the allocation to throw instead is not enough: a build
  with exceptions compiled out cannot answer a throw.

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

The filter is **opt-in and off by default**: `shuffle: ShuffleSpec = False` on
both `write` (`:562`) and `Writer.write` (`:752`), and the comment at `:145`
says so outright. Three settings exist, and only one of them is adaptive:

| `shuffle=` | Behaviour                                                 |
| ---------- | --------------------------------------------------------- |
| `False`    | Never shuffle. Default. Every `filter_id` is 0.           |
| `True`     | Always shuffle, unconditionally, every array.             |
| `"auto"`   | Run the trial-compression heuristic below, **per array**. |

Measured on a 40x40x40 grid: `False` -> 1,081,004 B, `True` -> 319,494 B,
`"auto"` -> 732,299 B, with `"auto"` selecting shuffle for the two float
arrays and declining it for an `int32` ramp. So `"auto"` is genuinely
per-array, and it is not simply "the smaller of the two" — it optimises each
array locally, which on this input lands between the two static choices.

When `shuffle="auto"`, the choice is **adaptive**
(`_auto_shuffle_beneficial`, `:159-172`):

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

## 5.1 Compression settings are part of the bytes

Reproducing the layout is not enough to reproduce the file. Two settings feed
straight into the compressed frames:

- **`level`** (default 3), and
- **the worker count**, which the reference derives from the total array bytes:
  `n_threads = floor(total_MiB / 2)`, promoted to `-1` once that exceeds 8
  (`_set_n_threads`, `:406-416`). A negative count means one worker per logical
  CPU.

The worker count is load-bearing because **zstd's multi-threaded mode emits
different bytes than its single-threaded mode**. Measured here on the same two
buffers at level 3: `threads=0` produced 2,846,994 bytes, `threads>=1` produced
2,853,206, and every value `>= 1` agreed with every other. A plain
single-threaded `ZSTD_compress` reproduces the `threads=0` output exactly.

Two consequences worth stating plainly:

1. **The reference writer's own output is a step function of data size.** Under
   2 MiB of arrays the rule yields 0 workers and the single-threaded byte
   stream; above it, the multi-threaded one. Nothing about the file announces
   which.
2. **Files written with a negative worker count are only byte-reproducible on a
   machine with the same CPU count.** That is a property of the existing
   format's writer, not of this port.

MT output is otherwise deterministic -- five repeats at `threads=4` produced
identical bytes -- so byte-identity is achievable in both regimes provided the
libzstd versions match. The C++ writer therefore replicates `_set_n_threads`
and sets `ZSTD_c_nbWorkers` rather than always compressing single-threaded.

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

## 9. The questions that blocked the C++ core, and how they closed

The first revision of this document listed four unresolved questions. Three are
now closed by measurement against a live writer, not by re-reading the source.

1. **Where does array dtype live?** **CLOSED — binary header.** It is
   `arr.dtype.str`, space-padded to 16 bytes, between `shape` and the optional
   filter byte (§3). The JSON `points_dtype` / `celltypes_dtype` fields are not
   the per-array authority.

2. **What does `FILE_VERSION_FIXED_WIDTH_CELLS` change?** **CLOSED — it drops
   the offsets array.** When every cell has the same point count, no
   `cells_offset` frame is written at all and `__ds_metadata.fixed_cell_sizes`
   carries the constant stride (`{"cells": 8}` for an all-hex grid,
   `{"cells": 4}` for all-tet). A mixed hex+tet grid emits a `cells_offset`
   frame and `fixed_cell_sizes == {}`. A reader must reconstruct offsets as
   `arange(n_cells + 1) * stride` when the frame is absent.

3. **Is frame order semantically meaningful?** **CLOSED — yes, load-bearing.**
   Name-to-frame resolution is positional arithmetic
   (`frame_names.index(key) * 2`, `:1315`), so a reader may not reorder frames
   and a writer may not emit them in a different order than `frame_names`
   (§1.2). Confirmed equal across all four dataset classes tested.

4. **What is the append path's contract** when a file is extended —
   specifically whether the index is rewritten in place or appended to.
   **CLOSED — the index is rewritten whole.** An append cannot extend the
   index, because the index is not what sits at the end: the frame count is.
   The committed frames are copied verbatim by offset, never decompressed or
   recompressed, the new frames are written after them, the two metadata frames
   are regenerated, and a fresh full index and count are emitted. Cost is
   therefore proportional to what is added, not to the size of the file.
   `pvzstd_append_arrays` stages that into a new file and commits it by rename;
   `pvzstd_stream_append` does it in place against the open original (§1.1).

   The namespace question the first revision raised resolved the other way.
   Appended arrays are ordinary arrays: their frames join the body and index,
   their names join the file metadata, and they are registered as `field_data`
   on the root dataset under the `…__field_data` suffix, so an ordinary read
   surfaces them with no reader change. What is scoped to them is the _partial
   read_ surface — `pvzstd_field_array_count` / `pvzstd_field_array_name_at` /
   `pvzstd_find_field_array`, and `AppendReader` above them — which lists field
   arrays only, and takes its list from the dataset metadata rather than by
   scanning frame names for the suffix, since an ordinary array may end that
   way too. MultiBlock has no single root dataset to append to and is refused
   with `PVZSTD_E_UNSUPPORTED`.

### 9.1 How these were closed

An independent reader (`tests/conformance/ref_reader.py`) was written **only**
from the byte layout, importing nothing from `pyvista_zstd`, so it could not inherit the
library's assumptions. It reproduces every array bit-exactly (dtype, shape and
contents) for PolyData, UnstructuredGrid, ImageData and StructuredGrid.

That result alone would have been misleading: with the default `shuffle=False`
the filter branch never executed, so a green comparison said nothing about it.
Re-running with `shuffle=True` exercised the branch, and disabling the
unshuffle step as a negative control turned the comparison red on three arrays
— which is what makes the green meaningful. **This reader is the conformance
oracle for the C++ core; the C++ implementation must agree with it, and every
conformance case must include an input that actually reaches the branch under
test.** That is enforced rather than intended: `tests/conformance/` compares
the core against it on every run.

## 10. What this document leaves to the C ABI

A few format-adjacent guarantees belong with the API rather than with the byte
layout, and are specified in `doc/c_api.rst`: `pvzstd_open_memory`'s borrowing
rules, the status enum, the promise that no exception crosses the boundary and
what a WebAssembly build must be compiled with for that to hold, and
`PVZSTD_ABI_VERSION`, which versions the _interface_ and is unrelated to the
container `file_version` versioned in §2.
