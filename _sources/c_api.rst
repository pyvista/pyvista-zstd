The C library
=============

The reader and writer live in a C++ core, ``pvzstd``, and the whole of its
public surface is a C ABI declared in a single header,
``cpp/include/pvzstd/pvzstd.h``. The Python package is one consumer of that
ABI, bound with :mod:`ctypes`; nothing in the ABI is Python-specific, and a C
or C++ consumer can link the same library directly.

Two properties are what make that worth doing:

* **The interface is pure C.** No C++ type, no zstd type, and no VTK type
  appears in the header, so it binds from C, from C++, from ``ctypes``, or from
  any FFI that can call a C function. Turning a dataset into arrays stays with
  the caller, which is what keeps VTK out of the library and lets it build as
  WebAssembly.
* **The only dependency is zstd**, and it is linked privately. The header never
  names a zstd type, so for the default shared build zstd is an implementation
  detail of ``libpvzstd`` rather than something a consumer has to find. (See
  :ref:`c_api_static_caveat` for the one case where that does not hold.)

The on-disk format the library reads and writes is specified in
``doc/format/container-v2.md``.


Building
--------

The core is an ordinary CMake project rooted at ``cpp/``:

.. code:: bash

   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   cmake --install build --prefix /where/you/want/it

The install tree carries the library, the header under ``include/pvzstd/``, and
a CMake package under ``lib/cmake/pvzstd/``.

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Option
     - Default
     - Meaning
   * - ``PVZSTD_BUILD_SHARED``
     - ``ON``
     - Build a shared library. The Python ``ctypes`` binding needs one. A
       static build is possible but exports zstd resolution to the consumer;
       see :ref:`c_api_static_caveat`.
   * - ``PVZSTD_BUILD_TOOLS``
     - ``ON``
     - Build the command-line tools ``pvzstd_dump``, ``pvzstd_rewrite``,
       ``pvzstd_append`` and ``pvzstd_stream``. They are conformance and
       inspection aids, not part of the ABI; turn them off for a build that
       only ships the library.
   * - ``PVZSTD_THREADS``
     - ``ON``
     - Use threads for batch decompression and for zstd's own workers. Turn it
       off for a target with no thread runtime. Reads are unaffected. Writes
       lose byte-identity with the reference writer above its 2 MiB threshold,
       and say so: a zstd built without multithreading rejects a non-zero
       worker count, so ``pvzstd_writer_write`` returns ``PVZSTD_E_ZSTD``
       rather than quietly emitting frames that would not match.
   * - ``PVZSTD_ZSTD_PROVIDER``
     - ``auto``
     - Where zstd comes from. ``auto`` prefers an installed zstd and falls back
       to the pinned source build; ``system`` requires an installed one and
       fails configuration if there is none; ``vendored`` always builds the
       pinned source, ignoring anything installed.
   * - ``PVZSTD_VENDOR_ZSTD``
     - ``ON``
     - Whether the pinned source build is available as a fallback at all. With
       it ``OFF`` and no zstd installed, configuration fails rather than
       downloading anything.

``vendored`` is not paranoia about versions in general: different zstd releases
emit different, equally valid frames for the same input, so the conformance
suite has to pin the same source the reference writer carries. Linking a system
zstd instead was measured to fail the byte-identity tests on multi-frame data
that platforms without a system zstd passed. A distribution build that cares
more about receiving zstd's security updates than about byte-identity should
prefer ``system``.

The pinned source build is also what makes Windows and WebAssembly possible at
all, since neither has a system zstd to find.


Consuming it
------------

Both paths yield the same target name, ``pvzstd::pvzstd``, so the rest of a
consumer's CMake does not have to know which one was taken.

Against an install tree
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cmake

   find_package(pvzstd CONFIG REQUIRED)
   target_link_libraries(my_app PRIVATE pvzstd::pvzstd)

Point ``CMAKE_PREFIX_PATH`` at the install prefix if it is not already on the
search path. The package's version is the released version line
(``find_package(pvzstd 0.3)`` means what a reader expects), and it is compatible
within a major version. It is *not* the compatibility contract a consumer should
rely on at runtime: that is ``PVZSTD_ABI_VERSION``, below.

Against the source
~~~~~~~~~~~~~~~~~~

As a subdirectory -- a submodule, a vendored copy, or a sibling checkout:

.. code:: cmake

   add_subdirectory(third_party/pyvista-zstd/cpp pvzstd)
   target_link_libraries(my_app PRIVATE pvzstd::pvzstd)

or fetched at configure time:

.. code:: cmake

   include(FetchContent)
   FetchContent_Declare(pvzstd
     GIT_REPOSITORY https://github.com/pyvista/pyvista-zstd.git
     GIT_TAG        main
     SOURCE_SUBDIR  cpp)
   FetchContent_MakeAvailable(pvzstd)
   target_link_libraries(my_app PRIVATE pvzstd::pvzstd)

``pvzstd::pvzstd`` is an ``ALIAS`` of the in-tree target in these two cases and
an imported target in the ``find_package`` case, which is the point of having
the alias: one link line serves all three.

.. _c_api_static_caveat:

The static build does not carry zstd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

zstd is linked ``PRIVATE``. For the default shared build that folds zstd's
object code into ``libpvzstd`` itself, so the installed library is
self-contained whether zstd was vendored or found on the system.

It does **not** hold for ``-DPVZSTD_BUILD_SHARED=OFF``. CMake does not fold a
private static dependency's objects into the resulting archive, so
``libpvzstd.a`` still needs zstd's symbols resolved at the consumer's link step,
and the exported package does not declare that dependency -- zstd is treated as
this build's implementation detail, not a re-exported one. With a system zstd a
consumer can supply it themselves. With ``PVZSTD_ZSTD_PROVIDER=vendored`` the
vendored zstd happens to install its own findable package alongside this one, as
a side effect of ``FetchContent`` adding it as a subdirectory rather than
anything this project arranges, so it is not strictly unfindable -- but that is
fragile and not worth designing around. Prefer the shared build for anything
that will be installed and linked downstream.


The ABI version
---------------

.. code:: c

   #define PVZSTD_ABI_VERSION 11u
   PVZSTD_API uint32_t pvzstd_abi_version(void);

The macro is the version the caller compiled against; the function is the
version the library it actually loaded was built with. **Compare them for
equality, not as a floor.** The number is bumped on any change to the header,
additions included, because a caller that binds every symbol up front -- which
is what an FFI binding does -- is broken by an addition just as surely as by a
removal, if it looks for a symbol the loaded library does not have.

.. code:: c

   if (pvzstd_abi_version() != PVZSTD_ABI_VERSION) {
     /* wrong library; do not call anything else */
   }

Calling the function is also the cheapest real proof that a library linked and
loaded at all. A header parses whether or not anything is behind it, and CMake
reporting a successful link says only that the symbols resolved at build time,
not that the shared object the loader picks up at runtime is the right one -- or
that it is present. ``pvzstd_abi_version()`` is a call into that object with no
arguments and no failure mode, so a correct answer from it means the library is
there, is loadable, and is the one the header describes. The Python binding does
exactly this at import time, so that a bad install fails at import naming what
it tried rather than at the first read.

A second, separate number describes the *files* rather than the library:

.. code:: c

   #define PVZSTD_FILE_VERSION_MAX 2u
   PVZSTD_API uint32_t pvzstd_max_file_version(void);

This is the highest container ``file_version`` the build can decode. A container
stamped higher is refused with ``PVZSTD_E_VERSION`` rather than read, because a
newer format may transform payloads in a way this build cannot invert, and
reading one would hand back plausible-looking corrupt values instead of failing.
The ceiling lives beside the decoder it describes so that every caller of the
ABI gets the same answer; a binding keeping its own copy would refuse files the
library can read, or accept files it cannot.


Reading a container
-------------------

A reader is opened, queried, read from, and closed. Opening parses only the
trailer and the array headers; payloads are decompressed on demand, one array at
a time.

.. code:: c

   pvzstd_status pvzstd_open(const char *path, pvzstd_reader **out);
   pvzstd_status pvzstd_open_versioned(const char *path, pvzstd_reader **out,
                                       uint32_t *file_version);
   pvzstd_status pvzstd_open_memory(const void *data, uint64_t size,
                                    pvzstd_reader **out);
   pvzstd_status pvzstd_open_memory_versioned(const void *data, uint64_t size,
                                              pvzstd_reader **out,
                                              uint32_t *file_version);
   void pvzstd_close(pvzstd_reader *reader);

On success ``*out`` receives a reader that must be released with
``pvzstd_close``. On failure ``*out`` is set to ``NULL``. ``pvzstd_close``
accepts ``NULL``.

The ``_versioned`` variants additionally report the container's own
``file_version``, **including when the open was refused for being too new**,
which is precisely the case in which a caller needs the number -- to say what it
found against ``pvzstd_max_file_version()``. The out-parameter may be ``NULL``,
and is left untouched when the container carried no readable version.

Opening from memory
~~~~~~~~~~~~~~~~~~~

``pvzstd_open_memory`` takes ``size`` contiguous bytes of a whole container that
the caller already holds: an archive member, an HTTP response body, or a build
with no filesystem to open a path on. It refuses a crafted buffer wherever
``pvzstd_open`` would refuse a crafted file.

**The bytes are borrowed, never copied.** The caller owns the buffer and must
keep it allocated and unmodified until ``pvzstd_close``, which is what makes
this cheaper than staging the container somewhere the path entry point can
reach. ``pvzstd_close`` releases only what the reader itself acquired; the
buffer is left alone and is the caller's to free afterwards.

Modifying the buffer while a reader is open **is not detected**. Offsets and
sizes were read at open time, so arrays read afterwards hold undefined contents
and no status reports it.

One difference between the two doors is deliberate. A ``NULL`` pointer, or a
``size`` of zero, is ``PVZSTD_E_INVALID``: a zero size is a bad argument. A
zero-length *file* is a property of the thing being opened and so reaches
``pvzstd_open`` as ``PVZSTD_E_IO``. A caller writing one error handler over both
entry points should expect that.

Describing arrays
~~~~~~~~~~~~~~~~~

.. code:: c

   typedef struct pvzstd_array_info {
     const char *name;      /* NUL-terminated, UTF-8, includes the UID prefix */
     const uint64_t *shape; /* ndim entries; NULL when ndim == 0 */
     uint32_t ndim;
     uint8_t filter_id;
     char dtype[PVZSTD_DTYPE_LEN + 1]; /* e.g. "<f8", "|u1"; NUL-terminated */
     uint64_t nbytes;                  /* decompressed payload size */
   } pvzstd_array_info;

   uint64_t      pvzstd_array_count(const pvzstd_reader *reader);
   pvzstd_status pvzstd_array_info_at(const pvzstd_reader *reader, uint64_t index,
                                      pvzstd_array_info *out);
   pvzstd_status pvzstd_array_info_range(const pvzstd_reader *reader, uint64_t first,
                                         uint64_t count, pvzstd_array_info *out);
   int64_t       pvzstd_find_array(const pvzstd_reader *reader, const char *name);

Arrays are addressed by index in frame order, or by name. Frame order is
significant: it is how the container maps names to frames. ``pvzstd_array_count``
excludes the two JSON metadata frames. ``pvzstd_find_array`` returns ``-1`` when
there is no such name; note that stored names carry a 16-character dataset-UID
prefix, so a caller matching a logical name usually compares suffixes rather than
whole strings.

The pointers in ``pvzstd_array_info`` **are owned by the reader** and stay valid
only until ``pvzstd_close``. ``dtype`` is an inline array, so it is copied into
the caller's struct; ``name`` and ``shape`` are not. A caller that outlives the
reader must copy them.

``pvzstd_array_info_range`` describes ``count`` arrays in one boundary crossing
rather than one per array, which matters for a binding where each call is
expensive. It returns ``PVZSTD_E_RANGE`` and writes nothing if the range runs
past the end; a count of zero succeeds.

Reading payloads
~~~~~~~~~~~~~~~~

.. code:: c

   pvzstd_status pvzstd_read_array_at(const pvzstd_reader *reader, uint64_t index,
                                      void *dst, uint64_t dst_size);
   pvzstd_status pvzstd_read_arrays(const pvzstd_reader *reader,
                                    const uint64_t *indices, uint64_t count,
                                    void *const *dsts, const uint64_t *dst_sizes,
                                    int n_threads, uint64_t *failed_slot);

The destination is the caller's, always. ``dst_size`` must be at least the
``nbytes`` the array's info reported, or ``PVZSTD_E_RANGE`` comes back and
nothing is written. Any per-array filter is reversed on the way out; an unknown
filter id is ``PVZSTD_E_FILTER``, never a passthrough, because handing back
filtered bytes as-is would silently corrupt the array.

``pvzstd_read_arrays`` decompresses several arrays over ``n_threads`` workers,
``indices[i]`` into ``dsts[i]``. Frames are independent, so this is a pure
fan-out: the work and its order do not depend on the thread setting. ``0`` or
``1`` runs inline, a negative value means one worker per logical CPU, and
``PVZSTD_THREADS_AUTO`` picks from the total size. It returns the first non-OK
status *by slot*, not by whichever worker failed first, so the result is
deterministic. ``failed_slot`` is optional and receives the index into
``indices`` that the status is about, or ``PVZSTD_SLOT_NONE``; without it a
caller learns only that one of the batch was refused and has to re-read them
singly to find out which.

Metadata and frame sizes
~~~~~~~~~~~~~~~~~~~~~~~~

``pvzstd_ds_metadata_json`` and ``pvzstd_file_metadata_json`` return the two JSON
documents, NUL-terminated and owned by the reader, either possibly ``NULL``. For
a MultiBlock container the dataset-metadata accessor reports an arbitrary
block's, so a caller rebuilding a hierarchy should use the by-index family --
``pvzstd_metadata_count``, ``pvzstd_metadata_name_at``,
``pvzstd_metadata_json_at`` -- which yields every document in file order along
with the frame name each was stored under.

``pvzstd_field_array_count``, ``pvzstd_field_array_name_at`` and
``pvzstd_find_field_array`` cover the field arrays an append adds, named without
the UID prefix or the ``__field_data`` suffix the frame carries.

``pvzstd_frame_count`` and ``pvzstd_frame_sizes`` expose the trailer's own
numbers -- two frames per array, ``(header, payload)``, in file order, metadata
frames included -- so a caller can answer "how big is this file decompressed"
without parsing the trailer itself.


Errors
------

**Every entry point reports failure as a status code**, and no exception is ever
allowed to cross the boundary: a caller reaching this ABI through ``ctypes`` or
another FFI has no way to catch one. Accessors that return a pointer or a count
instead of a status report failure as ``NULL`` or as zero, which are the values
they already use for "cannot answer". ``pvzstd_status_message`` maps a status to
a static, human-readable string and is never ``NULL``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Status
     - Meaning
   * - ``PVZSTD_OK``
     - Success.
   * - ``PVZSTD_E_IO``
     - File missing, unreadable, or truncated.
   * - ``PVZSTD_E_FORMAT``
     - Trailer or header did not parse.
   * - ``PVZSTD_E_ZSTD``
     - zstd rejected a frame or a compression parameter.
   * - ``PVZSTD_E_RANGE``
     - Index or count out of range, or destination too small.
   * - ``PVZSTD_E_NOMEM``
     - Allocation failed.
   * - ``PVZSTD_E_FILTER``
     - A per-array filter id this build cannot reverse.
   * - ``PVZSTD_E_INVALID``
     - ``NULL`` argument or misuse.
   * - ``PVZSTD_E_UNSUPPORTED``
     - The container is a shape this operation cannot serve.
   * - ``PVZSTD_E_EXISTS``
     - The name is already taken, and would be overwritten.
   * - ``PVZSTD_E_VERSION``
     - The container's ``file_version`` is newer than this build decodes.
   * - ``PVZSTD_E_BUSY``
     - Another append holds this container, or left its lock file behind.
   * - ``PVZSTD_E_CHANGED``
     - The container was replaced while this call was staging its result.

Everything from ``PVZSTD_E_UNSUPPORTED`` down is a refusal rather than damage:
the file parsed, and the operation is the thing being declined. A caller that
cannot tell them from ``PVZSTD_E_FORMAT`` has to report a well-formed container
as corrupt. The last two are further worth telling apart from ``PVZSTD_E_IO``,
because nothing is wrong with the file or the disk and the call is worth making
again -- see :ref:`c_api_append_lock`.

.. _c_api_wasm_exceptions:

WebAssembly must catch exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The promise above is implemented: every ``extern "C"`` entry point is a function
try block ending in ``catch (...)``. That machinery has to survive the build.
Emscripten defaults to ``-sDISABLE_EXCEPTION_CATCHING=1``, which compiles every
one of those handlers away and turns a throw into ``abort()`` -- so on that
target the header's promise is not kept, and a caller parsing untrusted input
gets a dead process instead of ``PVZSTD_E_NOMEM``.

This project's own CMake therefore adds ``-fwasm-exceptions`` at compile and at
link when ``EMSCRIPTEN`` is set, and adds it ``PUBLIC``: a wasm link cannot mix
exception models, so a consumer compiling its own translation units against this
library has to be on the same one. It is scoped to the target rather than set
globally, so a project embedding this library keeps its own flags everywhere
else. ``-fwasm-exceptions`` rather than the legacy JavaScript-based unwinder,
which is slower, larger, and taxes every call site whether or not anything
throws.

A WebAssembly consumer that builds this library through some other build system
must reproduce that flag, or the ABI silently stops returning statuses for the
failures that matter most.


Writing and appending
---------------------

The reading surface above is the one most consumers need. The header also
carries three write-side surfaces, documented in place:

* ``pvzstd_writer_*`` -- the container layer only. It takes arrays the caller
  has already produced and emits the file; turning a dataset into arrays stays
  with the caller. Reproducing the reference writer byte for byte needs the same
  compression level *and* worker count, because zstd's threaded mode emits
  different, equally valid bytes.
* ``pvzstd_append_arrays`` -- add field arrays without rewriting the
  container. Existing frames are copied by offset and never decompressed, so the
  cost is what is added rather than the file size. Each call commits by rename,
  so an interrupted append cannot damage what was there. It also takes a lock;
  see :ref:`c_api_append_lock` below, which is the one thing on this page that
  can leave a container needing a human.
* ``pvzstd_stream_*`` -- the same edit with the parsed state held open across
  commits, so per-commit cost is flat in container size rather than growing with
  it. The trade is crash behaviour: an interrupted stream commit leaves a trailer
  describing frames that were not fully written. Use ``pvzstd_append_arrays``
  when every commit must leave a valid file.

.. _c_api_append_lock:

One append at a time
~~~~~~~~~~~~~~~~~~~~

An append is a read-modify-write: it reads the container, adds to what it read,
and commits the result. Two of them running at once each commit "what was there
plus mine", and the second to land replaces the first's arrays with a body
copied before those arrays existed -- with both callers told they succeeded.
Noticing that at the end does not work either: two appends doing equal work
reach their commits within microseconds of each other, so neither one's check
sees the other's result yet. Measured on four concurrent appends, three sets of
arrays were lost in every run and every caller reported success.

``pvzstd_append_arrays`` therefore takes an advisory lock for the duration of
the call: a file named ``<path>.append.lock``, beside the container, created
exclusively. A second append meanwhile returns ``PVZSTD_E_BUSY`` **immediately,
rather than waiting.** It does not block, because a library has no business
choosing how long its caller waits; retrying is the caller's decision, and a
retry succeeds as soon as the other append finishes.

Exclusive file creation is the primitive rather than ``flock`` because it is the
one form of mutual exclusion every target this builds for implements the same
way. On the WebAssembly target ``flock`` returns success without locking
anything, so a second holder would be handed a lock the first one believes it
has.

.. warning::

   **A killed append leaves its lock behind, and a human has to delete it.**
   The lock file is removed when the call returns, however it returns -- but a
   process that dies between taking it and finishing never gets to. Every later
   append to that container then returns ``PVZSTD_E_BUSY`` until the file is
   removed, and no retry clears it.

   Recovery is deleting ``<path>.append.lock``. That is safe whenever no append
   is actually running against the container; the file carries no state, and the
   container itself was never touched by the append that died. The trade is
   deliberate: this is a visible, named, recoverable failure, where what it
   replaces was one writer's arrays disappearing with nothing reported.

The lock binds appends and nothing else. A writer that does not take it --
another tool, or a plain ``mv`` onto the path -- is caught separately: each
append stages into a file the operating system names, so no two callers are ever
handed the same staging file, and before committing, a call checks that its path
still names the container it read. One replaced during the staging returns
``PVZSTD_E_CHANGED`` having written nothing, and running it again picks up the
other writer's result. That one is a check and not a lock: it covers the
staging, which is the slow part, and not the microseconds between the check and
the rename.

Streams take no lock and are covered by none. ``pvzstd_stream_*`` writes into
the container in place, so it has neither a commit point at which to notice
another writer nor a staging file to hold back, and it returns neither
``PVZSTD_E_BUSY`` nor ``PVZSTD_E_CHANGED``. One stream at a time, and no append
against the same container while a stream is open, is the caller's to arrange.

On WebAssembly
^^^^^^^^^^^^^^

The lock is real on that target, not inert: exclusive creation works under both
Emscripten's own filesystems and ``NODERAWFS``, and a stale lock refuses a later
append there exactly as it does natively. What it cannot do is protect a
consumer against itself. A single-threaded module with no other writer never
contends, so the lock costs it two filesystem operations per append and nothing
else; two module instances sharing one real file through ``NODERAWFS`` are two
writers and are excluded properly. Two instances over separate in-memory
filesystems are not sharing a container at all, so there is nothing to exclude
-- and if such a build ever loses a lock file to a discarded filesystem, the
container is discarded with it.


A worked example
----------------

Open a container, read its first array, close. This is complete: it compiles as
C, links against ``pvzstd::pvzstd``, and runs.

.. code:: c

   #include <pvzstd/pvzstd.h>

   #include <stdio.h>
   #include <stdlib.h>

   int main(int argc, char **argv) {
     if (argc != 2) {
       fprintf(stderr, "usage: %s CONTAINER.pv\n", argv[0]);
       return 2;
     }

     /* Proves a library actually linked, and that it is the one this header
        describes: the two are compared for equality, not as a floor. */
     if (pvzstd_abi_version() != PVZSTD_ABI_VERSION) {
       fprintf(stderr, "pvzstd ABI mismatch: built against %u, linked %u\n",
               (unsigned)PVZSTD_ABI_VERSION, pvzstd_abi_version());
       return 1;
     }

     pvzstd_reader *reader = NULL;
     uint32_t file_version = 0;
     pvzstd_status status = pvzstd_open_versioned(argv[1], &reader, &file_version);
     if (status != PVZSTD_OK) {
       fprintf(stderr, "open: %s (container version %u, this build decodes %u)\n",
               pvzstd_status_message(status), file_version, pvzstd_max_file_version());
       return 1;
     }

     if (pvzstd_array_count(reader) == 0) {
       fprintf(stderr, "container holds no arrays\n");
       pvzstd_close(reader);
       return 1;
     }

     pvzstd_array_info info;
     status = pvzstd_array_info_at(reader, 0, &info);
     if (status != PVZSTD_OK) {
       fprintf(stderr, "info: %s\n", pvzstd_status_message(status));
       pvzstd_close(reader);
       return 1;
     }

     void *dst = malloc((size_t)info.nbytes);
     if (dst == NULL && info.nbytes != 0) {
       pvzstd_close(reader);
       return 1;
     }

     status = pvzstd_read_array_at(reader, 0, dst, info.nbytes);
     if (status != PVZSTD_OK) {
       fprintf(stderr, "read: %s\n", pvzstd_status_message(status));
       free(dst);
       pvzstd_close(reader);
       return 1;
     }

     printf("%s: dtype %s, ndim %u, %llu bytes\n", info.name, info.dtype,
            (unsigned)info.ndim, (unsigned long long)info.nbytes);

     free(dst);            /* the destination is the caller's */
     pvzstd_close(reader); /* info.name and info.shape die here */
     return 0;
   }

with

.. code:: cmake

   cmake_minimum_required(VERSION 3.16)
   project(pvzstd_example LANGUAGES C)

   find_package(pvzstd CONFIG REQUIRED)

   add_executable(read_one read_one.c)
   target_link_libraries(read_one PRIVATE pvzstd::pvzstd)

Against a container written by ``pyvista_zstd.write``, it prints something like::

   00007c31d6209960points: dtype <f4, ndim 2, 10104 bytes

The UID prefix on the name is the dataset UID the format carries. It derives
from an object address in the writing process, so it is not stable across
processes and must be treated as an opaque token, never parsed for meaning.
