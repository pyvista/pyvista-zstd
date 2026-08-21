// Reader half of the .pv container. See doc/format/container-v2.md.
//
// The parsing rules that are easy to get wrong, and are therefore enforced
// rather than assumed:
//
//   * index entries carry each frame's END offset, so starts are derived by
//     shifting; frame 0 begins at byte 0 and there is no magic number;
//   * frames pair as (header, payload), so the count is even;
//   * the header's optional filter byte is present only when non-zero, which
//     makes header length -- not the file version -- the signal.
//
// Each frame's declared decompressed size is checked against what zstd
// actually produces. That check is what turns a misparsed index from silent
// data corruption into an error.

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "detail.h"
#include "json_read.h"
#include "pvzstd/pvzstd.h"
#include "threads.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr uint64_t kTrailerCountBytes = 8;
constexpr uint64_t kIndexEntryBytes = 16;
constexpr char kDsMetadataSuffix[] = "__ds_metadata";
constexpr char kFileMetadataSuffix[] = "__pyvista_zstd_metadata";
constexpr char kMultiblockSuffix[] = "__multiblock__ds_metadata";
constexpr char kFieldDataSuffix[] = "__field_data";
constexpr size_t kUidNChar = 16;

uint64_t LoadU64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[static_cast<size_t>(i)];
  return v;
}

uint32_t LoadU32(const uint8_t *p) {
  uint32_t v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | p[static_cast<size_t>(i)];
  return v;
}

bool EndsWith(const std::string &s, const char *suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

struct ArrayEntry {
  std::string name;
  std::vector<uint64_t> shape;
  char dtype[PVZSTD_DTYPE_LEN + 1];
  uint8_t filter_id;
  uint64_t nbytes;         // decompressed payload size
  uint64_t payload_start;  // byte range of the compressed payload frame
  uint64_t payload_end;
};

// Invert the byte-plane split applied by the shuffle filter: the on-disk form
// is itemsize planes of n_elem bytes, and the array wants them interleaved.
void Unshuffle(const uint8_t *src, uint8_t *dst, uint64_t nbytes, uint64_t itemsize) {
  const uint64_t n_elem = nbytes / itemsize;
  for (uint64_t plane = 0; plane < itemsize; ++plane) {
    const uint8_t *in = src + plane * n_elem;
    uint8_t *out = dst + plane;
    for (uint64_t i = 0; i < n_elem; ++i) out[i * itemsize] = in[i];
  }
}

// Byte width implied by a numpy dtype string such as "<f8" or "|u1", or 0 when
// this build cannot read one off the tag. Shared with the writer so the two
// halves cannot disagree about a stride.
uint64_t DtypeItemsize(const char *dtype) {
  const pvzstd::detail::Dtype d = pvzstd::detail::ParseDtype(dtype);
  return d.valid ? d.itemsize : 0;
}

// Whether the payload size the trailer declares agrees with the shape and
// dtype the header announces.
//
// Nothing else checks this. pvzstd_read_array_at honours the declared payload
// size, while a caller sizes its destination from the shape and dtype, so a
// header saying "10 float64" over a payload declaring 8000 bytes writes 8000
// bytes into an 80-byte destination. That is a buffer overrun produced by a
// file, which makes refusing it the reader's job and not the caller's.
//
// A dtype whose width cannot be read off the tag is left unchecked rather than
// rejected: the format carries whatever numpy spelled, and refusing a
// spelling this build does not recognise would reject a valid file.
bool DeclaredSizeAgrees(const std::vector<uint64_t> &shape, const char *dtype, uint64_t nbytes) {
  uint64_t n = DtypeItemsize(dtype);
  if (n == 0) return true;
  for (const uint64_t dim : shape) {
    if (dim != 0 && n > UINT64_MAX / dim) return false;
    n *= dim;
  }
  return n == nbytes;
}

// Total decompressed bytes below which AUTO decompresses inline instead of
// spawning workers.
//
// This is a floor, not a tuned optimum. Thread-spawn cost is a property of the
// machine, so the exact crossover moves; what does not move is that below some
// size the spawn dominates, and that picking workers from frame *count* alone
// ignores this entirely. Measured here: a 10 KB / 11-frame file ran 2.5x
// slower with one thread per frame than inline, while a 2.7 MB file was
// faster threaded. 4 MiB sits above the observed crossover with margin, so
// the branch errs toward inline only where the work is demonstrably tiny.
constexpr uint64_t kParallelDecompressFloor = 4ull << 20;

// A read-only view of the file.
//
// Mapping rather than reading matters more than it looks: the reference
// reader mmaps, and copying the file into a buffer at open cost 7.85 ms on a
// 21 MB container here -- enough to turn a decompression win into an
// end-to-end loss. Pages are also faulted in only for the frames actually
// read, so a selective read never touches the rest of the file.
class FileMapping {
 public:
  FileMapping() = default;
  ~FileMapping() { Reset(); }
  FileMapping(const FileMapping &) = delete;
  FileMapping &operator=(const FileMapping &) = delete;

  pvzstd_status Open(const char *path) {
    Reset();
#if defined(_WIN32)
    handle_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) return PVZSTD_E_IO;
    LARGE_INTEGER li;
    if (GetFileSizeEx(handle_, &li) == 0 || li.QuadPart <= 0) {
      Reset();
      return PVZSTD_E_IO;
    }
    size_ = static_cast<uint64_t>(li.QuadPart);
    mapping_ = CreateFileMappingA(handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_ == nullptr) {
      Reset();
      return PVZSTD_E_IO;
    }
    data_ = static_cast<const uint8_t *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (data_ == nullptr) {
      Reset();
      return PVZSTD_E_IO;
    }
#else
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) return PVZSTD_E_IO;
    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
      Reset();
      return PVZSTD_E_IO;
    }
    size_ = static_cast<uint64_t>(st.st_size);
    void *addr = ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr == MAP_FAILED) {
      Reset();
      return PVZSTD_E_IO;
    }
    data_ = static_cast<const uint8_t *>(addr);
#endif
    return PVZSTD_OK;
  }

  const uint8_t *data() const { return data_; }
  uint64_t size() const { return size_; }

 private:
  void Reset() {
#if defined(_WIN32)
    if (data_ != nullptr) UnmapViewOfFile(data_);
    if (mapping_ != nullptr) CloseHandle(mapping_);
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    mapping_ = nullptr;
    handle_ = INVALID_HANDLE_VALUE;
#else
    if (data_ != nullptr) ::munmap(const_cast<uint8_t *>(data_), static_cast<size_t>(size_));
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    data_ = nullptr;
    size_ = 0;
  }

  const uint8_t *data_ = nullptr;
  uint64_t size_ = 0;
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace

struct pvzstd_reader {
  FileMapping map;
  std::vector<ArrayEntry> arrays;
  std::string ds_metadata;
  std::string file_metadata;
  bool has_ds_metadata = false;
  bool has_file_metadata = false;
  // Field-data blocks of the root dataset, in the order the dataset metadata
  // lists them. Empty for a MultiBlock container, which has no single root.
  std::vector<std::string> field_names;
  std::vector<int64_t> field_indices;  // into `arrays`; -1 if the frame is gone
  // Filled in by pvzstd_array_info_at so the caller sees stable pointers.
  mutable std::vector<uint64_t> scratch_shape;
};

namespace {

// A decompression context owned by, and reused across every frame decompressed
// on, the calling thread.
//
// zstd's own guidance: "When decompressing many times, it is recommended to
// allocate a context only once, and reuse it for each successive compression
// operation" and "Use one context per thread for parallel execution."
// `thread_local` satisfies both without putting a context handle in the C ABI,
// which matters because the ABI is the part we cannot revise later.
//
// A container is many small frames, so the one-shot ZSTD_decompress() this
// replaces was paying context setup per frame rather than per read.
//
// Returns nullptr if the context could not be allocated; callers fall back to
// the one-shot entry point, which is correct, just slower.
ZSTD_DCtx *ThreadDCtx() {
  struct Holder {
    ZSTD_DCtx *ctx = ZSTD_createDCtx();
    Holder() = default;
    ~Holder() {
      if (ctx != nullptr) ZSTD_freeDCtx(ctx);
    }
    Holder(const Holder &) = delete;
    Holder &operator=(const Holder &) = delete;
  };
  static thread_local Holder holder;
  return holder.ctx;
}

// ZSTD_decompressDCtx on the thread's context, falling back to the one-shot
// call when no context is available. Same result either way.
size_t DecompressInto(void *dst, size_t dst_capacity, const void *src, size_t src_size) {
  ZSTD_DCtx *const ctx = ThreadDCtx();
  if (ctx == nullptr) return ZSTD_decompress(dst, dst_capacity, src, src_size);
  return ZSTD_decompressDCtx(ctx, dst, dst_capacity, src, src_size);
}

pvzstd_status DecompressFrame(const uint8_t *src, uint64_t src_size, uint64_t expected,
                              std::vector<uint8_t> *out) {
  try {
    out->resize(static_cast<size_t>(expected));
  } catch (const std::bad_alloc &) {
    return PVZSTD_E_NOMEM;
  }
  if (expected == 0) return PVZSTD_OK;
  const size_t got = DecompressInto(out->data(), out->size(), src, static_cast<size_t>(src_size));
  if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
  // A mismatch here is the signature of a misparsed index: every frame still
  // decompresses, but each one yields a neighbour's payload.
  if (got != expected) return PVZSTD_E_FORMAT;
  return PVZSTD_OK;
}

pvzstd_status ParseHeader(const std::vector<uint8_t> &buf, ArrayEntry *entry) {
  size_t off = 0;
  if (buf.size() < 4) return PVZSTD_E_FORMAT;
  const uint32_t name_len = LoadU32(buf.data());
  off += 4;
  if (buf.size() < off + name_len + 4) return PVZSTD_E_FORMAT;
  entry->name.assign(reinterpret_cast<const char *>(buf.data() + off), name_len);
  off += name_len;

  const uint32_t ndim = LoadU32(buf.data() + off);
  off += 4;
  if (buf.size() < off + static_cast<size_t>(ndim) * 8 + PVZSTD_DTYPE_LEN) return PVZSTD_E_FORMAT;
  entry->shape.clear();
  for (uint32_t i = 0; i < ndim; ++i) {
    entry->shape.push_back(LoadU64(buf.data() + off));
    off += 8;
  }

  // dtype is space-padded to 16 bytes; strip trailing blanks.
  size_t dtype_len = PVZSTD_DTYPE_LEN;
  while (dtype_len > 0 && buf[off + dtype_len - 1] == ' ') --dtype_len;
  std::memcpy(entry->dtype, buf.data() + off, dtype_len);
  entry->dtype[dtype_len] = '\0';
  off += PVZSTD_DTYPE_LEN;

  // The filter byte is written only when a filter is in use, so its absence
  // means PVZSTD_FILTER_NONE. Testing file_version here instead would mis-parse
  // a shuffled version-2 file.
  entry->filter_id = PVZSTD_FILTER_NONE;
  if (off < buf.size()) {
    entry->filter_id = buf[off];
    off += 1;
  }
  if (off != buf.size()) return PVZSTD_E_FORMAT;
  return PVZSTD_OK;
}

}  // namespace

extern "C" {

pvzstd_status pvzstd_open(const char *path, pvzstd_reader **out) {
  if (path == nullptr || out == nullptr) return PVZSTD_E_INVALID;
  *out = nullptr;

  pvzstd_reader *reader = new (std::nothrow) pvzstd_reader();
  if (reader == nullptr) return PVZSTD_E_NOMEM;

  pvzstd_status st = reader->map.Open(path);
  if (st != PVZSTD_OK) {
    delete reader;
    return st;
  }

  const FileMapping &raw = reader->map;
  if (raw.size() < kTrailerCountBytes) {
    delete reader;
    return PVZSTD_E_FORMAT;
  }

  const uint64_t n_frames = LoadU64(raw.data() + raw.size() - kTrailerCountBytes);
  if (n_frames == 0 || (n_frames % 2) != 0) {
    delete reader;  // frames pair as (header, payload)
    return PVZSTD_E_FORMAT;
  }
  // Bounded by division rather than by comparing against n_frames * 16: the
  // count is read from the file, and the product overflows for a large enough
  // value, which turns a size check into a pass. The vectors below are sized
  // from this count, so an unchecked one is an allocation the file never
  // justified -- pvzstd_append_arrays already guards this way.
  if (n_frames > (raw.size() - kTrailerCountBytes) / kIndexEntryBytes) {
    delete reader;
    return PVZSTD_E_FORMAT;
  }
  const uint64_t index_bytes = n_frames * kIndexEntryBytes;
  const uint64_t index_off = raw.size() - kTrailerCountBytes - index_bytes;

  std::vector<uint64_t> ends(static_cast<size_t>(n_frames));
  std::vector<uint64_t> sizes(static_cast<size_t>(n_frames));
  for (uint64_t i = 0; i < n_frames; ++i) {
    const uint8_t *p = raw.data() + index_off + i * kIndexEntryBytes;
    ends[static_cast<size_t>(i)] = LoadU64(p);  // END offset, not start
    sizes[static_cast<size_t>(i)] = LoadU64(p + 8);
  }

  // The root dataset's UID and metadata, recorded as the frames go past. A
  // MultiBlock container has no single root, so seeing its metadata frame
  // abandons the field-array index rather than guessing which block owns it.
  std::string ds_id;
  std::string root_ds_json;
  bool multiblock = false;

  std::vector<uint8_t> frame;
  for (uint64_t i = 0; i + 1 < n_frames; i += 2) {
    const uint64_t hdr_start = (i == 0) ? 0 : ends[static_cast<size_t>(i - 1)];
    const uint64_t hdr_end = ends[static_cast<size_t>(i)];
    const uint64_t pay_end = ends[static_cast<size_t>(i + 1)];
    if (hdr_start > hdr_end || hdr_end > pay_end || pay_end > index_off) {
      delete reader;
      return PVZSTD_E_FORMAT;
    }

    st = DecompressFrame(raw.data() + hdr_start, hdr_end - hdr_start, sizes[static_cast<size_t>(i)],
                         &frame);
    if (st != PVZSTD_OK) {
      delete reader;
      return st;
    }

    ArrayEntry entry;
    st = ParseHeader(frame, &entry);
    if (st != PVZSTD_OK) {
      delete reader;
      return st;
    }
    entry.nbytes = sizes[static_cast<size_t>(i + 1)];
    entry.payload_start = hdr_end;
    entry.payload_end = pay_end;
    if (!DeclaredSizeAgrees(entry.shape, entry.dtype, entry.nbytes)) {
      delete reader;
      return PVZSTD_E_FORMAT;
    }

    const bool is_ds = EndsWith(entry.name, kDsMetadataSuffix);
    const bool is_file = EndsWith(entry.name, kFileMetadataSuffix);
    if (is_ds || is_file) {
      std::vector<uint8_t> payload;
      st = DecompressFrame(raw.data() + entry.payload_start,
                           entry.payload_end - entry.payload_start, entry.nbytes, &payload);
      if (st != PVZSTD_OK) {
        delete reader;
        return st;
      }
      std::string json(reinterpret_cast<const char *>(payload.data()), payload.size());
      if (is_ds) {
        reader->ds_metadata = json;
        reader->has_ds_metadata = true;
        if (EndsWith(entry.name, kMultiblockSuffix)) {
          multiblock = true;
        } else if (ds_id.empty() && entry.name.size() >= kUidNChar) {
          ds_id = entry.name.substr(0, kUidNChar);
          root_ds_json = json;
        }
      } else {
        reader->file_metadata = json;
        reader->has_file_metadata = true;
      }
      continue;
    }
    reader->arrays.push_back(entry);
  }

  // The field-array index. Its names come from the dataset metadata rather
  // than from the frame names, because that document is what defines which
  // blocks are field data and in which order -- a frame-name scan would also
  // pick up an array whose name merely happens to end the same way.
  if (!multiblock && !ds_id.empty()) {
    size_t fdk_open = 0;
    size_t fdk_past = 0;
    std::vector<std::string> keys;
    if (pvzstd::json::MemberObjectSpan(root_ds_json, "field_data_keys", &fdk_open, &fdk_past) &&
        pvzstd::json::ObjectKeys(root_ds_json, fdk_open, &keys)) {
      for (const std::string &key : keys) {
        const std::string frame_name = ds_id + key + kFieldDataSuffix;
        int64_t found = -1;
        for (size_t k = 0; k < reader->arrays.size(); ++k) {
          if (reader->arrays[k].name == frame_name) {
            found = static_cast<int64_t>(k);
            break;
          }
        }
        // A key with no frame is kept, not dropped: the reference reader lists
        // it too, and refuses only when it is actually read. Reporting a
        // shorter list here would hide the desync instead of surfacing it.
        reader->field_names.push_back(key);
        reader->field_indices.push_back(found);
      }
    }
  }

  *out = reader;
  return PVZSTD_OK;
}

void pvzstd_close(pvzstd_reader *reader) { delete reader; }

uint64_t pvzstd_array_count(const pvzstd_reader *reader) {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->arrays.size());
}

namespace {

// Every pointer here aliases storage the reader owns for its whole life, so
// the filled struct stays valid until pvzstd_close() -- nothing is copied but the
// dtype tag, which is a fixed-size array inside the struct.
void FillArrayInfo(const ArrayEntry &e, pvzstd_array_info *out) {
  out->name = e.name.c_str();
  out->shape = e.shape.empty() ? nullptr : e.shape.data();
  out->ndim = static_cast<uint32_t>(e.shape.size());
  out->filter_id = e.filter_id;
  std::memcpy(out->dtype, e.dtype, sizeof(out->dtype));
  out->nbytes = e.nbytes;
}

}  // namespace

pvzstd_status pvzstd_array_info_at(const pvzstd_reader *reader, uint64_t index,
                                   pvzstd_array_info *out) {
  if (reader == nullptr || out == nullptr) return PVZSTD_E_INVALID;
  if (index >= reader->arrays.size()) return PVZSTD_E_RANGE;
  FillArrayInfo(reader->arrays[static_cast<size_t>(index)], out);
  return PVZSTD_OK;
}

pvzstd_status pvzstd_array_info_range(const pvzstd_reader *reader, uint64_t first, uint64_t count,
                                      pvzstd_array_info *out) {
  if (reader == nullptr) return PVZSTD_E_INVALID;
  if (count == 0) return PVZSTD_OK;
  if (out == nullptr) return PVZSTD_E_INVALID;

  const uint64_t total = static_cast<uint64_t>(reader->arrays.size());
  // Checked against the remaining count rather than as first + count, which
  // would wrap for a large first and silently accept an out-of-range span.
  if (first > total || count > total - first) return PVZSTD_E_RANGE;

  for (uint64_t i = 0; i < count; ++i) {
    FillArrayInfo(reader->arrays[static_cast<size_t>(first + i)], &out[i]);
  }
  return PVZSTD_OK;
}

uint64_t pvzstd_field_array_count(const pvzstd_reader *reader) {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->field_names.size());
}

const char *pvzstd_field_array_name_at(const pvzstd_reader *reader, uint64_t index) {
  if (reader == nullptr || index >= reader->field_names.size()) return nullptr;
  return reader->field_names[static_cast<size_t>(index)].c_str();
}

int64_t pvzstd_find_field_array(const pvzstd_reader *reader, const char *name) {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->field_names.size(); ++i) {
    if (reader->field_names[i] == name) return reader->field_indices[i];
  }
  return -1;
}

int64_t pvzstd_find_array(const pvzstd_reader *reader, const char *name) {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->arrays.size(); ++i) {
    if (reader->arrays[i].name == name) return static_cast<int64_t>(i);
  }
  return -1;
}

pvzstd_status pvzstd_read_array_at(const pvzstd_reader *reader, uint64_t index, void *dst,
                                   uint64_t dst_size) {
  if (reader == nullptr || dst == nullptr) return PVZSTD_E_INVALID;
  if (index >= reader->arrays.size()) return PVZSTD_E_RANGE;
  const ArrayEntry &e = reader->arrays[static_cast<size_t>(index)];
  if (dst_size < e.nbytes) return PVZSTD_E_RANGE;
  if (e.nbytes == 0) return PVZSTD_OK;

  const uint8_t *src = reader->map.data() + e.payload_start;
  const uint64_t src_size = e.payload_end - e.payload_start;

  if (e.filter_id == PVZSTD_FILTER_NONE) {
    const size_t got =
        DecompressInto(dst, static_cast<size_t>(dst_size), src, static_cast<size_t>(src_size));
    if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
    return got == e.nbytes ? PVZSTD_OK : PVZSTD_E_FORMAT;
  }

  if (e.filter_id != PVZSTD_FILTER_SHUFFLE) return PVZSTD_E_FILTER;

  const uint64_t itemsize = DtypeItemsize(e.dtype);
  if (itemsize == 0 || (e.nbytes % itemsize) != 0) return PVZSTD_E_FORMAT;

  std::vector<uint8_t> filtered;
  const pvzstd_status st = DecompressFrame(src, src_size, e.nbytes, &filtered);
  if (st != PVZSTD_OK) return st;
  Unshuffle(filtered.data(), static_cast<uint8_t *>(dst), e.nbytes, itemsize);
  return PVZSTD_OK;
}

pvzstd_status pvzstd_read_arrays(const pvzstd_reader *reader, const uint64_t *indices,
                                 uint64_t count, void *const *dsts, const uint64_t *dst_sizes,
                                 int n_threads) {
  if (reader == nullptr || indices == nullptr || dsts == nullptr || dst_sizes == nullptr) {
    return PVZSTD_E_INVALID;
  }
  if (count == 0) return PVZSTD_OK;

  int workers = n_threads;
  if (workers == PVZSTD_THREADS_AUTO) {
    // Spawning a thread costs more than decompressing a small frame. Deciding
    // on frame *count* alone spawned one thread per frame for a 10 KB file and
    // measured 2.5x slower than doing the same work inline -- so the size of
    // the work has to enter the decision, not just how many pieces it is in.
    //
    // AUTO is a request for the fastest setting, not for maximum parallelism;
    // an explicit n_threads is still honoured verbatim.
    uint64_t total = 0;
    for (uint64_t i = 0; i < count; ++i) total += dst_sizes[i];
    workers = total < kParallelDecompressFloor ? 1 : pvzstd::detail::HardwareWorkers();
  }
  if (workers > static_cast<int>(count)) workers = static_cast<int>(count);
  // A build with no thread runtime has no pool to spread over. Doing the work
  // inline is the same work in the same order, so this is a speed difference
  // and not a behavioural one -- which is why it is a clamp and not an error.
  if (!pvzstd::detail::kHasThreads) workers = 1;

  if (workers <= 1) {
    for (uint64_t i = 0; i < count; ++i) {
      const pvzstd_status st = pvzstd_read_array_at(reader, indices[i], dsts[i], dst_sizes[i]);
      if (st != PVZSTD_OK) return st;
    }
    return PVZSTD_OK;
  }

  // Static striding rather than a work queue: frames vary in size but there
  // is no shared state to contend on, so the simplest partition that keeps
  // every worker busy is enough. Each slot is written by exactly one thread.
  std::vector<pvzstd_status> results(static_cast<size_t>(count), PVZSTD_OK);
  pvzstd::detail::ParallelStride(workers, count, [&](uint64_t i) {
    results[static_cast<size_t>(i)] =
        pvzstd_read_array_at(reader, indices[i], dsts[i], dst_sizes[i]);
  });

  for (uint64_t i = 0; i < count; ++i) {
    if (results[static_cast<size_t>(i)] != PVZSTD_OK) return results[static_cast<size_t>(i)];
  }
  return PVZSTD_OK;
}

const char *pvzstd_ds_metadata_json(const pvzstd_reader *reader) {
  if (reader == nullptr || !reader->has_ds_metadata) return nullptr;
  return reader->ds_metadata.c_str();
}

const char *pvzstd_file_metadata_json(const pvzstd_reader *reader) {
  if (reader == nullptr || !reader->has_file_metadata) return nullptr;
  return reader->file_metadata.c_str();
}

const char *pvzstd_status_message(pvzstd_status status) {
  switch (status) {
    case PVZSTD_OK:
      return "ok";
    case PVZSTD_E_IO:
      return "file missing, unreadable, or truncated";
    case PVZSTD_E_FORMAT:
      return "container did not parse as a .pv trailer-indexed file";
    case PVZSTD_E_ZSTD:
      // Both directions: the writer reports a rejected compression parameter
      // through the same code, and "failed to decompress" reads as a damaged
      // file when the cause was a request this zstd build cannot serve.
      return "zstd rejected a frame or a compression parameter";
    case PVZSTD_E_RANGE:
      return "index or count out of range, or destination buffer too small";
    case PVZSTD_E_NOMEM:
      return "allocation failed";
    case PVZSTD_E_FILTER:
      return "array uses a filter this build cannot reverse";
    case PVZSTD_E_INVALID:
      return "invalid argument";
  }
  return "unknown status";
}

uint32_t pvzstd_abi_version(void) { return PVZSTD_ABI_VERSION; }

}  // extern "C"
