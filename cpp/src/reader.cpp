// Reader half of the .pv container. See doc/format/container-v2.md.
//
// Index entries carry each frame's END offset, so starts are derived by shifting;
// frames pair as (header, payload); the header's filter byte is present only when
// non-zero, so header length is the signal, not the file version.

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <memory>
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
// The .zvtk-era spelling of the same frame. Accepted so a legacy container
// reaches the reader at all; which one a file used is reported by name through
// pvz_metadata_name_at, because the caller may want to say so.
constexpr char kLegacyFileMetadataSuffix[] = "__zvtk_metadata";
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

// Invert the shuffle filter: on disk it is itemsize planes of n_elem bytes.
void Unshuffle(const uint8_t *src, uint8_t *dst, uint64_t nbytes, uint64_t itemsize) {
  const uint64_t n_elem = nbytes / itemsize;
  for (uint64_t plane = 0; plane < itemsize; ++plane) {
    const uint8_t *in = src + plane * n_elem;
    uint8_t *out = dst + plane;
    for (uint64_t i = 0; i < n_elem; ++i) out[i * itemsize] = in[i];
  }
}

// Byte width implied by a numpy dtype string, or 0 when it cannot be read off the
// tag. Shared with the writer so the two halves cannot disagree about a stride.
uint64_t DtypeItemsize(const char *dtype) {
  const pvzstd::detail::Dtype d = pvzstd::detail::ParseDtype(dtype);
  return d.valid ? d.itemsize : 0;
}

// Whether the declared payload size agrees with the header's shape and dtype.
//
// Nothing else checks this, and the mismatch is a buffer overrun produced by a
// file: reads honour the declared payload size while callers size destinations
// from shape and dtype, so a "10 float64" header over an 8000-byte payload writes
// 8000 bytes into 80. An unrecognised dtype spelling is left unchecked rather than
// rejected, since the format carries whatever numpy spelled.
bool DeclaredSizeAgrees(const std::vector<uint64_t> &shape, const char *dtype, uint64_t nbytes) {
  uint64_t n = DtypeItemsize(dtype);
  if (n == 0) return true;
  for (const uint64_t dim : shape) {
    if (dim != 0 && n > UINT64_MAX / dim) return false;
    n *= dim;
  }
  return n == nbytes;
}

// Total decompressed bytes below which AUTO stays inline. A floor, not a tuned
// optimum: below it, spawning a thread per frame costs more than it saves, and
// the crossover is machine-dependent.
constexpr uint64_t kParallelDecompressFloor = 4ull << 20;

// A read-only view of the file. Mapping rather than copying, so that opening a
// container faults in only the frames actually read.
class FileMapping {
 public:
  FileMapping() = default;
  ~FileMapping() { Reset(); }
  FileMapping(const FileMapping &) = delete;
  FileMapping &operator=(const FileMapping &) = delete;

  pvz_status Open(const char *path) {
    Reset();
#if defined(_WIN32)
    handle_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) return PVZ_E_IO;
    LARGE_INTEGER li;
    if (GetFileSizeEx(handle_, &li) == 0 || li.QuadPart <= 0) {
      Reset();
      return PVZ_E_IO;
    }
    size_ = static_cast<uint64_t>(li.QuadPart);
    mapping_ = CreateFileMappingA(handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_ == nullptr) {
      Reset();
      return PVZ_E_IO;
    }
    data_ = static_cast<const uint8_t *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (data_ == nullptr) {
      Reset();
      return PVZ_E_IO;
    }
#else
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) return PVZ_E_IO;
    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
      Reset();
      return PVZ_E_IO;
    }
    size_ = static_cast<uint64_t>(st.st_size);
    void *addr = ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr == MAP_FAILED) {
      Reset();
      return PVZ_E_IO;
    }
    data_ = static_cast<const uint8_t *>(addr);
#endif
    return PVZ_OK;
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

struct pvz_reader {
  FileMapping map;
  std::vector<ArrayEntry> arrays;
  std::string ds_metadata;
  std::string file_metadata;
  bool has_ds_metadata = false;
  bool has_file_metadata = false;
  // Every metadata document, in file order, paired with the frame name it came
  // under. The two strings above hold only the last of each kind, which cannot
  // describe a MultiBlock.
  std::vector<std::string> metadata_names;
  std::vector<std::string> metadata_docs;
  // Trailer sizes, in file order, so a caller does not have to re-read them.
  std::vector<uint64_t> frame_decompressed;
  std::vector<uint64_t> frame_compressed;
  // Root dataset's field-data blocks, in metadata order. Empty for MultiBlock.
  std::vector<std::string> field_names;
  std::vector<int64_t> field_indices;  // into `arrays`; -1 if the frame is gone
  // Filled in by pvz_array_info_at so the caller sees stable pointers.
  mutable std::vector<uint64_t> scratch_shape;
};

namespace {

// One decompression context per thread, reused across frames -- zstd asks for a
// context allocated once and one per thread, and `thread_local` gives both without
// putting a handle in the ABI we cannot revise later. A container is many small
// frames, so one-shot ZSTD_decompress() paid setup per frame. nullptr if it could
// not be allocated; callers fall back to the one-shot entry point.
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

// Falls back to the one-shot call when no context is available; same result.
size_t DecompressInto(void *dst, size_t dst_capacity, const void *src, size_t src_size) {
  ZSTD_DCtx *const ctx = ThreadDCtx();
  if (ctx == nullptr) return ZSTD_decompress(dst, dst_capacity, src, src_size);
  return ZSTD_decompressDCtx(ctx, dst, dst_capacity, src, src_size);
}

pvz_status DecompressFrame(const uint8_t *src, uint64_t src_size, uint64_t expected,
                           std::vector<uint8_t> *out) {
  try {
    out->resize(static_cast<size_t>(expected));
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  if (expected == 0) return PVZ_OK;
  const size_t got = DecompressInto(out->data(), out->size(), src, static_cast<size_t>(src_size));
  if (ZSTD_isError(got) != 0) return PVZ_E_ZSTD;
  // A mismatch is the signature of a misparsed index: every frame still
  // decompresses, but each yields a neighbour's payload.
  if (got != expected) return PVZ_E_FORMAT;
  return PVZ_OK;
}

pvz_status ParseHeader(const std::vector<uint8_t> &buf, ArrayEntry *entry) {
  size_t off = 0;
  if (buf.size() < 4) return PVZ_E_FORMAT;
  const uint32_t name_len = LoadU32(buf.data());
  off += 4;
  if (buf.size() < off + name_len + 4) return PVZ_E_FORMAT;
  entry->name.assign(reinterpret_cast<const char *>(buf.data() + off), name_len);
  off += name_len;

  const uint32_t ndim = LoadU32(buf.data() + off);
  off += 4;
  if (buf.size() < off + static_cast<size_t>(ndim) * 8 + PVZSTD_DTYPE_LEN) return PVZ_E_FORMAT;
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

  // Absent filter byte means PVZ_FILTER_NONE. Testing file_version instead
  // would mis-parse a shuffled version-2 file.
  entry->filter_id = PVZ_FILTER_NONE;
  if (off < buf.size()) {
    entry->filter_id = buf[off];
    off += 1;
  }
  if (off != buf.size()) return PVZ_E_FORMAT;
  return PVZ_OK;
}

}  // namespace

extern "C" {

pvz_status pvz_open_versioned(const char *path, pvz_reader **out, uint32_t *file_version) try {
  if (path == nullptr || out == nullptr) return PVZ_E_INVALID;
  *out = nullptr;

  // Owning: eleven paths below refuse the container, and the function-try
  // handler catches throws from decompression sized by a field read out of
  // the file. A raw pointer leaked the fd and the mapping on that last one.
  std::unique_ptr<pvz_reader> reader(new (std::nothrow) pvz_reader());
  if (reader == nullptr) return PVZ_E_NOMEM;

  pvz_status st = reader->map.Open(path);
  if (st != PVZ_OK) {
    return st;
  }

  const FileMapping &raw = reader->map;
  if (raw.size() < kTrailerCountBytes) {
    return PVZ_E_FORMAT;
  }

  const uint64_t n_frames = LoadU64(raw.data() + raw.size() - kTrailerCountBytes);
  if (n_frames == 0 || (n_frames % 2) != 0) {
    return PVZ_E_FORMAT;  // frames pair as (header, payload)
  }
  // Bounded by division: n_frames comes from the file and n_frames * 16
  // overflows for a large enough value, turning the size check into a pass.
  if (n_frames > (raw.size() - kTrailerCountBytes) / kIndexEntryBytes) {
    return PVZ_E_FORMAT;
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

  // Banked before the frame walk: the walk consumes the metadata frames and
  // reports only arrays, so afterwards there is no longer a per-frame view.
  reader->frame_decompressed = sizes;
  reader->frame_compressed.resize(static_cast<size_t>(n_frames));
  for (uint64_t i = 0; i < n_frames; ++i) {
    const uint64_t prev = (i == 0) ? 0 : ends[static_cast<size_t>(i - 1)];
    reader->frame_compressed[static_cast<size_t>(i)] = ends[static_cast<size_t>(i)] - prev;
  }

  // MultiBlock has no single root, so its metadata frame abandons the field-array
  // index rather than guessing which block owns it.
  std::string ds_id;
  std::string root_ds_json;
  bool multiblock = false;

  std::vector<uint8_t> frame;
  for (uint64_t i = 0; i + 1 < n_frames; i += 2) {
    const uint64_t hdr_start = (i == 0) ? 0 : ends[static_cast<size_t>(i - 1)];
    const uint64_t hdr_end = ends[static_cast<size_t>(i)];
    const uint64_t pay_end = ends[static_cast<size_t>(i + 1)];
    if (hdr_start > hdr_end || hdr_end > pay_end || pay_end > index_off) {
      return PVZ_E_FORMAT;
    }

    st = DecompressFrame(raw.data() + hdr_start, hdr_end - hdr_start, sizes[static_cast<size_t>(i)],
                         &frame);
    if (st != PVZ_OK) {
      return st;
    }

    ArrayEntry entry;
    st = ParseHeader(frame, &entry);
    if (st != PVZ_OK) {
      return st;
    }
    entry.nbytes = sizes[static_cast<size_t>(i + 1)];
    entry.payload_start = hdr_end;
    entry.payload_end = pay_end;
    if (!DeclaredSizeAgrees(entry.shape, entry.dtype, entry.nbytes)) {
      return PVZ_E_FORMAT;
    }

    const bool is_ds = EndsWith(entry.name, kDsMetadataSuffix);
    const bool is_file = EndsWith(entry.name, kFileMetadataSuffix) ||
                         EndsWith(entry.name, kLegacyFileMetadataSuffix);
    if (is_ds || is_file) {
      std::vector<uint8_t> payload;
      st = DecompressFrame(raw.data() + entry.payload_start,
                           entry.payload_end - entry.payload_start, entry.nbytes, &payload);
      if (st != PVZ_OK) {
        return st;
      }
      std::string json(reinterpret_cast<const char *>(payload.data()), payload.size());
      reader->metadata_names.push_back(entry.name);
      reader->metadata_docs.push_back(json);
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

  // Refuse a container this build cannot decode, before anything is read out of
  // it. The version is the format's own statement of what it takes to read a
  // file, so the ceiling belongs beside the decoder rather than in one of its
  // front ends -- a C, WASM or Python caller all get the same answer, and none
  // of them can drift by keeping its own copy of the number. A newer file may
  // transform payloads in a way this build cannot invert, which would hand back
  // corrupt values rather than fail.
  if (reader->has_file_metadata) {
    long long version = 0;
    if (pvzstd::json::MemberInt(reader->file_metadata, "file_version", &version) && version >= 0) {
      if (file_version != nullptr) *file_version = static_cast<uint32_t>(version);
      if (version > static_cast<long long>(PVZSTD_FILE_VERSION_MAX)) {
        return PVZ_E_VERSION;
      }
    }
  }

  // Names come from the dataset metadata, not the frame names: a frame-name scan
  // would also pick up an array whose name merely ends the same way.
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
        // Kept, not dropped: the reference reader lists it too and refuses only
        // on read, and a shorter list here would hide the desync.
        reader->field_names.push_back(key);
        reader->field_indices.push_back(found);
      }
    }
  }

  *out = reader.release();
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

pvz_status pvz_open(const char *path, pvz_reader **out) try {
  return pvz_open_versioned(path, out, nullptr);
} catch (...) {
  return PVZ_E_NOMEM;
}

uint32_t pvz_max_file_version(void) { return PVZSTD_FILE_VERSION_MAX; }

void pvz_close(pvz_reader *reader) try { delete reader; } catch (...) {
}

uint64_t pvz_array_count(const pvz_reader *reader) try {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->arrays.size());
} catch (...) {
  return 0;
}

namespace {

// Every pointer aliases storage the reader owns for its whole life, so the filled
// struct stays valid until pvz_close().
void FillArrayInfo(const ArrayEntry &e, pvz_array_info *out) {
  out->name = e.name.c_str();
  out->shape = e.shape.empty() ? nullptr : e.shape.data();
  out->ndim = static_cast<uint32_t>(e.shape.size());
  out->filter_id = e.filter_id;
  std::memcpy(out->dtype, e.dtype, sizeof(out->dtype));
  out->nbytes = e.nbytes;
}

}  // namespace

pvz_status pvz_array_info_at(const pvz_reader *reader, uint64_t index, pvz_array_info *out) try {
  if (reader == nullptr || out == nullptr) return PVZ_E_INVALID;
  if (index >= reader->arrays.size()) return PVZ_E_RANGE;
  FillArrayInfo(reader->arrays[static_cast<size_t>(index)], out);
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

pvz_status pvz_array_info_range(const pvz_reader *reader, uint64_t first, uint64_t count,
                                pvz_array_info *out) try {
  if (reader == nullptr) return PVZ_E_INVALID;
  if (count == 0) return PVZ_OK;
  if (out == nullptr) return PVZ_E_INVALID;

  const uint64_t total = static_cast<uint64_t>(reader->arrays.size());
  // Against the remaining count, not first + count, which would wrap.
  if (first > total || count > total - first) return PVZ_E_RANGE;

  for (uint64_t i = 0; i < count; ++i) {
    FillArrayInfo(reader->arrays[static_cast<size_t>(first + i)], &out[i]);
  }
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

uint64_t pvz_field_array_count(const pvz_reader *reader) try {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->field_names.size());
} catch (...) {
  return 0;
}

const char *pvz_field_array_name_at(const pvz_reader *reader, uint64_t index) try {
  if (reader == nullptr || index >= reader->field_names.size()) return nullptr;
  return reader->field_names[static_cast<size_t>(index)].c_str();
} catch (...) {
  return nullptr;
}

int64_t pvz_find_field_array(const pvz_reader *reader, const char *name) try {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->field_names.size(); ++i) {
    if (reader->field_names[i] == name) return reader->field_indices[i];
  }
  return -1;
} catch (...) {
  return -1;
}

int64_t pvz_find_array(const pvz_reader *reader, const char *name) try {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->arrays.size(); ++i) {
    if (reader->arrays[i].name == name) return static_cast<int64_t>(i);
  }
  return -1;
} catch (...) {
  return -1;
}

pvz_status pvz_read_array_at(const pvz_reader *reader, uint64_t index, void *dst,
                             uint64_t dst_size) try {
  if (reader == nullptr || dst == nullptr) return PVZ_E_INVALID;
  if (index >= reader->arrays.size()) return PVZ_E_RANGE;
  const ArrayEntry &e = reader->arrays[static_cast<size_t>(index)];
  if (dst_size < e.nbytes) return PVZ_E_RANGE;
  if (e.nbytes == 0) return PVZ_OK;

  const uint8_t *src = reader->map.data() + e.payload_start;
  const uint64_t src_size = e.payload_end - e.payload_start;

  if (e.filter_id == PVZ_FILTER_NONE) {
    const size_t got =
        DecompressInto(dst, static_cast<size_t>(dst_size), src, static_cast<size_t>(src_size));
    if (ZSTD_isError(got) != 0) return PVZ_E_ZSTD;
    return got == e.nbytes ? PVZ_OK : PVZ_E_FORMAT;
  }

  if (e.filter_id != PVZ_FILTER_SHUFFLE) return PVZ_E_FILTER;

  const uint64_t itemsize = DtypeItemsize(e.dtype);
  if (itemsize == 0 || (e.nbytes % itemsize) != 0) return PVZ_E_FORMAT;

  std::vector<uint8_t> filtered;
  const pvz_status st = DecompressFrame(src, src_size, e.nbytes, &filtered);
  if (st != PVZ_OK) return st;
  Unshuffle(filtered.data(), static_cast<uint8_t *>(dst), e.nbytes, itemsize);
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

pvz_status pvz_read_arrays(const pvz_reader *reader, const uint64_t *indices, uint64_t count,
                           void *const *dsts, const uint64_t *dst_sizes, int n_threads) try {
  if (reader == nullptr || indices == nullptr || dsts == nullptr || dst_sizes == nullptr) {
    return PVZ_E_INVALID;
  }
  if (count == 0) return PVZ_OK;

  int workers = n_threads;
  if (workers == PVZ_THREADS_AUTO) {
    // Size enters the decision, not just frame count. AUTO asks for the fastest
    // setting; an explicit n_threads is honoured.
    uint64_t total = 0;
    for (uint64_t i = 0; i < count; ++i) total += dst_sizes[i];
    workers = total < kParallelDecompressFloor ? 1 : pvzstd::detail::HardwareWorkers();
  } else if (workers < 0) {
    // Negative means every core, the same sign convention pvz_writer_set_threads
    // uses. It meant "inline" here until the clamp below was written against a
    // signed count, which is the opposite of what the two share an ABI to say.
    workers = pvzstd::detail::HardwareWorkers();
  }
  // In uint64: count is unbounded, and narrowing it to int first could land on a
  // negative and silently drop this call to a single worker.
  if (count < static_cast<uint64_t>(workers)) workers = static_cast<int>(count);
  // Same work in the same order, so a clamp rather than an error.
  if (!pvzstd::detail::kHasThreads) workers = 1;

  if (workers <= 1) {
    for (uint64_t i = 0; i < count; ++i) {
      const pvz_status st = pvz_read_array_at(reader, indices[i], dsts[i], dst_sizes[i]);
      if (st != PVZ_OK) return st;
    }
    return PVZ_OK;
  }

  // Static striding rather than a work queue; each slot is written by one thread.
  std::vector<pvz_status> results(static_cast<size_t>(count), PVZ_OK);
  pvzstd::detail::ParallelStride(workers, count, [&](uint64_t i) {
    results[static_cast<size_t>(i)] = pvz_read_array_at(reader, indices[i], dsts[i], dst_sizes[i]);
  });

  for (uint64_t i = 0; i < count; ++i) {
    if (results[static_cast<size_t>(i)] != PVZ_OK) return results[static_cast<size_t>(i)];
  }
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

const char *pvz_ds_metadata_json(const pvz_reader *reader) try {
  if (reader == nullptr || !reader->has_ds_metadata) return nullptr;
  return reader->ds_metadata.c_str();
} catch (...) {
  return nullptr;
}

const char *pvz_file_metadata_json(const pvz_reader *reader) try {
  if (reader == nullptr || !reader->has_file_metadata) return nullptr;
  return reader->file_metadata.c_str();
} catch (...) {
  return nullptr;
}

uint64_t pvz_metadata_count(const pvz_reader *reader) try {
  if (reader == nullptr) return 0;
  return reader->metadata_docs.size();
} catch (...) {
  return 0;
}

const char *pvz_metadata_name_at(const pvz_reader *reader, uint64_t index) try {
  if (reader == nullptr || index >= reader->metadata_names.size()) return nullptr;
  return reader->metadata_names[static_cast<size_t>(index)].c_str();
} catch (...) {
  return nullptr;
}

const char *pvz_metadata_json_at(const pvz_reader *reader, uint64_t index) try {
  if (reader == nullptr || index >= reader->metadata_docs.size()) return nullptr;
  return reader->metadata_docs[static_cast<size_t>(index)].c_str();
} catch (...) {
  return nullptr;
}

uint64_t pvz_frame_count(const pvz_reader *reader) try {
  if (reader == nullptr) return 0;
  return reader->frame_decompressed.size();
} catch (...) {
  return 0;
}

pvz_status pvz_frame_sizes(const pvz_reader *reader, uint64_t *decompressed, uint64_t *compressed,
                           uint64_t capacity) try {
  if (reader == nullptr) return PVZ_E_INVALID;
  const size_t n = reader->frame_decompressed.size();
  // The caller sized these from pvz_frame_count(), a separate call: nothing
  // else here can tell a correctly sized buffer from one allocated against a
  // different reader, so refuse rather than write past the end.
  if (capacity < static_cast<uint64_t>(n)) return PVZ_E_RANGE;
  if (decompressed != nullptr) {
    std::memcpy(decompressed, reader->frame_decompressed.data(), n * sizeof(uint64_t));
  }
  if (compressed != nullptr) {
    std::memcpy(compressed, reader->frame_compressed.data(), n * sizeof(uint64_t));
  }
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

const char *pvz_status_message(pvz_status status) {
  switch (status) {
    case PVZ_OK:
      return "ok";
    case PVZ_E_IO:
      return "file missing, unreadable, or truncated";
    case PVZ_E_FORMAT:
      return "container did not parse as a .pv trailer-indexed file";
    case PVZ_E_ZSTD:
      // Both directions: the writer reports a rejected compression parameter
      // through this code too.
      return "zstd rejected a frame or a compression parameter";
    case PVZ_E_RANGE:
      return "index or count out of range, or destination buffer too small";
    case PVZ_E_NOMEM:
      return "allocation failed";
    case PVZ_E_FILTER:
      return "array uses a filter this build cannot reverse";
    case PVZ_E_INVALID:
      return "invalid argument";
    case PVZ_E_UNSUPPORTED:
      return "the container is a shape this operation cannot serve";
    case PVZ_E_EXISTS:
      return "an array of that name is already in the container";
    case PVZ_E_VERSION:
      return "the container's file version is newer than this build can decode";
  }
  return "unknown status";
}

uint32_t pvz_abi_version(void) { return PVZSTD_ABI_VERSION; }

}  // extern "C"
