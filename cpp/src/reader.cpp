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

#include "pvzstd/pvzstd.h"

#include "json_read.h"

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
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
  uint64_t nbytes;        // decompressed payload size
  uint64_t payload_start; // byte range of the compressed payload frame
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

// Byte width implied by a numpy dtype string such as "<f8" or "|u1".
uint64_t DtypeItemsize(const char *dtype) {
  const size_t n = std::strlen(dtype);
  if (n < 3) return 0;
  uint64_t width = 0;
  for (size_t i = 2; i < n; ++i) {
    if (dtype[i] < '0' || dtype[i] > '9') return 0;
    width = width * 10 + static_cast<uint64_t>(dtype[i] - '0');
  }
  return width;
}

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
    struct stat st {};
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
  // Field-data blocks of the root dataset, in the order the dataset metadata
  // lists them. Empty for a MultiBlock container, which has no single root.
  std::vector<std::string> field_names;
  std::vector<int64_t> field_indices;  // into `arrays`; -1 if the frame is gone
  // Filled in by pvz_array_info_at so the caller sees stable pointers.
  mutable std::vector<uint64_t> scratch_shape;
};

namespace {

pvz_status DecompressFrame(const uint8_t *src, uint64_t src_size, uint64_t expected,
                           std::vector<uint8_t> *out) {
  try {
    out->resize(static_cast<size_t>(expected));
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  if (expected == 0) return PVZ_OK;
  const size_t got = ZSTD_decompress(out->data(), out->size(), src, static_cast<size_t>(src_size));
  if (ZSTD_isError(got) != 0) return PVZ_E_ZSTD;
  // A mismatch here is the signature of a misparsed index: every frame still
  // decompresses, but each one yields a neighbour's payload.
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

  // The filter byte is written only when a filter is in use, so its absence
  // means PVZ_FILTER_NONE. Testing file_version here instead would mis-parse
  // a shuffled version-2 file.
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

pvz_status pvz_open(const char *path, pvz_reader **out) {
  if (path == nullptr || out == nullptr) return PVZ_E_INVALID;
  *out = nullptr;

  pvz_reader *reader = new (std::nothrow) pvz_reader();
  if (reader == nullptr) return PVZ_E_NOMEM;

  pvz_status st = reader->map.Open(path);
  if (st != PVZ_OK) {
    delete reader;
    return st;
  }

  const FileMapping &raw = reader->map;
  if (raw.size() < kTrailerCountBytes) {
    delete reader;
    return PVZ_E_FORMAT;
  }

  const uint64_t n_frames = LoadU64(raw.data() + raw.size() - kTrailerCountBytes);
  if (n_frames == 0 || (n_frames % 2) != 0) {
    delete reader;  // frames pair as (header, payload)
    return PVZ_E_FORMAT;
  }
  const uint64_t index_bytes = n_frames * kIndexEntryBytes;
  if (raw.size() < kTrailerCountBytes + index_bytes) {
    delete reader;
    return PVZ_E_FORMAT;
  }
  const uint64_t index_off = raw.size() - kTrailerCountBytes - index_bytes;

  std::vector<uint64_t> ends(static_cast<size_t>(n_frames));
  std::vector<uint64_t> sizes(static_cast<size_t>(n_frames));
  for (uint64_t i = 0; i < n_frames; ++i) {
    const uint8_t *p = raw.data() + index_off + i * kIndexEntryBytes;
    ends[static_cast<size_t>(i)] = LoadU64(p);       // END offset, not start
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
      return PVZ_E_FORMAT;
    }

    st = DecompressFrame(raw.data() + hdr_start, hdr_end - hdr_start,
                         sizes[static_cast<size_t>(i)], &frame);
    if (st != PVZ_OK) {
      delete reader;
      return st;
    }

    ArrayEntry entry;
    st = ParseHeader(frame, &entry);
    if (st != PVZ_OK) {
      delete reader;
      return st;
    }
    entry.nbytes = sizes[static_cast<size_t>(i + 1)];
    entry.payload_start = hdr_end;
    entry.payload_end = pay_end;

    const bool is_ds = EndsWith(entry.name, kDsMetadataSuffix);
    const bool is_file = EndsWith(entry.name, kFileMetadataSuffix);
    if (is_ds || is_file) {
      std::vector<uint8_t> payload;
      st = DecompressFrame(raw.data() + entry.payload_start,
                           entry.payload_end - entry.payload_start, entry.nbytes, &payload);
      if (st != PVZ_OK) {
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
  return PVZ_OK;
}

void pvz_close(pvz_reader *reader) { delete reader; }

uint64_t pvz_array_count(const pvz_reader *reader) {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->arrays.size());
}

pvz_status pvz_array_info_at(const pvz_reader *reader, uint64_t index, pvz_array_info *out) {
  if (reader == nullptr || out == nullptr) return PVZ_E_INVALID;
  if (index >= reader->arrays.size()) return PVZ_E_RANGE;
  const ArrayEntry &e = reader->arrays[static_cast<size_t>(index)];
  out->name = e.name.c_str();
  out->shape = e.shape.empty() ? nullptr : e.shape.data();
  out->ndim = static_cast<uint32_t>(e.shape.size());
  out->filter_id = e.filter_id;
  std::memcpy(out->dtype, e.dtype, sizeof(out->dtype));
  out->nbytes = e.nbytes;
  return PVZ_OK;
}

uint64_t pvz_field_array_count(const pvz_reader *reader) {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->field_names.size());
}

const char *pvz_field_array_name_at(const pvz_reader *reader, uint64_t index) {
  if (reader == nullptr || index >= reader->field_names.size()) return nullptr;
  return reader->field_names[static_cast<size_t>(index)].c_str();
}

int64_t pvz_find_field_array(const pvz_reader *reader, const char *name) {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->field_names.size(); ++i) {
    if (reader->field_names[i] == name) return reader->field_indices[i];
  }
  return -1;
}

int64_t pvz_find_array(const pvz_reader *reader, const char *name) {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->arrays.size(); ++i) {
    if (reader->arrays[i].name == name) return static_cast<int64_t>(i);
  }
  return -1;
}

pvz_status pvz_read_array_at(const pvz_reader *reader, uint64_t index, void *dst,
                             uint64_t dst_size) {
  if (reader == nullptr || dst == nullptr) return PVZ_E_INVALID;
  if (index >= reader->arrays.size()) return PVZ_E_RANGE;
  const ArrayEntry &e = reader->arrays[static_cast<size_t>(index)];
  if (dst_size < e.nbytes) return PVZ_E_RANGE;
  if (e.nbytes == 0) return PVZ_OK;

  const uint8_t *src = reader->map.data() + e.payload_start;
  const uint64_t src_size = e.payload_end - e.payload_start;

  if (e.filter_id == PVZ_FILTER_NONE) {
    const size_t got = ZSTD_decompress(dst, static_cast<size_t>(dst_size), src,
                                       static_cast<size_t>(src_size));
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
}

pvz_status pvz_read_arrays(const pvz_reader *reader, const uint64_t *indices, uint64_t count,
                           void *const *dsts, const uint64_t *dst_sizes, int n_threads) {
  if (reader == nullptr || indices == nullptr || dsts == nullptr || dst_sizes == nullptr) {
    return PVZ_E_INVALID;
  }
  if (count == 0) return PVZ_OK;

  int workers = n_threads;
  if (workers == PVZ_THREADS_AUTO) {
    const unsigned hw = std::thread::hardware_concurrency();
    workers = hw == 0 ? 1 : static_cast<int>(hw);
  }
  if (workers > static_cast<int>(count)) workers = static_cast<int>(count);

  if (workers <= 1) {
    for (uint64_t i = 0; i < count; ++i) {
      const pvz_status st = pvz_read_array_at(reader, indices[i], dsts[i], dst_sizes[i]);
      if (st != PVZ_OK) return st;
    }
    return PVZ_OK;
  }

  // Static striding rather than a work queue: frames vary in size but there
  // is no shared state to contend on, so the simplest partition that keeps
  // every worker busy is enough. Each slot is written by exactly one thread.
  std::vector<pvz_status> results(static_cast<size_t>(count), PVZ_OK);
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers));
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&, w]() {
      for (uint64_t i = static_cast<uint64_t>(w); i < count;
           i += static_cast<uint64_t>(workers)) {
        results[static_cast<size_t>(i)] =
            pvz_read_array_at(reader, indices[i], dsts[i], dst_sizes[i]);
      }
    });
  }
  for (std::thread &t : pool) t.join();

  for (uint64_t i = 0; i < count; ++i) {
    if (results[static_cast<size_t>(i)] != PVZ_OK) return results[static_cast<size_t>(i)];
  }
  return PVZ_OK;
}

const char *pvz_ds_metadata_json(const pvz_reader *reader) {
  if (reader == nullptr || !reader->has_ds_metadata) return nullptr;
  return reader->ds_metadata.c_str();
}

const char *pvz_file_metadata_json(const pvz_reader *reader) {
  if (reader == nullptr || !reader->has_file_metadata) return nullptr;
  return reader->file_metadata.c_str();
}

const char *pvz_status_message(pvz_status status) {
  switch (status) {
    case PVZ_OK: return "ok";
    case PVZ_E_IO: return "file missing, unreadable, or truncated";
    case PVZ_E_FORMAT: return "container did not parse as a .pv trailer-indexed file";
    case PVZ_E_ZSTD: return "a zstd frame failed to decompress";
    case PVZ_E_RANGE: return "index out of range, or destination buffer too small";
    case PVZ_E_NOMEM: return "allocation failed";
    case PVZ_E_FILTER: return "array uses a filter this build cannot reverse";
    case PVZ_E_INVALID: return "invalid argument";
  }
  return "unknown status";
}

uint32_t pvz_abi_version(void) { return PVZSTD_ABI_VERSION; }

}  // extern "C"
