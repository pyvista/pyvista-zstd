// Reader half of the .pv container. See doc/format/container-v2.md.
//
// Index entries carry each frame's END offset, so starts are derived by shifting;
// frames pair as (header, payload); the header's filter byte is present only when
// non-zero, so header length is the signal, not the file version.

#include <zstd.h>

#include <cstdint>
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
#include "thumbnail.h"

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
// The .zvtk-era spelling of the same frame. Accepted so a legacy container
// reaches the reader at all; which one a file used is reported by name through
// pvzstd_metadata_name_at, because the caller may want to say so.
constexpr char kLegacyFileMetadataSuffix[] = "__zvtk_metadata";

// Shared with the write, append and stream paths (detail.h).
using pvzstd::detail::EndsWith;
using pvzstd::detail::ExceedsSizeT;
using pvzstd::detail::kDsMetadataSuffix;
using pvzstd::detail::kFieldDataSuffix;
using pvzstd::detail::kFileMetadataSuffix;
using pvzstd::detail::kIndexEntryBytes;
using pvzstd::detail::kMultiblockSuffix;
using pvzstd::detail::kUidNChar;
using pvzstd::detail::LoadU32;

// The most a zstd frame can expand, per compressed byte. A zstd block covers at
// most 128 KiB, and the cheapest way to spell one is an RLE block: a 3-byte
// block header plus a single byte of content. So no frame can yield more than
// 131072 / 4 = 32768 bytes per byte it occupies, and a real frame is well under
// that -- a large array of one repeated value, the most compressible input
// there is, measures around 12000:1 at every compression level this writer
// uses. Every frame lives inside the container, so the container's own length
// is an upper bound on any frame's compressed size, and this ratio times that
// length is a size no honest frame can declare.
constexpr uint64_t kMaxExpansionRatio = 32768;

uint64_t LoadU64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | p[static_cast<size_t>(i)];
  return v;
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

// Invert the shuffle filter for elements [first, last): on disk it is itemsize
// planes of n_elem bytes, and each element's bytes come one from every plane,
// so a range of elements can be restored without the others.
void UnshuffleRange(const uint8_t *src, uint8_t *dst, uint64_t n_elem, uint64_t itemsize,
                    uint64_t first, uint64_t last) {
  for (uint64_t plane = 0; plane < itemsize; ++plane) {
    const uint8_t *in = src + plane * n_elem;
    uint8_t *out = dst + plane;
    for (uint64_t i = first; i < last; ++i) out[i * itemsize] = in[i];
  }
}

// Unshuffle on `workers` threads, each taking a run of elements.
void Unshuffle(const uint8_t *src, uint8_t *dst, uint64_t nbytes, uint64_t itemsize, int workers) {
  const uint64_t n_elem = nbytes / itemsize;
  if (workers <= 1 || n_elem < static_cast<uint64_t>(workers)) {
    UnshuffleRange(src, dst, n_elem, itemsize, 0, n_elem);
    return;
  }
  const uint64_t per =
      (n_elem + static_cast<uint64_t>(workers) - 1) / static_cast<uint64_t>(workers);
  pvzstd::detail::ParallelStride(workers, static_cast<uint64_t>(workers), [&](uint64_t w) {
    const uint64_t first = w * per;
    const uint64_t last = first + per < n_elem ? first + per : n_elem;
    if (first < last) UnshuffleRange(src, dst, n_elem, itemsize, first, last);
  });
}

// Byte width implied by a numpy dtype string, or 0 when it cannot be read off the
// tag. Shared with the writer so the two halves cannot disagree about a stride.
uint64_t DtypeItemsize(const char *dtype) {
  const pvzstd::detail::Dtype d = pvzstd::detail::ParseDtype(dtype);
  return d.valid ? d.itemsize : 0;
}

// Whether the declared payload size agrees with the header's shape and dtype.
//
// A mismatch is a file-driven buffer overrun: reads honour the declared payload
// size while callers size destinations from shape and dtype. An unrecognised
// dtype spelling is left unchecked, since the format carries whatever numpy
// spelled.
bool DeclaredSizeAgrees(const std::vector<uint64_t> &shape, const char *dtype, uint64_t nbytes) {
  uint64_t n = DtypeItemsize(dtype);
  if (n == 0) return true;
  for (const uint64_t dim : shape) {
    if (dim != 0 && n > UINT64_MAX / dim) return false;
    n *= dim;
  }
  return n == nbytes;
}

// Optimization: total decompressed bytes below which AUTO stays inline. A floor, not a tuned
// optimum: below it, spawning a thread per frame costs more than it saves, and
// the crossover is machine-dependent.
constexpr uint64_t kParallelDecompressFloor = 4ull << 20;

// A read-only view of the container's bytes: either mapped from a file, or
// borrowed from a caller that already holds them. Mapping rather than copying,
// so that opening a file faults in only the frames actually read; borrowing
// rather than copying, so a caller holding the container pays for it once.
//
// The two cases differ only in where the bytes came from and who releases them.
// That is what lets one parse serve both.
class ContainerBytes {
 public:
  ContainerBytes() = default;
  ~ContainerBytes() { Reset(); }
  ContainerBytes(const ContainerBytes &) = delete;
  ContainerBytes &operator=(const ContainerBytes &) = delete;

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
    // A container larger than the address space cannot be mapped whole, and
    // mapping the low bits of its length would leave every offset in the parse
    // pointing outside the view. Refuse it instead. Unreachable where size_t is
    // 64 bits.
    if (ExceedsSizeT(size_)) {
      Reset();
      return PVZSTD_E_IO;
    }
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
    // See the Windows arm: a file too large to address cannot be mapped whole,
    // and truncating its length into mmap would give a view every later offset
    // reads past the end of.
    if (ExceedsSizeT(size_)) {
      Reset();
      return PVZSTD_E_IO;
    }
    void *addr = ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr == MAP_FAILED) {
      Reset();
      return PVZSTD_E_IO;
    }
    data_ = static_cast<const uint8_t *>(addr);
#endif
    return PVZSTD_OK;
  }

  // Take a byte range the caller owns. Nothing is opened, mapped or copied, so
  // there is no file I/O anywhere on this path -- which is the whole point on a
  // build with no filesystem to stage the bytes into. The range must outlive
  // the reader; Reset() releases nothing here, because nothing was acquired.
  pvzstd_status Borrow(const void *data, uint64_t size) {
    Reset();
    if (data == nullptr || size == 0) return PVZSTD_E_INVALID;
    // A borrowed range longer than the address space is a caller error, not a
    // buffer: nothing could have allocated it.
    if (ExceedsSizeT(size)) return PVZSTD_E_INVALID;
    data_ = static_cast<const uint8_t *>(data);
    size_ = size;
    owned_ = false;
    return PVZSTD_OK;
  }

  const uint8_t *data() const { return data_; }
  uint64_t size() const { return size_; }

 private:
  void Reset() {
    if (!owned_) {
      data_ = nullptr;
      size_ = 0;
      owned_ = true;
      return;
    }
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
  // Whether this view acquired what it points at. False for a borrowed range,
  // so closing a reader cannot unmap or close a buffer the caller still owns.
  bool owned_ = true;
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace

struct pvzstd_reader {
  // The container's bytes: mapped from a file, or borrowed from the caller.
  ContainerBytes map;
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
  // The thumbnail record, inside the container's bytes; null when it has none.
  const uint8_t *thumbnail_record = nullptr;
  pvzstd::detail::ThumbnailRecord thumbnail;
  bool thumbnail_stale = false;
  // Filled in by pvzstd_array_info_at so the caller sees stable pointers.
  mutable std::vector<uint64_t> scratch_shape;
};

namespace {

// One decompression context per thread, reused across frames: zstd wants a
// context allocated once per thread, and a container is many small frames.
// nullptr if it could not be allocated; callers fall back to one-shot decode.
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

// One zstd frame of a payload and where its bytes go.
struct SubFrame {
  const uint8_t *src;
  size_t src_size;
  uint8_t *dst;
  uint64_t dst_size;
};

// Split a payload into the zstd frames it concatenates, each placed after the
// previous one's output. False when the payload is not a run of whole frames
// whose content sizes are declared and add up to `nbytes`; the one-shot decode
// then reports what is wrong with it.
bool SplitSubFrames(const uint8_t *src, uint64_t src_size, uint8_t *dst, uint64_t nbytes,
                    std::vector<SubFrame> *out) {
  out->clear();
  uint64_t off = 0;
  uint64_t produced = 0;
  while (off < src_size) {
    const size_t csize =
        ZSTD_findFrameCompressedSize(src + off, static_cast<size_t>(src_size - off));
    if (ZSTD_isError(csize) != 0 || csize == 0) return false;
    const unsigned long long dsize = ZSTD_getFrameContentSize(src + off, csize);
    if (dsize == ZSTD_CONTENTSIZE_UNKNOWN || dsize == ZSTD_CONTENTSIZE_ERROR) return false;
    if (dsize > nbytes - produced) return false;
    out->push_back(SubFrame{src + off, csize, dst + produced, static_cast<uint64_t>(dsize)});
    off += csize;
    produced += dsize;
  }
  return produced == nbytes;
}

pvzstd_status DecompressSubFrame(const SubFrame &sub) {
  const size_t got =
      DecompressInto(sub.dst, static_cast<size_t>(sub.dst_size), sub.src, sub.src_size);
  if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
  return got == sub.dst_size ? PVZSTD_OK : PVZSTD_E_FORMAT;
}

// Decompress a payload into `dst`, its sub-frames in parallel when there are
// several and `workers` allows. A payload that does not split cleanly goes
// through the one-shot decode, which accepts concatenated frames too.
pvzstd_status DecompressPayload(const uint8_t *src, uint64_t src_size, uint8_t *dst,
                                uint64_t nbytes, int workers) {
  if (workers > 1) {
    std::vector<SubFrame> subs;
    if (SplitSubFrames(src, src_size, dst, nbytes, &subs) && subs.size() > 1) {
      std::vector<pvzstd_status> results(subs.size(), PVZSTD_OK);
      const uint64_t count = subs.size();
      const int pool = count < static_cast<uint64_t>(workers) ? static_cast<int>(count) : workers;
      pvzstd::detail::ParallelStride(pool, count, [&](uint64_t k) {
        results[static_cast<size_t>(k)] = DecompressSubFrame(subs[static_cast<size_t>(k)]);
      });
      for (const pvzstd_status st : results) {
        if (st != PVZSTD_OK) return st;
      }
      return PVZSTD_OK;
    }
  }
  const size_t got =
      DecompressInto(dst, static_cast<size_t>(nbytes), src, static_cast<size_t>(src_size));
  if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
  // A mismatch is the signature of a misparsed index: every frame still
  // decompresses, but each yields a neighbour's payload.
  return got == nbytes ? PVZSTD_OK : PVZSTD_E_FORMAT;
}

pvzstd_status DecompressFrame(const uint8_t *src, uint64_t src_size, uint64_t expected,
                              std::vector<uint8_t> *out) {
  // Narrowing a 64-bit declared size into resize() would ask for the low bits
  // of it on a 32-bit target. ParseContainer already refuses every frame size
  // the container cannot justify; this is the same refusal stated where the
  // allocation is made, so no future caller can reach the resize without it.
  if (ExceedsSizeT(expected)) return PVZSTD_E_FORMAT;
  try {
    out->resize(static_cast<size_t>(expected));
  } catch (const std::exception &) {
    // Wider than std::bad_alloc on purpose: an oversized resize() throws
    // std::length_error, which is not a bad_alloc, so the narrower handler let
    // it out of the library and left the entry point's catch(...) to answer.
    return PVZSTD_E_NOMEM;
  }
  if (expected == 0) return PVZSTD_OK;
  const size_t got = DecompressInto(out->data(), out->size(), src, static_cast<size_t>(src_size));
  if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
  // A mismatch is the signature of a misparsed index: every frame still
  // decompresses, but each yields a neighbour's payload.
  if (got != expected) return PVZSTD_E_FORMAT;
  return PVZSTD_OK;
}

// Every length here is a 32-bit field read out of the file while `off` is a
// size_t, which is 32 bits on a WebAssembly target. So each bounds check is
// written against the bytes that remain -- subtracting from the frame's length
// rather than adding to an offset, and dividing rather than multiplying -- and
// the subtractions are ordered so a short frame cannot underflow one. Written
// the other way round, `off + ndim * 8 + PVZSTD_DTYPE_LEN` wraps for an ndim
// near 2^29 and turns the guard into a pass, after which the loop below reads
// eight bytes per dimension past the end of the container.
pvzstd_status ParseHeader(const std::vector<uint8_t> &buf, ArrayEntry *entry) {
  const uint64_t size = buf.size();
  uint64_t off = 0;
  if (size < 4) return PVZSTD_E_FORMAT;
  const uint32_t name_len = LoadU32(buf.data());
  off = 4;
  // size >= 4 == off, so the subtraction stands; the sum is in uint64, where a
  // uint32 name length plus four cannot wrap.
  if (size - off < static_cast<uint64_t>(name_len) + 4) return PVZSTD_E_FORMAT;
  entry->name.assign(reinterpret_cast<const char *>(buf.data() + static_cast<size_t>(off)),
                     name_len);
  off += name_len;

  const uint32_t ndim = LoadU32(buf.data() + static_cast<size_t>(off));
  off += 4;
  // off <= size holds from the check above, and the dtype field is subtracted
  // before the divide so neither step can underflow.
  if (size - off < PVZSTD_DTYPE_LEN) return PVZSTD_E_FORMAT;
  if ((size - off - PVZSTD_DTYPE_LEN) / 8 < ndim) return PVZSTD_E_FORMAT;
  entry->shape.clear();
  for (uint32_t i = 0; i < ndim; ++i) {
    entry->shape.push_back(LoadU64(buf.data() + static_cast<size_t>(off)));
    off += 8;
  }

  // dtype is space-padded to 16 bytes; strip trailing blanks.
  size_t dtype_len = PVZSTD_DTYPE_LEN;
  while (dtype_len > 0 && buf[static_cast<size_t>(off) + dtype_len - 1] == ' ') --dtype_len;
  std::memcpy(entry->dtype, buf.data() + static_cast<size_t>(off), dtype_len);
  entry->dtype[dtype_len] = '\0';
  off += PVZSTD_DTYPE_LEN;

  // Absent filter byte means PVZSTD_FILTER_NONE. Testing file_version instead
  // would mis-parse a shuffled version-2 file.
  entry->filter_id = PVZSTD_FILTER_NONE;
  if (off < size) {
    entry->filter_id = buf[static_cast<size_t>(off)];
    off += 1;
  }
  if (off != size) return PVZSTD_E_FORMAT;
  return PVZSTD_OK;
}

// Take in the thumbnail pair; a record this build does not read is left out.
void ReadThumbnail(pvzstd_reader *reader, const ArrayEntry &entry, uint64_t index_off,
                   uint64_t n_frames) {
  using pvzstd::detail::kSkippablePrefixBytes;
  const ContainerBytes &raw = reader->map;
  const uint64_t frame_len = entry.payload_end - entry.payload_start;
  if (frame_len < kSkippablePrefixBytes) return;
  const uint8_t *frame = raw.data() + entry.payload_start;
  if (LoadU32(frame) != pvzstd::detail::kSkippableMagic) return;
  const uint64_t record_len = LoadU32(frame + 4);
  if (record_len != frame_len - kSkippablePrefixBytes) return;
  const uint8_t *record = frame + kSkippablePrefixBytes;
  pvzstd::detail::ThumbnailRecord parsed;
  if (!pvzstd::detail::ParseThumbnailRecord(record, record_len, &parsed)) return;
  reader->thumbnail_record = record;
  reader->thumbnail = parsed;
  reader->thumbnail_stale =
      pvzstd::detail::IndexCrc(raw.data() + index_off, n_frames) != parsed.data_crc;
}

// Parse a container out of the bytes the reader already points at, however it
// came by them. Every entry point that produces a reader meets here, because a
// second copy of this walk would be a second set of bounds checks to keep in
// step with the first -- and it is these refusals, not the source of the bytes,
// that stand between a crafted container and an out-of-bounds read.
//
// Throws are left to the entry point's function-try handler.
pvzstd_status ParseContainer(pvzstd_reader *reader, uint32_t *file_version) {
  const ContainerBytes &raw = reader->map;
  if (raw.size() < kTrailerCountBytes) {
    return PVZSTD_E_FORMAT;
  }

  const uint64_t n_frames = LoadU64(raw.data() + raw.size() - kTrailerCountBytes);
  if (n_frames == 0 || (n_frames % 2) != 0) {
    return PVZSTD_E_FORMAT;  // frames pair as (header, payload)
  }
  // Bounded by division: n_frames comes from the file and n_frames * 16
  // overflows for a large enough value, turning the size check into a pass.
  if (n_frames > (raw.size() - kTrailerCountBytes) / kIndexEntryBytes) {
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
    // Refuse an impossible decompressed size here, before anything asks the
    // allocator for it. The index is file-supplied and unrelated to what the
    // frame actually holds, so a crafted one can name any 64-bit number; a
    // frame can only occupy bytes the container has, and kMaxExpansionRatio is
    // the most zstd can turn each of those into. Divided rather than multiplied
    // so the comparison itself cannot wrap. Relying on the resize to throw
    // instead is not enough: the throw is what a build with exceptions turned
    // off cannot answer, and on a 32-bit target the size would be truncated
    // before it ever reached the allocator.
    const uint64_t declared = sizes[static_cast<size_t>(i)];
    if (declared / kMaxExpansionRatio > raw.size() || ExceedsSizeT(declared)) {
      return PVZSTD_E_FORMAT;
    }
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
      return PVZSTD_E_FORMAT;
    }

    pvzstd_status st = DecompressFrame(raw.data() + hdr_start, hdr_end - hdr_start,
                                       sizes[static_cast<size_t>(i)], &frame);
    if (st != PVZSTD_OK) {
      return st;
    }

    ArrayEntry entry;
    st = ParseHeader(frame, &entry);
    if (st != PVZSTD_OK) {
      return st;
    }
    entry.nbytes = sizes[static_cast<size_t>(i + 1)];
    entry.payload_start = hdr_end;
    entry.payload_end = pay_end;
    if (!DeclaredSizeAgrees(entry.shape, entry.dtype, entry.nbytes)) {
      return PVZSTD_E_FORMAT;
    }

    // Only the fixed raw bytes of the thumbnail pair are taken as one, as by a reader
    // without zstd; any other pair 0 is an ordinary array whatever its name.
    if (i == 0 && pvzstd::detail::StartsWithThumbnailPrefix(raw.data(), raw.size())) {
      ReadThumbnail(reader, entry, index_off, n_frames);
      continue;
    }

    const bool is_ds = EndsWith(entry.name, kDsMetadataSuffix);
    const bool is_file = EndsWith(entry.name, kFileMetadataSuffix) ||
                         EndsWith(entry.name, kLegacyFileMetadataSuffix);
    if (is_ds || is_file) {
      std::vector<uint8_t> payload;
      st = DecompressFrame(raw.data() + entry.payload_start,
                           entry.payload_end - entry.payload_start, entry.nbytes, &payload);
      if (st != PVZSTD_OK) {
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

  // Refuse a container this build cannot decode before reading anything out of
  // it: a newer file may transform payloads in a way this build cannot invert.
  // The ceiling lives here, beside the decoder, so every front end agrees.
  if (reader->has_file_metadata) {
    long long version = 0;
    if (pvzstd::json::MemberInt(reader->file_metadata, "file_version", &version) && version >= 0) {
      if (file_version != nullptr) *file_version = static_cast<uint32_t>(version);
      if (version > static_cast<long long>(PVZSTD_FILE_VERSION_MAX)) {
        return PVZSTD_E_VERSION;
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

  return PVZSTD_OK;
}

}  // namespace

extern "C" {

pvzstd_status pvzstd_open_versioned(const char *path, pvzstd_reader **out,
                                    uint32_t *file_version) try {
  // Cleared before the argument check, not after: a refused open leaves *out at
  // NULL on every path out of here, so a caller who did not initialise it is
  // not left holding whatever was in that slot.
  if (out != nullptr) *out = nullptr;
  if (path == nullptr || out == nullptr) return PVZSTD_E_INVALID;

  // Owning: the parse below refuses the container on eleven paths and can throw
  // from a decompression sized by a field read out of the file, and every one of
  // those has to give back the fd and the mapping. A raw pointer leaked both.
  std::unique_ptr<pvzstd_reader> reader(new (std::nothrow) pvzstd_reader());
  if (reader == nullptr) return PVZSTD_E_NOMEM;

  pvzstd_status st = reader->map.Open(path);
  if (st != PVZSTD_OK) {
    return st;
  }

  st = ParseContainer(reader.get(), file_version);
  if (st != PVZSTD_OK) {
    return st;
  }

  *out = reader.release();
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_open(const char *path, pvzstd_reader **out) try {
  return pvzstd_open_versioned(path, out, nullptr);
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_open_memory_versioned(const void *data, uint64_t size, pvzstd_reader **out,
                                           uint32_t *file_version) try {
  // As in pvzstd_open_versioned: cleared first, so a NULL `data` or a zero
  // `size` leaves *out at NULL rather than untouched.
  if (out != nullptr) *out = nullptr;
  if (data == nullptr || size == 0 || out == nullptr) return PVZSTD_E_INVALID;

  std::unique_ptr<pvzstd_reader> reader(new (std::nothrow) pvzstd_reader());
  if (reader == nullptr) return PVZSTD_E_NOMEM;

  // Where the bytes came from is the only difference from the file case: no
  // path is opened and nothing is copied, and the parse below is the same call,
  // so a crafted buffer meets the trailer bounds check, the divide-first
  // frame-count guard and the declared-size check exactly as a crafted file does.
  pvzstd_status st = reader->map.Borrow(data, size);
  if (st != PVZSTD_OK) {
    return st;
  }

  st = ParseContainer(reader.get(), file_version);
  if (st != PVZSTD_OK) {
    return st;
  }

  *out = reader.release();
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_open_memory(const void *data, uint64_t size, pvzstd_reader **out) try {
  return pvzstd_open_memory_versioned(data, size, out, nullptr);
} catch (...) {
  return PVZSTD_E_NOMEM;
}

uint32_t pvzstd_max_file_version(void) { return PVZSTD_FILE_VERSION_MAX; }

void pvzstd_close(pvzstd_reader *reader) try { delete reader; } catch (...) {
}

uint64_t pvzstd_array_count(const pvzstd_reader *reader) try {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->arrays.size());
} catch (...) {
  return 0;
}

namespace {

// Every pointer aliases storage the reader owns for its whole life, so the filled
// struct stays valid until pvzstd_close().
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
                                   pvzstd_array_info *out) try {
  if (reader == nullptr || out == nullptr) return PVZSTD_E_INVALID;
  if (index >= reader->arrays.size()) return PVZSTD_E_RANGE;
  FillArrayInfo(reader->arrays[static_cast<size_t>(index)], out);
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_array_info_range(const pvzstd_reader *reader, uint64_t first, uint64_t count,
                                      pvzstd_array_info *out) try {
  if (reader == nullptr) return PVZSTD_E_INVALID;
  if (count == 0) return PVZSTD_OK;
  if (out == nullptr) return PVZSTD_E_INVALID;

  const uint64_t total = static_cast<uint64_t>(reader->arrays.size());
  // Against the remaining count, not first + count, which would wrap.
  if (first > total || count > total - first) return PVZSTD_E_RANGE;

  for (uint64_t i = 0; i < count; ++i) {
    FillArrayInfo(reader->arrays[static_cast<size_t>(first + i)], &out[i]);
  }
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

uint64_t pvzstd_field_array_count(const pvzstd_reader *reader) try {
  return reader == nullptr ? 0 : static_cast<uint64_t>(reader->field_names.size());
} catch (...) {
  return 0;
}

const char *pvzstd_field_array_name_at(const pvzstd_reader *reader, uint64_t index) try {
  if (reader == nullptr || index >= reader->field_names.size()) return nullptr;
  return reader->field_names[static_cast<size_t>(index)].c_str();
} catch (...) {
  return nullptr;
}

int64_t pvzstd_find_field_array(const pvzstd_reader *reader, const char *name) try {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->field_names.size(); ++i) {
    if (reader->field_names[i] == name) return reader->field_indices[i];
  }
  return -1;
} catch (...) {
  return -1;
}

int64_t pvzstd_find_array(const pvzstd_reader *reader, const char *name) try {
  if (reader == nullptr || name == nullptr) return -1;
  for (size_t i = 0; i < reader->arrays.size(); ++i) {
    if (reader->arrays[i].name == name) return static_cast<int64_t>(i);
  }
  return -1;
} catch (...) {
  return -1;
}

namespace {

// Read one array into `dst`, using up to `workers` threads for a payload stored
// as several frames and for reversing its filter.
pvzstd_status ReadArrayInto(const pvzstd_reader *reader, uint64_t index, void *dst,
                            uint64_t dst_size, int workers) try {
  if (index >= reader->arrays.size()) return PVZSTD_E_RANGE;
  const ArrayEntry &e = reader->arrays[static_cast<size_t>(index)];
  if (dst_size < e.nbytes) return PVZSTD_E_RANGE;
  if (e.nbytes == 0) return PVZSTD_OK;
  if (dst == nullptr) return PVZSTD_E_INVALID;

  const uint8_t *src = reader->map.data() + e.payload_start;
  const uint64_t src_size = e.payload_end - e.payload_start;

  if (e.filter_id == PVZSTD_FILTER_NONE) {
    return DecompressPayload(src, src_size, static_cast<uint8_t *>(dst), e.nbytes, workers);
  }

  if (e.filter_id != PVZSTD_FILTER_SHUFFLE) return PVZSTD_E_FILTER;

  const uint64_t itemsize = DtypeItemsize(e.dtype);
  if (itemsize == 0 || (e.nbytes % itemsize) != 0) return PVZSTD_E_FORMAT;

  // See DecompressFrame: the size is refused here, where the allocation is made.
  if (ExceedsSizeT(e.nbytes)) return PVZSTD_E_FORMAT;
  std::vector<uint8_t> filtered(static_cast<size_t>(e.nbytes));
  const pvzstd_status st = DecompressPayload(src, src_size, filtered.data(), e.nbytes, workers);
  if (st != PVZSTD_OK) return st;
  Unshuffle(filtered.data(), static_cast<uint8_t *>(dst), e.nbytes, itemsize, workers);
  return PVZSTD_OK;
} catch (...) {
  // Runs on ParallelStride workers, which cannot answer a throw.
  return PVZSTD_E_NOMEM;
}

// Threads for one array read on its own: every core once the payload is large
// enough for them to matter, as PVZSTD_THREADS_AUTO decides for a batch.
int WorkersFor(uint64_t nbytes) {
  if (!pvzstd::detail::kHasThreads || nbytes < kParallelDecompressFloor) return 1;
  return pvzstd::detail::HardwareWorkers();
}

}  // namespace

pvzstd_status pvzstd_read_array_at(const pvzstd_reader *reader, uint64_t index, void *dst,
                                   uint64_t dst_size) try {
  if (reader == nullptr) return PVZSTD_E_INVALID;
  if (index >= reader->arrays.size()) return PVZSTD_E_RANGE;
  const uint64_t nbytes = reader->arrays[static_cast<size_t>(index)].nbytes;
  return ReadArrayInto(reader, index, dst, dst_size, WorkersFor(nbytes));
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_read_arrays(const pvzstd_reader *reader, const uint64_t *indices,
                                 uint64_t count, void *const *dsts, const uint64_t *dst_sizes,
                                 int n_threads, uint64_t *failed_slot) try {
  if (failed_slot != nullptr) *failed_slot = PVZSTD_SLOT_NONE;
  if (reader == nullptr || indices == nullptr || dsts == nullptr || dst_sizes == nullptr) {
    return PVZSTD_E_INVALID;
  }
  if (count == 0) return PVZSTD_OK;
  // The parallel path indexes a vector of `count` results with a uint64 loop
  // counter. Narrowing the count into that vector's size on a 32-bit target
  // would size it from the low bits and then write past the end of it.
  if (ExceedsSizeT(count)) return PVZSTD_E_RANGE;

  int workers = n_threads;
  if (workers == PVZSTD_THREADS_AUTO) {
    // Size enters the decision, not just frame count. AUTO asks for the fastest
    // setting; an explicit n_threads is honoured.
    uint64_t total = 0;
    for (uint64_t i = 0; i < count; ++i) total += dst_sizes[i];
    workers = total < kParallelDecompressFloor ? 1 : pvzstd::detail::HardwareWorkers();
  } else if (workers < 0) {
    // Negative means every core, the same sign convention pvzstd_writer_set_threads
    // uses. It meant "inline" here until the clamp below was written against a
    // signed count, which is the opposite of what the two share an ABI to say.
    workers = pvzstd::detail::HardwareWorkers();
  }
  // Optimization: fewer arrays than workers leaves cores idle, so each array then gets a
  // share of them for its own sub-frames. In uint64: count is unbounded, and
  // narrowing it to int first could land on a negative and silently drop this
  // call to a single worker.
  int inner = 1;
  if (count < static_cast<uint64_t>(workers)) {
    inner = workers / static_cast<int>(count);
    workers = static_cast<int>(count);
  }
  // Same work in the same order, so a clamp rather than an error.
  if (!pvzstd::detail::kHasThreads) {
    workers = 1;
    inner = 1;
  }

  if (workers <= 1) {
    for (uint64_t i = 0; i < count; ++i) {
      const pvzstd_status st = ReadArrayInto(reader, indices[i], dsts[i], dst_sizes[i], inner);
      if (st != PVZSTD_OK) {
        if (failed_slot != nullptr) *failed_slot = i;
        return st;
      }
    }
    return PVZSTD_OK;
  }

  // Static striding rather than a work queue; each slot is written by one thread.
  std::vector<pvzstd_status> results(static_cast<size_t>(count), PVZSTD_OK);
  pvzstd::detail::ParallelStride(workers, count, [&](uint64_t i) {
    results[static_cast<size_t>(i)] =
        ReadArrayInto(reader, indices[i], dsts[i], dst_sizes[i], inner);
  });

  for (uint64_t i = 0; i < count; ++i) {
    if (results[static_cast<size_t>(i)] != PVZSTD_OK) {
      if (failed_slot != nullptr) *failed_slot = i;
      return results[static_cast<size_t>(i)];
    }
  }
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

const char *pvzstd_ds_metadata_json(const pvzstd_reader *reader) try {
  if (reader == nullptr || !reader->has_ds_metadata) return nullptr;
  return reader->ds_metadata.c_str();
} catch (...) {
  return nullptr;
}

const char *pvzstd_file_metadata_json(const pvzstd_reader *reader) try {
  if (reader == nullptr || !reader->has_file_metadata) return nullptr;
  return reader->file_metadata.c_str();
} catch (...) {
  return nullptr;
}

uint64_t pvzstd_metadata_count(const pvzstd_reader *reader) try {
  if (reader == nullptr) return 0;
  return reader->metadata_docs.size();
} catch (...) {
  return 0;
}

const char *pvzstd_metadata_name_at(const pvzstd_reader *reader, uint64_t index) try {
  if (reader == nullptr || index >= reader->metadata_names.size()) return nullptr;
  return reader->metadata_names[static_cast<size_t>(index)].c_str();
} catch (...) {
  return nullptr;
}

const char *pvzstd_metadata_json_at(const pvzstd_reader *reader, uint64_t index) try {
  if (reader == nullptr || index >= reader->metadata_docs.size()) return nullptr;
  return reader->metadata_docs[static_cast<size_t>(index)].c_str();
} catch (...) {
  return nullptr;
}

uint64_t pvzstd_frame_count(const pvzstd_reader *reader) try {
  if (reader == nullptr) return 0;
  return reader->frame_decompressed.size();
} catch (...) {
  return 0;
}

pvzstd_status pvzstd_frame_sizes(const pvzstd_reader *reader, uint64_t *decompressed,
                                 uint64_t *compressed, uint64_t capacity) try {
  if (reader == nullptr) return PVZSTD_E_INVALID;
  const size_t n = reader->frame_decompressed.size();
  // The caller sized these from pvzstd_frame_count(), a separate call: nothing
  // else here can tell a correctly sized buffer from one allocated against a
  // different reader, so refuse rather than write past the end.
  if (capacity < static_cast<uint64_t>(n)) return PVZSTD_E_RANGE;
  if (decompressed != nullptr) {
    std::memcpy(decompressed, reader->frame_decompressed.data(), n * sizeof(uint64_t));
  }
  if (compressed != nullptr) {
    std::memcpy(compressed, reader->frame_compressed.data(), n * sizeof(uint64_t));
  }
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
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
      // through this code too.
      return "zstd rejected a frame or a compression parameter";
    case PVZSTD_E_RANGE:
      return "index or count out of range, or destination buffer too small";
    case PVZSTD_E_NOMEM:
      return "allocation failed";
    case PVZSTD_E_FILTER:
      return "array uses a filter this build cannot reverse";
    case PVZSTD_E_INVALID:
      return "invalid argument";
    case PVZSTD_E_UNSUPPORTED:
      return "the container is a shape this operation cannot serve";
    case PVZSTD_E_EXISTS:
      return "an array of that name is already in the container";
    case PVZSTD_E_VERSION:
      return "the container's file version is newer than this build can decode";
    case PVZSTD_E_BUSY:
      return "another append holds this container, or left its lock file behind";
    case PVZSTD_E_CHANGED:
      return "another writer replaced the container while this call was staging its result";
  }
  return "unknown status";
}

uint32_t pvzstd_abi_version(void) { return PVZSTD_ABI_VERSION; }

pvzstd_status pvzstd_thumbnail(const pvzstd_reader *reader, pvzstd_thumbnail_info *out) try {
  if (reader == nullptr || out == nullptr) return PVZSTD_E_INVALID;
  *out = pvzstd_thumbnail_info{};
  if (reader->thumbnail_record == nullptr) return PVZSTD_OK;
  out->png = reader->thumbnail_record + reader->thumbnail.png_offset;
  out->nbytes = reader->thumbnail.png_nbytes;
  out->width = reader->thumbnail.width;
  out->height = reader->thumbnail.height;
  out->stale = reader->thumbnail_stale ? 1 : 0;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

}  // extern "C"
