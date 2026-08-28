// Append half of the .pv container. See doc/format/container-v2.md.
//
// Existing frames are copied verbatim by offset and only the two that grow are
// regenerated. Two details carry byte-identity: the reference append compresses
// with threads=0 (unlike pvzstd_writer_write, which sizes a pool from the payload
// total), and the dataset-metadata JSON is spliced rather than re-emitted.

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "detail.h"
#include "fileio.h"
#include "json_read.h"
#include "pvzstd/pvzstd.h"

using namespace pvzstd::detail;  // NOLINT(google-build-using-namespace)
using namespace pvzstd::json;    // NOLINT(google-build-using-namespace)

namespace {

constexpr const char kDsMetadataKey[] = "__ds_metadata";
constexpr const char kMultiblockKey[] = "__multiblock__ds_metadata";
constexpr const char kFileMetadataKey[] = "__pyvista_zstd_metadata";
constexpr const char kFieldDataSuffix[] = "__field_data";
constexpr size_t kUidNChar = 16;
constexpr size_t kCopyChunk = 8u << 20;
constexpr int kFileVersionShuffle = 1;

class ScopedFile {
 public:
  explicit ScopedFile(std::FILE *fp) : fp_(fp) {}
  ~ScopedFile() { reset(nullptr); }
  ScopedFile(const ScopedFile &) = delete;
  ScopedFile &operator=(const ScopedFile &) = delete;

  std::FILE *get() const { return fp_; }
  void reset(std::FILE *fp) {
    if (fp_ != nullptr) std::fclose(fp_);
    fp_ = fp;
  }

 private:
  std::FILE *fp_;
};

struct FrameEntry {
  uint64_t end = 0;
  uint64_t decomp = 0;
};

bool ReadAt(std::FILE *fp, uint64_t offset, uint64_t len, std::vector<uint8_t> *out) {
  // Every caller's `len` is bounded by the container's own length, which is not
  // itself bounded by size_t on a 32-bit target: a container larger than the
  // address space would be read into the low bits of its length. Refusing it
  // here also bounds the frame count, whose table is read through this call --
  // a count past size_t needs a table of 16 times it, which cannot pass.
  if (ExceedsSizeT(len)) return false;
  if (SeekTo(fp, static_cast<int64_t>(offset), SEEK_SET) != 0) return false;
  try {
    out->resize(static_cast<size_t>(len));
  } catch (const std::exception &) {
    // Wider than std::bad_alloc on purpose: an oversized resize() throws
    // std::length_error, which is not a bad_alloc, so the narrower handler let
    // it out of the library.
    return false;
  }
  return len == 0 || std::fread(out->data(), 1, static_cast<size_t>(len), fp) == len;
}

// The trailer: per frame [cumulative_end:u64][decompressed_size:u64], then the
// frame count as the final 8 bytes.
pvzstd_status ReadFooter(std::FILE *fp, std::vector<FrameEntry> *frames, uint64_t *body_end) {
  if (SeekTo(fp, 0, SEEK_END) != 0) return PVZSTD_E_IO;
  const int64_t end = TellAt(fp);
  if (end < 8) return PVZSTD_E_FORMAT;
  const uint64_t size = static_cast<uint64_t>(end);

  std::vector<uint8_t> buf;
  if (!ReadAt(fp, size - 8, 8, &buf)) return PVZSTD_E_IO;
  const uint64_t n_frames = LoadU64(buf.data());
  // An empty or odd count cannot be a (header, payload) pairing.
  if (n_frames < 4 || (n_frames % 2) != 0) return PVZSTD_E_FORMAT;
  if (n_frames > (size - 8) / 16) return PVZSTD_E_FORMAT;

  const uint64_t table_bytes = n_frames * 16;
  if (!ReadAt(fp, size - 8 - table_bytes, table_bytes, &buf)) return PVZSTD_E_IO;
  frames->resize(static_cast<size_t>(n_frames));
  uint64_t prev = 0;
  for (uint64_t i = 0; i < n_frames; ++i) {
    (*frames)[i].end = LoadU64(buf.data() + i * 16);
    (*frames)[i].decomp = LoadU64(buf.data() + i * 16 + 8);
    // Ends are cumulative, so they must not go backwards or past the trailer.
    if ((*frames)[i].end < prev) return PVZSTD_E_FORMAT;
    prev = (*frames)[i].end;
  }
  if (prev > size - 8 - table_bytes) return PVZSTD_E_FORMAT;
  *body_end = prev;
  return PVZSTD_OK;
}

uint64_t FrameStart(const std::vector<FrameEntry> &frames, size_t i) {
  return i == 0 ? 0 : frames[i - 1].end;
}

pvzstd_status DecompressFrame(std::FILE *fp, const std::vector<FrameEntry> &frames, size_t index,
                              std::string *out) {
  const uint64_t start = FrameStart(frames, index);
  const uint64_t len = frames[index].end - start;
  std::vector<uint8_t> raw;
  if (!ReadAt(fp, start, len, &raw)) return PVZSTD_E_IO;
  std::string plain;
  // The trailer's decompressed size is a file-supplied 64-bit field that
  // ReadFooter has no way to check against the frame's contents, so a crafted
  // container can name any number here. Narrowing it would ask for its low bits
  // and then compare the result against a size the parse still believes.
  if (ExceedsSizeT(frames[index].decomp)) return PVZSTD_E_FORMAT;
  try {
    plain.resize(static_cast<size_t>(frames[index].decomp));
  } catch (const std::exception &) {
    // See ReadAt: std::length_error is not a std::bad_alloc.
    return PVZSTD_E_NOMEM;
  }
  const size_t got = ZSTD_decompress(plain.data(), plain.size(), raw.data(), raw.size());
  if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
  // Trailer and body disagreeing: trusting either would be a guess.
  if (got != plain.size()) return PVZSTD_E_FORMAT;
  *out = std::move(plain);
  return PVZSTD_OK;
}

bool EndsWith(const std::string &s, const char *suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

struct StagedFrame {
  bool copy = false;  // copied verbatim from the source
  uint64_t src_start = 0;
  uint64_t src_end = 0;
  std::vector<uint8_t> bytes;  // when !copy
  uint64_t decomp = 0;
};

}  // namespace

extern "C" {

pvzstd_status pvzstd_append_arrays(const char *path, const pvzstd_append_array *arrays,
                                   uint64_t count, int level, pvzstd_shuffle_mode shuffle,
                                   uint64_t *clash_slot, uint32_t *found_version) try {
  if (clash_slot != nullptr) *clash_slot = PVZSTD_SLOT_NONE;
  if (found_version != nullptr) *found_version = 0;
  if (path == nullptr) return PVZSTD_E_INVALID;
  if (count == 0) return PVZSTD_OK;  // appending nothing is a no-op, not an error
  if (arrays == nullptr) return PVZSTD_E_INVALID;

  for (uint64_t k = 0; k < count; ++k) {
    const pvzstd_append_array &a = arrays[k];
    if (a.name == nullptr || a.dtype == nullptr || a.dtype_name == nullptr) return PVZSTD_E_INVALID;
    if (a.ndim > 0 && a.shape == nullptr) return PVZSTD_E_INVALID;
    if (a.nbytes > 0 && a.data == nullptr) return PVZSTD_E_INVALID;
    // The payload buffer is sized from the narrowed nbytes while ShuffleBytes is
    // driven from the full 64-bit one, so a length past size_t writes the shape
    // of the whole array into a buffer holding its low bits. No caller can own
    // that many bytes on a target where the check is live, so it is a bad
    // argument rather than a shortage.
    if (ExceedsSizeT(a.nbytes)) return PVZSTD_E_INVALID;
    if (std::strlen(a.dtype) > PVZSTD_DTYPE_LEN) return PVZSTD_E_INVALID;
    if (!ParseDtype(a.dtype).valid) return PVZSTD_E_INVALID;
  }

  // Held for the whole read-modify-write below, and released by its destructor
  // on every way out of this function. Without it two appends both read the
  // container, both add their own arrays to what they read, and the second to
  // commit replaces the first one's result -- a lost update, which no amount of
  // checking at the end can turn back into two writes.
  AppendLock lock;
  const LockResult locked = lock.Acquire(path);
  if (locked == LockResult::kHeld) return PVZSTD_E_BUSY;
  if (locked != LockResult::kAcquired) return PVZSTD_E_IO;

  ScopedFile src(std::fopen(path, "rb"));
  if (src.get() == nullptr) return PVZSTD_E_IO;
  // Which file `path` names right now. Everything below is staged against this
  // one, and step 7 checks that it is still the file being replaced. The lock
  // covers the appends that take it; this covers everything else that can
  // replace a file.
  const FileIdentity opened_as = IdentifyOpen(src.get());

  std::vector<FrameEntry> frames;
  uint64_t body_end = 0;
  pvzstd_status st = ReadFooter(src.get(), &frames, &body_end);
  if (st != PVZSTD_OK) return st;
  const size_t n_arrays = frames.size() / 2;
  const size_t file_meta_idx = n_arrays - 1;

  // 1. The file metadata is always the final array.
  std::string file_meta_json;
  st = DecompressFrame(src.get(), frames, file_meta_idx * 2 + 1, &file_meta_json);
  if (st != PVZSTD_OK) return st;

  std::vector<std::string> frame_names;
  long long old_level = 0;
  long long old_version = 0;
  std::string compression;
  if (!MemberStringArray(file_meta_json, "frame_names", &frame_names) ||
      !MemberInt(file_meta_json, "compression_level", &old_level) ||
      !MemberInt(file_meta_json, "file_version", &old_version) ||
      !MemberString(file_meta_json, "compression", &compression)) {
    return PVZSTD_E_FORMAT;
  }
  // Same ceiling the reader applies, for the same reason: a container this build
  // cannot decode is one it must not edit either. Appending would regenerate the
  // two metadata frames -- restamping a version whose meaning is unknown here --
  // and leave a file neither this build nor the one that wrote it can trust.
  if (old_version > static_cast<long long>(PVZSTD_FILE_VERSION_MAX)) {
    if (found_version != nullptr) {
      // The field is what the file says; a value past uint32 is a number this
      // build cannot report, not one it can round down and still be honest about.
      *found_version = old_version > static_cast<long long>(UINT32_MAX)
                           ? UINT32_MAX
                           : static_cast<uint32_t>(old_version);
    }
    return PVZSTD_E_VERSION;
  }

  // The file-metadata array's own name is absent from frame_names; disagreeing
  // counts would shift every name onto the wrong frame.
  if (frame_names.size() != n_arrays - 1) return PVZSTD_E_FORMAT;

  size_t root_idx = frame_names.size();
  for (size_t i = 0; i < frame_names.size(); ++i) {
    // MultiBlock metadata ends with the same suffix but has no root dataset to
    // append to; refuse rather than misparse it as a dataset's. Reported apart
    // from PVZSTD_E_FORMAT because the file is intact -- it is this operation that
    // has nowhere to put the arrays, and a caller told "format error" would
    // report a good container as damaged.
    if (EndsWith(frame_names[i], kMultiblockKey)) return PVZSTD_E_UNSUPPORTED;
    if (root_idx == frame_names.size() && EndsWith(frame_names[i], kDsMetadataKey)) root_idx = i;
  }
  if (root_idx == frame_names.size()) return PVZSTD_E_FORMAT;
  if (frame_names[root_idx].size() < kUidNChar) return PVZSTD_E_FORMAT;
  const std::string ds_id = frame_names[root_idx].substr(0, kUidNChar);

  // 2. The root dataset metadata, which is the document that grows.
  std::string ds_json;
  st = DecompressFrame(src.get(), frames, root_idx * 2 + 1, &ds_json);
  if (st != PVZSTD_OK) return st;

  size_t fdk_open = 0;
  size_t fdk_past = 0;
  if (!MemberObjectSpan(ds_json, "field_data_keys", &fdk_open, &fdk_past)) return PVZSTD_E_FORMAT;
  std::vector<std::string> existing;
  if (!ObjectKeys(ds_json, fdk_open, &existing)) return PVZSTD_E_FORMAT;
  for (uint64_t k = 0; k < count; ++k) {
    for (const std::string &have : existing) {
      // Refused, not overwritten: the old bytes would stay with nothing pointing
      // at them and the reader would surface whichever entry it found first.
      if (have == arrays[k].name) {
        if (clash_slot != nullptr) *clash_slot = k;
        return PVZSTD_E_EXISTS;
      }
    }
    for (uint64_t j = 0; j < k; ++j) {
      if (std::strcmp(arrays[j].name, arrays[k].name) == 0) {
        if (clash_slot != nullptr) *clash_slot = k;
        return PVZSTD_E_EXISTS;
      }
    }
  }

  const int use_level = (level == PVZSTD_LEVEL_FROM_FILE) ? static_cast<int>(old_level) : level;

  // 3. Stage the new frames. Kept frames are recorded as offset ranges; only
  //    the new and regenerated ones are built in memory.
  std::vector<StagedFrame> staged;
  std::vector<std::string> final_names;
  for (size_t ai = 0; ai < n_arrays; ++ai) {
    if (ai == root_idx || ai == file_meta_idx) continue;
    for (size_t f = ai * 2; f <= ai * 2 + 1; ++f) {
      StagedFrame keep;
      keep.copy = true;
      keep.src_start = FrameStart(frames, f);
      keep.src_end = frames[f].end;
      keep.decomp = frames[f].decomp;
      staged.push_back(std::move(keep));
    }
    final_names.push_back(frame_names[ai]);
  }

  std::string fdk_addition;
  bool any_shuffled = false;
  std::vector<std::vector<uint8_t>> plain;  // uncompressed payloads, in order
  for (uint64_t k = 0; k < count; ++k) {
    const pvzstd_append_array &a = arrays[k];
    const std::string frame_name = ds_id + a.name + kFieldDataSuffix;
    const Dtype d = ParseDtype(a.dtype);

    bool use_shuffle = false;
    if (shuffle != PVZSTD_SHUFFLE_NEVER && d.itemsize > 1) {
      if (shuffle == PVZSTD_SHUFFLE_ALWAYS) {
        use_shuffle = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use_shuffle = AutoShuffleBeneficial(static_cast<const uint8_t *>(a.data), a.nbytes,
                                            d.itemsize, use_level);
      }
    }
    const uint8_t filter = use_shuffle ? PVZSTD_FILTER_SHUFFLE : PVZSTD_FILTER_NONE;
    any_shuffled = any_shuffled || use_shuffle;

    std::vector<uint64_t> shape(a.shape, a.shape + a.ndim);
    try {
      plain.push_back(PackArrayMetadata(frame_name, a.dtype, shape, filter));
      std::vector<uint8_t> payload(static_cast<size_t>(a.nbytes));
      if (a.nbytes > 0) {
        const uint8_t *raw = static_cast<const uint8_t *>(a.data);
        if (use_shuffle) {
          ShuffleBytes(raw, payload.data(), a.nbytes, d.itemsize);
        } else {
          std::memcpy(payload.data(), raw, static_cast<size_t>(a.nbytes));
        }
      }
      plain.push_back(std::move(payload));
    } catch (const std::exception &) {
      // See ReadAt: std::length_error is not a std::bad_alloc.
      return PVZSTD_E_NOMEM;
    }
    final_names.push_back(frame_name);

    if (!fdk_addition.empty() || !existing.empty()) fdk_addition.push_back(',');
    AppendFieldEntry(&fdk_addition, a.name, a.dtype_name, a.shape, a.ndim);
  }

  // 4. Regenerate the dataset metadata by splicing, leaving every byte outside
  //    the field_data_keys object exactly as it was.
  std::string ds_new = ds_json;
  ds_new.insert(fdk_past - 1, fdk_addition);
  const std::string ds_name = ds_id + kDsMetadataKey;
  final_names.push_back(ds_name);

  // 5. Regenerate the file metadata, in the reference dataclass's field order.
  const long long new_version =
      any_shuffled ? (old_version > kFileVersionShuffle ? old_version : kFileVersionShuffle)
                   : old_version;
  std::string file_new = "{\"frame_names\":[";
  for (size_t i = 0; i < final_names.size(); ++i) {
    if (i != 0) file_new.push_back(',');
    AppendJsonString(&file_new, final_names[i]);
  }
  file_new += "],\"compression_level\":" + std::to_string(old_level);
  file_new += ",\"compression\":";
  AppendJsonString(&file_new, compression);
  file_new += ",\"file_version\":" + std::to_string(new_version) + "}";

  try {
    plain.push_back(PackArrayMetadata(ds_name, "|u1", {ds_new.size()}, PVZSTD_FILTER_NONE));
    plain.emplace_back(ds_new.begin(), ds_new.end());
    plain.push_back(
        PackArrayMetadata(kFileMetadataKey, "|u1", {file_new.size()}, PVZSTD_FILTER_NONE));
    plain.emplace_back(file_new.begin(), file_new.end());
  } catch (const std::exception &) {
    // See ReadAt: std::length_error is not a std::bad_alloc. No guard in front
    // of this one -- both documents are already-allocated std::strings, so
    // there is no 64-bit field to narrow.
    return PVZSTD_E_NOMEM;
  }

  // 6. Compress single-threaded, matching the reference append's threads=0.
  for (std::vector<uint8_t> &payload : plain) {
    StagedFrame frame;
    frame.decomp = payload.size();
    st = CompressFrame(payload.data(), payload.size(), use_level, 0, &frame.bytes);
    if (st != PVZSTD_OK) return st;
    staged.push_back(std::move(frame));
  }

  // 7. Commit by rename, so an interrupted append cannot damage what was there.
  //    The staging file is named by the OS rather than derived from `path`: a
  //    name derived from the container is the same name for every caller, so
  //    two appends running at once both write into it and the file the rename
  //    commits is a mixture of the two.
  std::string tmp_path;
  ScopedFile out(OpenUniqueTemp(path, src.get(), &tmp_path));
  if (out.get() == nullptr) return PVZSTD_E_IO;

  std::vector<uint8_t> chunk;
  std::vector<uint8_t> trailer;
  uint64_t offset = 0;
  bool ok = true;
  for (const StagedFrame &frame : staged) {
    if (frame.copy) {
      uint64_t remaining = frame.src_end - frame.src_start;
      if (SeekTo(src.get(), static_cast<int64_t>(frame.src_start), SEEK_SET) != 0) {
        ok = false;
        break;
      }
      while (remaining > 0 && ok) {
        const uint64_t want = remaining < kCopyChunk ? remaining : kCopyChunk;
        chunk.resize(static_cast<size_t>(want));
        if (std::fread(chunk.data(), 1, chunk.size(), src.get()) != chunk.size() ||
            std::fwrite(chunk.data(), 1, chunk.size(), out.get()) != chunk.size()) {
          ok = false;
          break;
        }
        remaining -= want;
        offset += want;
      }
    } else if (std::fwrite(frame.bytes.data(), 1, frame.bytes.size(), out.get()) !=
               frame.bytes.size()) {
      ok = false;
    } else {
      offset += frame.bytes.size();
    }
    if (!ok) break;
    StoreU64(&trailer, offset);
    StoreU64(&trailer, frame.decomp);
  }
  if (ok) {
    StoreU64(&trailer, staged.size());
    ok = std::fwrite(trailer.data(), 1, trailer.size(), out.get()) == trailer.size();
  }
  if (ok) ok = std::fflush(out.get()) == 0;
  out.reset(nullptr);
  if (!ok) {
    std::remove(tmp_path.c_str());
    return PVZSTD_E_IO;
  }

  // Still the container this call read? A writer that does not take the lock --
  // another tool, or a plain move onto this path -- has a result that this
  // rename would replace with a body copied before that result existed. A
  // rename swaps the directory entry, so the file `path` names is no longer the
  // file open here.
  //
  // This is a check and not a second lock: it catches a replacement made while
  // this call was staging, which is the whole slow part, and not one made in
  // the microseconds between here and the rename below.
  if (!SameFile(opened_as, IdentifyPath(path))) {
    std::remove(tmp_path.c_str());
    return PVZSTD_E_CHANGED;
  }

  // Windows will not rename onto an open handle, and the source is still open.
  src.reset(nullptr);
  if (std::rename(tmp_path.c_str(), path) != 0) {
    // Some platforms refuse a rename onto an existing file. Move the original
    // aside rather than deleting it: removing it first and then failing the
    // second rename would leave the caller with neither file, which is the one
    // outcome committing by rename exists to prevent.
    const std::string kept_path = tmp_path + ".orig";
    std::remove(kept_path.c_str());
    if (std::rename(path, kept_path.c_str()) != 0) {
      std::remove(tmp_path.c_str());
      return PVZSTD_E_IO;
    }
    if (std::rename(tmp_path.c_str(), path) != 0) {
      // Put the original back before reporting; the append did not happen.
      std::rename(kept_path.c_str(), path);
      std::remove(tmp_path.c_str());
      return PVZSTD_E_IO;
    }
    std::remove(kept_path.c_str());
  }
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

}  // extern "C"
