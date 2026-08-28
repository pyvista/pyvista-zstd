// Streaming append: the same file edit as pvzstd_append_arrays, held open.
//
// pvzstd_append_arrays copies the whole file per call, so cost grows with it: over
// 40 commits the per-commit cost rose 4.24x. A stream keeps the trailer, metadata
// offset and dataset-metadata document, so a commit costs what it adds. Output is
// byte-identical.
//
// The tradeoff is crash behaviour: a stream writes in place, so an interrupted
// commit leaves a trailer describing frames that were not fully written.

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "detail.h"
#include "fileio.h"
#include "json_read.h"
#include "pvzstd/pvzstd.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace pvzstd::detail;  // NOLINT(google-build-using-namespace)
using namespace pvzstd::json;    // NOLINT(google-build-using-namespace)

namespace {

constexpr const char kDsMetadataKey[] = "__ds_metadata";
constexpr const char kMultiblockKey[] = "__multiblock__ds_metadata";
constexpr const char kFileMetadataKey[] = "__pyvista_zstd_metadata";
constexpr const char kFieldDataSuffix[] = "__field_data";
constexpr size_t kUidNChar = 16;
constexpr int kFileVersionShuffle = 1;
// The metadata tail is two arrays -- dataset then file -- so four frames.
constexpr size_t kTailFrames = 4;

bool EndsWith(const std::string &s, const char *suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool TruncateTo(std::FILE *fp, uint64_t size) {
  if (std::fflush(fp) != 0) return false;
#if defined(_WIN32)
  return _chsize_s(_fileno(fp), static_cast<__int64>(size)) == 0;
#else
  return ::ftruncate(fileno(fp), static_cast<off_t>(size)) == 0;
#endif
}

struct FrameEntry {
  uint64_t end = 0;     // cumulative compressed end offset
  uint64_t decomp = 0;  // decompressed size
};

}  // namespace

struct pvzstd_stream {
  std::FILE *fp = nullptr;
  std::string path;

  // Frames in file order; the last four are the tail each commit rebuilds.
  std::vector<FrameEntry> frames;
  // One name per array, data arrays only; metadata names are added with the tail.
  std::vector<std::string> names;
  uint64_t body_end = 0;  // where the metadata tail begins

  std::string ds_id;
  std::string ds_name;
  std::string ds_json;  // spliced in place, never re-read from disk
  std::string compression;
  int level = kDefaultLevel;
  long long file_version = 0;

  uint64_t commits = 0;
  bool closed = false;
  // Set once a commit has begun mutating state and then failed. The frame list,
  // the name list and body_end are updated across several steps, so a failure
  // part-way leaves them describing frames that are not where they say they are.
  // Retrying on top of that writes a file that parses and gives wrong data.
  bool failed = false;
};

namespace {

// Decompress frame `index` of the open file into `out`.
pvzstd_status ReadFrame(pvzstd_stream *s, const std::vector<uint64_t> &ends,
                        const std::vector<uint64_t> &sizes, size_t index, std::string *out) {
  const uint64_t start = (index == 0) ? 0 : ends[index - 1];
  const uint64_t n = ends[index] - start;
  std::vector<uint8_t> comp;
  // Both lengths come out of the trailer. The compressed one is bounded only by
  // the container's own length, which is not itself bounded by size_t on a
  // 32-bit target; the decompressed one is a field nothing checks against the
  // frame's contents, so a crafted container can name any number. Narrowing
  // either would allocate its low bits and then read or write against the value
  // the parse still believes.
  if (ExceedsSizeT(n) || ExceedsSizeT(sizes[index])) return PVZSTD_E_FORMAT;
  try {
    comp.resize(static_cast<size_t>(n));
    out->resize(static_cast<size_t>(sizes[index]));
  } catch (const std::exception &) {
    // Wider than std::bad_alloc on purpose: an oversized resize() throws
    // std::length_error, which is not a bad_alloc, so the narrower handler let
    // it out past the cleanup this open does on every other failure.
    return PVZSTD_E_NOMEM;
  }
  if (SeekTo(s->fp, static_cast<int64_t>(start), SEEK_SET) != 0) return PVZSTD_E_IO;
  if (n > 0 && std::fread(comp.data(), 1, comp.size(), s->fp) != comp.size()) return PVZSTD_E_IO;
  if (sizes[index] == 0) return PVZSTD_OK;
  const size_t got = ZSTD_decompress(&(*out)[0], out->size(), comp.data(), comp.size());
  if (ZSTD_isError(got) != 0) return PVZSTD_E_ZSTD;
  if (got != out->size()) return PVZSTD_E_FORMAT;
  return PVZSTD_OK;
}

// Emit the tail at body_end, write the trailer, and truncate: the tail can shrink,
// and a stale suffix would leave the trailer no longer at the end.
pvzstd_status WriteTail(pvzstd_stream *s) {
  std::vector<std::string> final_names = s->names;
  final_names.push_back(s->ds_name);

  std::string file_new = "{\"frame_names\":[";
  for (size_t i = 0; i < final_names.size(); ++i) {
    if (i != 0) file_new.push_back(',');
    AppendJsonString(&file_new, final_names[i]);
  }
  file_new += "],\"compression_level\":" + std::to_string(s->level);
  file_new += ",\"compression\":";
  AppendJsonString(&file_new, s->compression);
  file_new += ",\"file_version\":" + std::to_string(s->file_version) + "}";

  std::vector<std::vector<uint8_t>> plain;
  try {
    plain.push_back(PackArrayMetadata(s->ds_name, "|u1", {s->ds_json.size()}, PVZSTD_FILTER_NONE));
    plain.emplace_back(s->ds_json.begin(), s->ds_json.end());
    plain.push_back(
        PackArrayMetadata(kFileMetadataKey, "|u1", {file_new.size()}, PVZSTD_FILTER_NONE));
    plain.emplace_back(file_new.begin(), file_new.end());
  } catch (const std::exception &) {
    // See ReadFrame. No guard in front: both documents are already-allocated
    // std::strings, so there is no 64-bit field to narrow.
    return PVZSTD_E_NOMEM;
  }

  // Plain statuses here: WriteTail is only reached from a commit, which poisons
  // the stream on any non-OK return of its own.
  if (SeekTo(s->fp, static_cast<int64_t>(s->body_end), SEEK_SET) != 0) return PVZSTD_E_IO;
  uint64_t offset = s->body_end;
  for (std::vector<uint8_t> &payload : plain) {
    std::vector<uint8_t> comp;
    // Single-threaded, matching the reference append.
    const pvzstd_status st = CompressFrame(payload.data(), payload.size(), s->level, 0, &comp);
    if (st != PVZSTD_OK) return st;
    if (std::fwrite(comp.data(), 1, comp.size(), s->fp) != comp.size()) return PVZSTD_E_IO;
    offset += comp.size();
    s->frames.push_back({offset, payload.size()});
  }

  std::vector<uint8_t> trailer;
  for (const FrameEntry &f : s->frames) {
    StoreU64(&trailer, f.end);
    StoreU64(&trailer, f.decomp);
  }
  StoreU64(&trailer, s->frames.size());
  if (std::fwrite(trailer.data(), 1, trailer.size(), s->fp) != trailer.size()) return PVZSTD_E_IO;

  return TruncateTo(s->fp, offset + trailer.size()) ? PVZSTD_OK : PVZSTD_E_IO;
}

}  // namespace

extern "C" {

pvzstd_status pvzstd_stream_open(const char *path, pvzstd_stream **out) try {
  if (path == nullptr || out == nullptr) return PVZSTD_E_INVALID;
  *out = nullptr;

  pvzstd_stream *s = new (std::nothrow) pvzstd_stream();
  if (s == nullptr) return PVZSTD_E_NOMEM;
  s->path = path;
  s->fp = std::fopen(path, "r+b");
  if (s->fp == nullptr) {
    delete s;
    return PVZSTD_E_IO;
  }

  // One parse, at open. Everything an append needs is kept from here on.
  auto fail = [&](pvzstd_status st) {
    std::fclose(s->fp);
    delete s;
    return st;
  };

  if (SeekTo(s->fp, 0, SEEK_END) != 0) return fail(PVZSTD_E_IO);
  const int64_t file_size = TellAt(s->fp);
  if (file_size < 8) return fail(PVZSTD_E_FORMAT);

  if (SeekTo(s->fp, -8, SEEK_END) != 0) return fail(PVZSTD_E_IO);
  uint8_t count_buf[8];
  if (std::fread(count_buf, 1, 8, s->fp) != 8) return fail(PVZSTD_E_IO);
  const uint64_t n_frames = LoadU64(count_buf);
  if (n_frames < kTailFrames || (n_frames % 2) != 0) return fail(PVZSTD_E_FORMAT);
  // Bounded by division: n_frames comes from the file and n_frames * 16 overflows
  // for a large enough value.
  if (n_frames > (static_cast<uint64_t>(file_size) - 8) / 16) return fail(PVZSTD_E_FORMAT);
  // The three vectors below are sized from the narrowed count and then indexed
  // by a 64-bit loop counter, and the table is 16 bytes per frame -- so on a
  // 32-bit target a count past size_t, or one whose table is, sizes them from
  // the low bits and the loop writes past the end. Both are refused before
  // anything is allocated.
  if (ExceedsSizeT(n_frames) || ExceedsSizeT(n_frames * 16)) return fail(PVZSTD_E_FORMAT);

  std::vector<uint64_t> ends(static_cast<size_t>(n_frames));
  std::vector<uint64_t> sizes(static_cast<size_t>(n_frames));
  {
    std::vector<uint8_t> table;
    try {
      table.resize(static_cast<size_t>(n_frames) * 16);
    } catch (const std::exception &) {
      // See ReadFrame: std::length_error is not a std::bad_alloc, and letting
      // it past `fail` leaks this stream and the file it holds open.
      return fail(PVZSTD_E_NOMEM);
    }
    if (SeekTo(s->fp, -static_cast<int64_t>(table.size() + 8), SEEK_END) != 0) {
      return fail(PVZSTD_E_IO);
    }
    if (std::fread(table.data(), 1, table.size(), s->fp) != table.size()) return fail(PVZSTD_E_IO);
    uint64_t prev = 0;
    for (uint64_t i = 0; i < n_frames; ++i) {
      ends[static_cast<size_t>(i)] = LoadU64(table.data() + i * 16);
      sizes[static_cast<size_t>(i)] = LoadU64(table.data() + i * 16 + 8);
      if (ends[static_cast<size_t>(i)] < prev) return fail(PVZSTD_E_FORMAT);
      prev = ends[static_cast<size_t>(i)];
    }
  }

  const size_t n_arrays = static_cast<size_t>(n_frames) / 2;
  std::string file_meta_json;
  pvzstd_status st = ReadFrame(s, ends, sizes, (n_arrays - 1) * 2 + 1, &file_meta_json);
  if (st != PVZSTD_OK) return fail(st);

  std::vector<std::string> frame_names;
  long long level = 0;
  long long version = 0;
  if (!MemberStringArray(file_meta_json, "frame_names", &frame_names) ||
      !MemberInt(file_meta_json, "compression_level", &level) ||
      !MemberInt(file_meta_json, "file_version", &version) ||
      !MemberString(file_meta_json, "compression", &s->compression)) {
    return fail(PVZSTD_E_FORMAT);
  }
  // The trailing file-metadata array's own name is deliberately absent.
  if (frame_names.size() != n_arrays - 1) return fail(PVZSTD_E_FORMAT);
  s->level = static_cast<int>(level);
  // Same ceiling the reader and the append apply: a commit re-stamps this
  // version, and re-stamping one whose meaning is unknown here is the edit
  // both of those refuse to make.
  if (version > static_cast<long long>(PVZSTD_FILE_VERSION_MAX)) return fail(PVZSTD_E_VERSION);
  s->file_version = version;

  size_t root_idx = frame_names.size();
  for (size_t i = 0; i < frame_names.size(); ++i) {
    // A MultiBlock container has no single root dataset to stream into. Same
    // refusal pvzstd_append_arrays makes, reported under the same code: one
    // condition answered two ways is a condition callers cannot handle once.
    if (EndsWith(frame_names[i], kMultiblockKey)) return fail(PVZSTD_E_UNSUPPORTED);
    if (root_idx == frame_names.size() && EndsWith(frame_names[i], kDsMetadataKey)) root_idx = i;
  }
  if (root_idx == frame_names.size()) return fail(PVZSTD_E_FORMAT);
  if (frame_names[root_idx].size() < kUidNChar) return fail(PVZSTD_E_FORMAT);
  // Rewriting the tail in place requires the two metadata arrays to be the tail.
  if (root_idx != n_arrays - 2) return fail(PVZSTD_E_FORMAT);
  s->ds_id = frame_names[root_idx].substr(0, kUidNChar);
  s->ds_name = frame_names[root_idx];

  st = ReadFrame(s, ends, sizes, root_idx * 2 + 1, &s->ds_json);
  if (st != PVZSTD_OK) return fail(st);
  size_t fdk_open = 0;
  size_t fdk_past = 0;
  if (!MemberObjectSpan(s->ds_json, "field_data_keys", &fdk_open, &fdk_past)) {
    return fail(PVZSTD_E_FORMAT);
  }

  for (size_t i = 0; i < root_idx; ++i) s->names.push_back(frame_names[i]);
  for (uint64_t i = 0; i < n_frames; ++i) {
    s->frames.push_back({ends[static_cast<size_t>(i)], sizes[static_cast<size_t>(i)]});
  }
  s->body_end = (root_idx == 0) ? 0 : ends[root_idx * 2 - 1];

  *out = s;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

// Poison the stream and report why. Used only past the point where a commit has
// started mutating state; validation failures before that leave it usable.
static pvzstd_status Fail(pvzstd_stream *s, pvzstd_status status) {
  s->failed = true;
  return status;
}

pvzstd_status pvzstd_stream_append(pvzstd_stream *s, const pvzstd_append_array *arrays,
                                   uint64_t count, pvzstd_shuffle_mode shuffle) try {
  if (s == nullptr) return PVZSTD_E_INVALID;
  if (s->closed) return PVZSTD_E_INVALID;
  if (s->failed) return PVZSTD_E_INVALID;
  if (count == 0) return PVZSTD_OK;
  if (arrays == nullptr) return PVZSTD_E_INVALID;

  std::vector<std::string> existing;
  size_t fdk_open = 0;
  size_t fdk_past = 0;
  if (!MemberObjectSpan(s->ds_json, "field_data_keys", &fdk_open, &fdk_past) ||
      !ObjectKeys(s->ds_json, fdk_open, &existing)) {
    return PVZSTD_E_FORMAT;
  }
  for (uint64_t k = 0; k < count; ++k) {
    const pvzstd_append_array &a = arrays[k];
    if (a.name == nullptr || a.dtype == nullptr || a.dtype_name == nullptr) return PVZSTD_E_INVALID;
    if (a.ndim > 0 && a.shape == nullptr) return PVZSTD_E_INVALID;
    if (a.nbytes > 0 && a.data == nullptr) return PVZSTD_E_INVALID;
    // The payload buffer is sized from the narrowed nbytes while ShuffleBytes is
    // driven from the full 64-bit one, so a length past size_t writes the shape
    // of the whole array into a buffer holding its low bits. Same refusal
    // pvzstd_append_arrays makes, for the same staging code.
    if (ExceedsSizeT(a.nbytes)) return PVZSTD_E_INVALID;
    if (std::strlen(a.dtype) > PVZSTD_DTYPE_LEN) return PVZSTD_E_INVALID;
    if (!ParseDtype(a.dtype).valid) return PVZSTD_E_INVALID;
    for (const std::string &have : existing) {
      // Refused, not overwritten: the old bytes would stay with nothing pointing
      // at them.
      if (have == arrays[k].name) return PVZSTD_E_EXISTS;
    }
    for (uint64_t j = 0; j < k; ++j) {
      if (std::strcmp(arrays[j].name, arrays[k].name) == 0) return PVZSTD_E_EXISTS;
    }
  }

  // Drop the tail's frame entries; WriteTail re-adds them.
  if (s->frames.size() < kTailFrames) return PVZSTD_E_FORMAT;
  s->frames.resize(s->frames.size() - kTailFrames);

  if (SeekTo(s->fp, static_cast<int64_t>(s->body_end), SEEK_SET) != 0) {
    return Fail(s, PVZSTD_E_IO);
  }
  uint64_t offset = s->body_end;
  std::string fdk_addition;

  for (uint64_t k = 0; k < count; ++k) {
    const pvzstd_append_array &a = arrays[k];
    const std::string frame_name = s->ds_id + a.name + kFieldDataSuffix;
    const Dtype d = ParseDtype(a.dtype);

    bool use_shuffle = false;
    if (shuffle != PVZSTD_SHUFFLE_NEVER && d.itemsize > 1) {
      if (shuffle == PVZSTD_SHUFFLE_ALWAYS) {
        use_shuffle = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use_shuffle = AutoShuffleBeneficial(static_cast<const uint8_t *>(a.data), a.nbytes,
                                            d.itemsize, s->level);
      }
    }
    const uint8_t filter = use_shuffle ? PVZSTD_FILTER_SHUFFLE : PVZSTD_FILTER_NONE;
    if (use_shuffle && s->file_version < kFileVersionShuffle) {
      s->file_version = kFileVersionShuffle;
    }

    std::vector<uint64_t> shape(a.shape, a.shape + a.ndim);
    std::vector<std::vector<uint8_t>> pair;
    try {
      pair.push_back(PackArrayMetadata(frame_name, a.dtype, shape, filter));
      std::vector<uint8_t> payload(static_cast<size_t>(a.nbytes));
      if (a.nbytes > 0) {
        const uint8_t *raw = static_cast<const uint8_t *>(a.data);
        if (use_shuffle) {
          ShuffleBytes(raw, payload.data(), a.nbytes, d.itemsize);
        } else {
          std::memcpy(payload.data(), raw, static_cast<size_t>(a.nbytes));
        }
      }
      pair.push_back(std::move(payload));
    } catch (const std::exception &) {
      // See ReadFrame: std::length_error is not a std::bad_alloc.
      return Fail(s, PVZSTD_E_NOMEM);
    }

    for (std::vector<uint8_t> &payload : pair) {
      std::vector<uint8_t> comp;
      const pvzstd_status st = CompressFrame(payload.data(), payload.size(), s->level, 0, &comp);
      if (st != PVZSTD_OK) return Fail(s, st);
      if (std::fwrite(comp.data(), 1, comp.size(), s->fp) != comp.size()) {
        return Fail(s, PVZSTD_E_IO);
      }
      offset += comp.size();
      s->frames.push_back({offset, payload.size()});
    }

    s->names.push_back(frame_name);
    if (!fdk_addition.empty() || !existing.empty()) fdk_addition.push_back(',');
    AppendFieldEntry(&fdk_addition, a.name, a.dtype_name, a.shape, a.ndim);
    existing.push_back(a.name);
  }

  s->body_end = offset;
  // Splice, never regenerate: every byte outside field_data_keys stays as the
  // writer left it, so we need not reproduce another library's key order.
  s->ds_json.insert(fdk_past - 1, fdk_addition);

  const pvzstd_status st = WriteTail(s);
  if (st != PVZSTD_OK) return Fail(s, st);
  ++s->commits;
  return PVZSTD_OK;
} catch (...) {
  return Fail(s, PVZSTD_E_NOMEM);
}

uint64_t pvzstd_stream_commit_count(const pvzstd_stream *s) try {
  return s == nullptr ? 0 : s->commits;
} catch (...) {
  return 0;
}

pvzstd_status pvzstd_stream_close(pvzstd_stream *s, uint64_t expected_commits) try {
  if (s == nullptr) return PVZSTD_E_INVALID;
  if (s->closed) return PVZSTD_OK;
  // Refused rather than closed: what is on disk past the last good commit is
  // whatever the failed one left, and this cannot say where that ends.
  if (s->failed) return PVZSTD_E_INVALID;
  // A short stream presented as complete reads back perfectly, with results
  // missing.
  if (s->commits != expected_commits) return PVZSTD_E_RANGE;
  // Flushed before it counts as closed: marking it first would let a second call
  // return OK for a stream whose last bytes never reached the file.
  if (std::fflush(s->fp) != 0) return Fail(s, PVZSTD_E_IO);
  s->closed = true;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

void pvzstd_stream_free(pvzstd_stream *s) try {
  if (s == nullptr) return;
  if (s->fp != nullptr) std::fclose(s->fp);
  delete s;
} catch (...) {
}

}  // extern "C"
