// Streaming append: the same file edit as pvz_append_arrays, held open.
//
// pvz_append_arrays copies the whole file per call, so cost grows with it: over
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

struct pvz_stream {
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
pvz_status ReadFrame(pvz_stream *s, const std::vector<uint64_t> &ends,
                     const std::vector<uint64_t> &sizes, size_t index, std::string *out) {
  const uint64_t start = (index == 0) ? 0 : ends[index - 1];
  const uint64_t n = ends[index] - start;
  std::vector<uint8_t> comp;
  try {
    comp.resize(static_cast<size_t>(n));
    out->resize(static_cast<size_t>(sizes[index]));
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  if (SeekTo(s->fp, static_cast<int64_t>(start), SEEK_SET) != 0) return PVZ_E_IO;
  if (n > 0 && std::fread(comp.data(), 1, comp.size(), s->fp) != comp.size()) return PVZ_E_IO;
  if (sizes[index] == 0) return PVZ_OK;
  const size_t got = ZSTD_decompress(&(*out)[0], out->size(), comp.data(), comp.size());
  if (ZSTD_isError(got) != 0) return PVZ_E_ZSTD;
  if (got != out->size()) return PVZ_E_FORMAT;
  return PVZ_OK;
}

// Emit the tail at body_end, write the trailer, and truncate: the tail can shrink,
// and a stale suffix would leave the trailer no longer at the end.
pvz_status WriteTail(pvz_stream *s) {
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
    plain.push_back(PackArrayMetadata(s->ds_name, "|u1", {s->ds_json.size()}, PVZ_FILTER_NONE));
    plain.emplace_back(s->ds_json.begin(), s->ds_json.end());
    plain.push_back(PackArrayMetadata(kFileMetadataKey, "|u1", {file_new.size()}, PVZ_FILTER_NONE));
    plain.emplace_back(file_new.begin(), file_new.end());
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }

  // Plain statuses here: WriteTail is only reached from a commit, which poisons
  // the stream on any non-OK return of its own.
  if (SeekTo(s->fp, static_cast<int64_t>(s->body_end), SEEK_SET) != 0) return PVZ_E_IO;
  uint64_t offset = s->body_end;
  for (std::vector<uint8_t> &payload : plain) {
    std::vector<uint8_t> comp;
    // Single-threaded, matching the reference append.
    const pvz_status st = CompressFrame(payload.data(), payload.size(), s->level, 0, &comp);
    if (st != PVZ_OK) return st;
    if (std::fwrite(comp.data(), 1, comp.size(), s->fp) != comp.size()) return PVZ_E_IO;
    offset += comp.size();
    s->frames.push_back({offset, payload.size()});
  }

  std::vector<uint8_t> trailer;
  for (const FrameEntry &f : s->frames) {
    StoreU64(&trailer, f.end);
    StoreU64(&trailer, f.decomp);
  }
  StoreU64(&trailer, s->frames.size());
  if (std::fwrite(trailer.data(), 1, trailer.size(), s->fp) != trailer.size()) return PVZ_E_IO;

  return TruncateTo(s->fp, offset + trailer.size()) ? PVZ_OK : PVZ_E_IO;
}

}  // namespace

extern "C" {

pvz_status pvz_stream_open(const char *path, pvz_stream **out) try {
  if (path == nullptr || out == nullptr) return PVZ_E_INVALID;
  *out = nullptr;

  pvz_stream *s = new (std::nothrow) pvz_stream();
  if (s == nullptr) return PVZ_E_NOMEM;
  s->path = path;
  s->fp = std::fopen(path, "r+b");
  if (s->fp == nullptr) {
    delete s;
    return PVZ_E_IO;
  }

  // One parse, at open. Everything an append needs is kept from here on.
  auto fail = [&](pvz_status st) {
    std::fclose(s->fp);
    delete s;
    return st;
  };

  if (SeekTo(s->fp, 0, SEEK_END) != 0) return fail(PVZ_E_IO);
  const int64_t file_size = TellAt(s->fp);
  if (file_size < 8) return fail(PVZ_E_FORMAT);

  if (SeekTo(s->fp, -8, SEEK_END) != 0) return fail(PVZ_E_IO);
  uint8_t count_buf[8];
  if (std::fread(count_buf, 1, 8, s->fp) != 8) return fail(PVZ_E_IO);
  const uint64_t n_frames = LoadU64(count_buf);
  if (n_frames < kTailFrames || (n_frames % 2) != 0) return fail(PVZ_E_FORMAT);
  // Bounded by division: n_frames comes from the file and n_frames * 16 overflows
  // for a large enough value.
  if (n_frames > (static_cast<uint64_t>(file_size) - 8) / 16) return fail(PVZ_E_FORMAT);

  std::vector<uint64_t> ends(static_cast<size_t>(n_frames));
  std::vector<uint64_t> sizes(static_cast<size_t>(n_frames));
  {
    std::vector<uint8_t> table;
    try {
      table.resize(static_cast<size_t>(n_frames) * 16);
    } catch (const std::bad_alloc &) {
      return fail(PVZ_E_NOMEM);
    }
    if (SeekTo(s->fp, -static_cast<int64_t>(table.size() + 8), SEEK_END) != 0) {
      return fail(PVZ_E_IO);
    }
    if (std::fread(table.data(), 1, table.size(), s->fp) != table.size()) return fail(PVZ_E_IO);
    uint64_t prev = 0;
    for (uint64_t i = 0; i < n_frames; ++i) {
      ends[static_cast<size_t>(i)] = LoadU64(table.data() + i * 16);
      sizes[static_cast<size_t>(i)] = LoadU64(table.data() + i * 16 + 8);
      if (ends[static_cast<size_t>(i)] < prev) return fail(PVZ_E_FORMAT);
      prev = ends[static_cast<size_t>(i)];
    }
  }

  const size_t n_arrays = static_cast<size_t>(n_frames) / 2;
  std::string file_meta_json;
  pvz_status st = ReadFrame(s, ends, sizes, (n_arrays - 1) * 2 + 1, &file_meta_json);
  if (st != PVZ_OK) return fail(st);

  std::vector<std::string> frame_names;
  long long level = 0;
  long long version = 0;
  if (!MemberStringArray(file_meta_json, "frame_names", &frame_names) ||
      !MemberInt(file_meta_json, "compression_level", &level) ||
      !MemberInt(file_meta_json, "file_version", &version) ||
      !MemberString(file_meta_json, "compression", &s->compression)) {
    return fail(PVZ_E_FORMAT);
  }
  // The trailing file-metadata array's own name is deliberately absent.
  if (frame_names.size() != n_arrays - 1) return fail(PVZ_E_FORMAT);
  s->level = static_cast<int>(level);
  s->file_version = version;

  size_t root_idx = frame_names.size();
  for (size_t i = 0; i < frame_names.size(); ++i) {
    // A MultiBlock container has no single root dataset to stream into. Same
    // refusal pvz_append_arrays makes, reported under the same code: one
    // condition answered two ways is a condition callers cannot handle once.
    if (EndsWith(frame_names[i], kMultiblockKey)) return fail(PVZ_E_UNSUPPORTED);
    if (root_idx == frame_names.size() && EndsWith(frame_names[i], kDsMetadataKey)) root_idx = i;
  }
  if (root_idx == frame_names.size()) return fail(PVZ_E_FORMAT);
  if (frame_names[root_idx].size() < kUidNChar) return fail(PVZ_E_FORMAT);
  // Rewriting the tail in place requires the two metadata arrays to be the tail.
  if (root_idx != n_arrays - 2) return fail(PVZ_E_FORMAT);
  s->ds_id = frame_names[root_idx].substr(0, kUidNChar);
  s->ds_name = frame_names[root_idx];

  st = ReadFrame(s, ends, sizes, root_idx * 2 + 1, &s->ds_json);
  if (st != PVZ_OK) return fail(st);
  size_t fdk_open = 0;
  size_t fdk_past = 0;
  if (!MemberObjectSpan(s->ds_json, "field_data_keys", &fdk_open, &fdk_past)) {
    return fail(PVZ_E_FORMAT);
  }

  for (size_t i = 0; i < root_idx; ++i) s->names.push_back(frame_names[i]);
  for (uint64_t i = 0; i < n_frames; ++i) {
    s->frames.push_back({ends[static_cast<size_t>(i)], sizes[static_cast<size_t>(i)]});
  }
  s->body_end = (root_idx == 0) ? 0 : ends[root_idx * 2 - 1];

  *out = s;
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

// Poison the stream and report why. Used only past the point where a commit has
// started mutating state; validation failures before that leave it usable.
static pvz_status Fail(pvz_stream *s, pvz_status status) {
  s->failed = true;
  return status;
}

pvz_status pvz_stream_append(pvz_stream *s, const pvz_append_array *arrays, uint64_t count,
                             pvz_shuffle_mode shuffle) try {
  if (s == nullptr) return PVZ_E_INVALID;
  if (s->closed) return PVZ_E_INVALID;
  if (s->failed) return PVZ_E_INVALID;
  if (count == 0) return PVZ_OK;
  if (arrays == nullptr) return PVZ_E_INVALID;

  std::vector<std::string> existing;
  size_t fdk_open = 0;
  size_t fdk_past = 0;
  if (!MemberObjectSpan(s->ds_json, "field_data_keys", &fdk_open, &fdk_past) ||
      !ObjectKeys(s->ds_json, fdk_open, &existing)) {
    return PVZ_E_FORMAT;
  }
  for (uint64_t k = 0; k < count; ++k) {
    if (arrays[k].name == nullptr || arrays[k].dtype == nullptr ||
        arrays[k].dtype_name == nullptr) {
      return PVZ_E_INVALID;
    }
    for (const std::string &have : existing) {
      // Refused, not overwritten: the old bytes would stay with nothing pointing
      // at them.
      if (have == arrays[k].name) return PVZ_E_EXISTS;
    }
    for (uint64_t j = 0; j < k; ++j) {
      if (std::strcmp(arrays[j].name, arrays[k].name) == 0) return PVZ_E_EXISTS;
    }
  }

  // Drop the tail's frame entries; WriteTail re-adds them.
  if (s->frames.size() < kTailFrames) return PVZ_E_FORMAT;
  s->frames.resize(s->frames.size() - kTailFrames);

  if (SeekTo(s->fp, static_cast<int64_t>(s->body_end), SEEK_SET) != 0) {
    return Fail(s, PVZ_E_IO);
  }
  uint64_t offset = s->body_end;
  std::string fdk_addition;

  for (uint64_t k = 0; k < count; ++k) {
    const pvz_append_array &a = arrays[k];
    const std::string frame_name = s->ds_id + a.name + kFieldDataSuffix;
    const Dtype d = ParseDtype(a.dtype);

    bool use_shuffle = false;
    if (shuffle != PVZ_SHUFFLE_NEVER && d.itemsize > 1) {
      if (shuffle == PVZ_SHUFFLE_ALWAYS) {
        use_shuffle = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use_shuffle = AutoShuffleBeneficial(static_cast<const uint8_t *>(a.data), a.nbytes,
                                            d.itemsize, s->level);
      }
    }
    const uint8_t filter = use_shuffle ? PVZ_FILTER_SHUFFLE : PVZ_FILTER_NONE;
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
    } catch (const std::bad_alloc &) {
      return Fail(s, PVZ_E_NOMEM);
    }

    for (std::vector<uint8_t> &payload : pair) {
      std::vector<uint8_t> comp;
      const pvz_status st = CompressFrame(payload.data(), payload.size(), s->level, 0, &comp);
      if (st != PVZ_OK) return Fail(s, st);
      if (std::fwrite(comp.data(), 1, comp.size(), s->fp) != comp.size()) {
        return Fail(s, PVZ_E_IO);
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

  const pvz_status st = WriteTail(s);
  if (st != PVZ_OK) return Fail(s, st);
  ++s->commits;
  return PVZ_OK;
} catch (...) {
  return Fail(s, PVZ_E_NOMEM);
}

uint64_t pvz_stream_commit_count(const pvz_stream *s) try {
  return s == nullptr ? 0 : s->commits;
} catch (...) {
  return 0;
}

pvz_status pvz_stream_close(pvz_stream *s, uint64_t expected_commits) try {
  if (s == nullptr) return PVZ_E_INVALID;
  if (s->closed) return PVZ_OK;
  // Refused rather than closed: what is on disk past the last good commit is
  // whatever the failed one left, and this cannot say where that ends.
  if (s->failed) return PVZ_E_INVALID;
  // A short stream presented as complete reads back perfectly, with results
  // missing.
  if (s->commits != expected_commits) return PVZ_E_RANGE;
  // Flushed before it counts as closed: marking it first would let a second call
  // return OK for a stream whose last bytes never reached the file.
  if (std::fflush(s->fp) != 0) return Fail(s, PVZ_E_IO);
  s->closed = true;
  return PVZ_OK;
} catch (...) {
  return PVZ_E_NOMEM;
}

void pvz_stream_free(pvz_stream *s) try {
  if (s == nullptr) return;
  if (s->fp != nullptr) std::fclose(s->fp);
  delete s;
} catch (...) {
}

}  // extern "C"
