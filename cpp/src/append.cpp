// Append half of the .pv container. See doc/format/container-v2.md.
//
// Adding to an existing container is not "write it again with more arrays".
// The frames already on disk are copied verbatim by offset -- never
// decompressed, never recompressed -- and only the two frames that grow (the
// dataset metadata and the trailing file metadata) are regenerated. That is
// what makes the cost proportional to what is being added rather than to the
// size of the file, and it is also why byte-identity here is a different
// problem from byte-identity in the writer: most of the output is a copy, and
// the part that is not has to agree exactly with what the reference append
// would have produced.
//
// Two details decide most of that agreement:
//
//   * the reference append compresses with threads=0, so these frames are
//     single-threaded -- unlike pvz_writer_write, which sizes a worker pool
//     from the payload total. Same level, different framing;
//   * the dataset-metadata JSON is edited by splicing into its
//     "field_data_keys" object rather than by parsing and re-emitting the
//     document. Re-emitting would have to reproduce another library's key
//     order and number formatting for every field this format may ever carry;
//     splicing only has to reproduce the entries actually being added, and
//     leaves every other byte untouched by construction.

#include "pvzstd/pvzstd.h"

#include "detail.h"

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace pvzstd::detail;  // NOLINT(google-build-using-namespace)

namespace {

constexpr const char kDsMetadataKey[] = "__ds_metadata";
constexpr const char kMultiblockKey[] = "__multiblock__ds_metadata";
constexpr const char kFileMetadataKey[] = "__pyvista_zstd_metadata";
constexpr const char kFieldDataSuffix[] = "__field_data";
constexpr size_t kUidNChar = 16;
constexpr size_t kCopyChunk = 8u << 20;
constexpr int kFileVersionShuffle = 1;

// Closes on every exit path, including the error ones, of which this file has
// a great many.
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
  if (std::fseek(fp, static_cast<long>(offset), SEEK_SET) != 0) return false;
  try {
    out->resize(static_cast<size_t>(len));
  } catch (const std::bad_alloc &) {
    return false;
  }
  return len == 0 || std::fread(out->data(), 1, static_cast<size_t>(len), fp) == len;
}

// The trailer: per frame [cumulative_end:u64][decompressed_size:u64], then the
// frame count as the final 8 bytes.
pvz_status ReadFooter(std::FILE *fp, std::vector<FrameEntry> *frames, uint64_t *body_end) {
  if (std::fseek(fp, 0, SEEK_END) != 0) return PVZ_E_IO;
  const long end = std::ftell(fp);
  if (end < 8) return PVZ_E_FORMAT;
  const uint64_t size = static_cast<uint64_t>(end);

  std::vector<uint8_t> buf;
  if (!ReadAt(fp, size - 8, 8, &buf)) return PVZ_E_IO;
  const uint64_t n_frames = LoadU64(buf.data());
  // An empty or odd frame count cannot be a (header, payload) pairing, and a
  // count that does not fit in the file is a truncation, not a container.
  if (n_frames < 4 || (n_frames % 2) != 0) return PVZ_E_FORMAT;
  if (n_frames > (size - 8) / 16) return PVZ_E_FORMAT;

  const uint64_t table_bytes = n_frames * 16;
  if (!ReadAt(fp, size - 8 - table_bytes, table_bytes, &buf)) return PVZ_E_IO;
  frames->resize(static_cast<size_t>(n_frames));
  uint64_t prev = 0;
  for (uint64_t i = 0; i < n_frames; ++i) {
    (*frames)[i].end = LoadU64(buf.data() + i * 16);
    (*frames)[i].decomp = LoadU64(buf.data() + i * 16 + 8);
    // Ends are cumulative, so they must not go backwards or past the trailer.
    if ((*frames)[i].end < prev) return PVZ_E_FORMAT;
    prev = (*frames)[i].end;
  }
  if (prev > size - 8 - table_bytes) return PVZ_E_FORMAT;
  *body_end = prev;
  return PVZ_OK;
}

uint64_t FrameStart(const std::vector<FrameEntry> &frames, size_t i) {
  return i == 0 ? 0 : frames[i - 1].end;
}

pvz_status DecompressFrame(std::FILE *fp, const std::vector<FrameEntry> &frames, size_t index,
                           std::string *out) {
  const uint64_t start = FrameStart(frames, index);
  const uint64_t len = frames[index].end - start;
  std::vector<uint8_t> raw;
  if (!ReadAt(fp, start, len, &raw)) return PVZ_E_IO;
  std::string plain;
  try {
    plain.resize(static_cast<size_t>(frames[index].decomp));
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  const size_t got = ZSTD_decompress(plain.data(), plain.size(), raw.data(), raw.size());
  if (ZSTD_isError(got) != 0) return PVZ_E_ZSTD;
  // A frame that decompressed to a different length than the trailer declared
  // means the trailer and the body disagree; trusting either would be a guess.
  if (got != plain.size()) return PVZ_E_FORMAT;
  *out = std::move(plain);
  return PVZ_OK;
}

// --- a small JSON reader ---------------------------------------------
//
// Deliberately small: it reads members of the two metadata documents this
// format defines, both of which are emitted by a compact json.dumps. It is not
// a general JSON parser and does not try to be one.

bool SkipWs(const std::string &s, size_t *i) {
  while (*i < s.size() && (s[*i] == ' ' || s[*i] == '\n' || s[*i] == '\t' || s[*i] == '\r')) ++*i;
  return *i < s.size();
}

// Reads a string starting at the opening quote; leaves *i past the close.
bool ReadJsonString(const std::string &s, size_t *i, std::string *out) {
  if (*i >= s.size() || s[*i] != '"') return false;
  ++*i;
  out->clear();
  while (*i < s.size()) {
    const char c = s[*i];
    if (c == '"') {
      ++*i;
      return true;
    }
    if (c == '\\') {
      if (*i + 1 >= s.size()) return false;
      const char esc = s[*i + 1];
      switch (esc) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'u': {
          // Only the escapes json.dumps emits for the identifiers this format
          // carries; a real code point outside Latin-1 would need UTF-8
          // re-encoding, which no name in this format uses.
          if (*i + 5 >= s.size()) return false;
          unsigned code = 0;
          for (int k = 0; k < 4; ++k) {
            const char h = s[*i + 2 + k];
            code <<= 4;
            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
            else return false;
          }
          if (code > 0x7F) return false;
          out->push_back(static_cast<char>(code));
          *i += 4;
          break;
        }
        default: return false;
      }
      *i += 2;
      continue;
    }
    out->push_back(c);
    ++*i;
  }
  return false;
}

// Advances *i past the value beginning at *i, whatever kind it is.
bool SkipValue(const std::string &s, size_t *i) {
  if (!SkipWs(s, i)) return false;
  const char c = s[*i];
  if (c == '"') {
    std::string ignored;
    return ReadJsonString(s, i, &ignored);
  }
  if (c == '{' || c == '[') {
    const char open = c;
    const char close = (c == '{') ? '}' : ']';
    int depth = 0;
    while (*i < s.size()) {
      const char d = s[*i];
      if (d == '"') {
        std::string ignored;
        if (!ReadJsonString(s, i, &ignored)) return false;
        continue;
      }
      if (d == open) ++depth;
      else if (d == close && --depth == 0) {
        ++*i;
        return true;
      }
      ++*i;
    }
    return false;
  }
  // number, true, false, null
  while (*i < s.size() && s[*i] != ',' && s[*i] != '}' && s[*i] != ']') ++*i;
  return true;
}

// Position of the value of a top-level member, or npos. Scans members rather
// than searching for the literal `"key":`, so a member whose *value* happens to
// contain that text -- a user-supplied array name, say -- cannot be mistaken
// for the member itself.
size_t FindMember(const std::string &s, const char *key) {
  size_t i = 0;
  if (!SkipWs(s, &i) || s[i] != '{') return std::string::npos;
  ++i;
  while (SkipWs(s, &i)) {
    if (s[i] == '}') return std::string::npos;
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::string name;
    if (!ReadJsonString(s, &i, &name)) return std::string::npos;
    if (!SkipWs(s, &i) || s[i] != ':') return std::string::npos;
    ++i;
    if (!SkipWs(s, &i)) return std::string::npos;
    if (name == key) return i;
    if (!SkipValue(s, &i)) return std::string::npos;
  }
  return std::string::npos;
}

bool MemberString(const std::string &s, const char *key, std::string *out) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos) return false;
  return ReadJsonString(s, &i, out);
}

bool MemberInt(const std::string &s, const char *key, long long *out) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos) return false;
  const size_t start = i;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
  size_t digits = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
    ++i;
    ++digits;
  }
  if (digits == 0) return false;
  *out = std::strtoll(s.substr(start, i - start).c_str(), nullptr, 10);
  return true;
}

bool MemberStringArray(const std::string &s, const char *key, std::vector<std::string> *out) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos || i >= s.size() || s[i] != '[') return false;
  ++i;
  out->clear();
  while (SkipWs(s, &i)) {
    if (s[i] == ']') return true;
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::string item;
    if (!ReadJsonString(s, &i, &item)) return false;
    out->push_back(std::move(item));
  }
  return false;
}

// The member's opening brace and the index just past its closing brace.
bool MemberObjectSpan(const std::string &s, const char *key, size_t *open, size_t *past_close) {
  size_t i = FindMember(s, key);
  if (i == std::string::npos || i >= s.size() || s[i] != '{') return false;
  *open = i;
  if (!SkipValue(s, &i)) return false;
  *past_close = i;
  return true;
}

bool ObjectKeys(const std::string &s, size_t open, std::vector<std::string> *out) {
  size_t i = open;
  if (s[i] != '{') return false;
  ++i;
  out->clear();
  while (SkipWs(s, &i)) {
    if (s[i] == '}') return true;
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::string name;
    if (!ReadJsonString(s, &i, &name)) return false;
    if (!SkipWs(s, &i) || s[i] != ':') return false;
    ++i;
    if (!SkipValue(s, &i)) return false;
    out->push_back(std::move(name));
  }
  return false;
}

bool EndsWith(const std::string &s, const char *suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// One "<name>":{"shape":[...],"dtype":"..."} entry, in the reference
// dataclass's field order and with json.dumps' compact separators.
void AppendFieldEntry(std::string *out, const std::string &name, const std::string &dtype_name,
                      const uint64_t *shape, uint32_t ndim) {
  AppendJsonString(out, name);
  *out += ":{\"shape\":[";
  for (uint32_t d = 0; d < ndim; ++d) {
    if (d != 0) out->push_back(',');
    *out += std::to_string(shape[d]);
  }
  *out += "],\"dtype\":";
  AppendJsonString(out, dtype_name);
  out->push_back('}');
}

struct StagedFrame {
  bool copy = false;      // copied verbatim from the source
  uint64_t src_start = 0;
  uint64_t src_end = 0;
  std::vector<uint8_t> bytes;  // when !copy
  uint64_t decomp = 0;
};

}  // namespace

extern "C" {

pvz_status pvz_append_arrays(const char *path, const pvz_append_array *arrays, uint64_t count,
                             int level, pvz_shuffle_mode shuffle) {
  if (path == nullptr) return PVZ_E_INVALID;
  if (count == 0) return PVZ_OK;  // appending nothing is a no-op, not an error
  if (arrays == nullptr) return PVZ_E_INVALID;

  for (uint64_t k = 0; k < count; ++k) {
    const pvz_append_array &a = arrays[k];
    if (a.name == nullptr || a.dtype == nullptr || a.dtype_name == nullptr) return PVZ_E_INVALID;
    if (a.ndim > 0 && a.shape == nullptr) return PVZ_E_INVALID;
    if (a.nbytes > 0 && a.data == nullptr) return PVZ_E_INVALID;
    if (std::strlen(a.dtype) > PVZSTD_DTYPE_LEN) return PVZ_E_INVALID;
    if (!ParseDtype(a.dtype).valid) return PVZ_E_INVALID;
  }

  ScopedFile src(std::fopen(path, "rb"));
  if (src.get() == nullptr) return PVZ_E_IO;

  std::vector<FrameEntry> frames;
  uint64_t body_end = 0;
  pvz_status st = ReadFooter(src.get(), &frames, &body_end);
  if (st != PVZ_OK) return st;
  const size_t n_arrays = frames.size() / 2;
  const size_t file_meta_idx = n_arrays - 1;

  // 1. The file metadata is always the final array.
  std::string file_meta_json;
  st = DecompressFrame(src.get(), frames, file_meta_idx * 2 + 1, &file_meta_json);
  if (st != PVZ_OK) return st;

  std::vector<std::string> frame_names;
  long long old_level = 0;
  long long old_version = 0;
  std::string compression;
  if (!MemberStringArray(file_meta_json, "frame_names", &frame_names) ||
      !MemberInt(file_meta_json, "compression_level", &old_level) ||
      !MemberInt(file_meta_json, "file_version", &old_version) ||
      !MemberString(file_meta_json, "compression", &compression)) {
    return PVZ_E_FORMAT;
  }
  // The trailing file-metadata array's own name is deliberately absent from
  // frame_names; if the counts disagree the pairing we are about to rebuild
  // would silently shift every name onto the wrong frame.
  if (frame_names.size() != n_arrays - 1) return PVZ_E_FORMAT;

  size_t root_idx = frame_names.size();
  for (size_t i = 0; i < frame_names.size(); ++i) {
    // MultiBlock metadata also ends with the dataset-metadata suffix, and a
    // MultiBlock file has no single root dataset to append to. Refuse it
    // rather than misparse its metadata as a dataset's.
    if (EndsWith(frame_names[i], kMultiblockKey)) return PVZ_E_FORMAT;
    if (root_idx == frame_names.size() && EndsWith(frame_names[i], kDsMetadataKey)) root_idx = i;
  }
  if (root_idx == frame_names.size()) return PVZ_E_FORMAT;
  if (frame_names[root_idx].size() < kUidNChar) return PVZ_E_FORMAT;
  const std::string ds_id = frame_names[root_idx].substr(0, kUidNChar);

  // 2. The root dataset metadata, which is the document that grows.
  std::string ds_json;
  st = DecompressFrame(src.get(), frames, root_idx * 2 + 1, &ds_json);
  if (st != PVZ_OK) return st;

  size_t fdk_open = 0;
  size_t fdk_past = 0;
  if (!MemberObjectSpan(ds_json, "field_data_keys", &fdk_open, &fdk_past)) return PVZ_E_FORMAT;
  std::vector<std::string> existing;
  if (!ObjectKeys(ds_json, fdk_open, &existing)) return PVZ_E_FORMAT;
  for (uint64_t k = 0; k < count; ++k) {
    for (const std::string &have : existing) {
      // Refused, not overwritten: the old block's bytes would stay in the file
      // with nothing pointing at them, and the reader would surface whichever
      // entry it happened to find first.
      if (have == arrays[k].name) return PVZ_E_INVALID;
    }
    for (uint64_t j = 0; j < k; ++j) {
      if (std::strcmp(arrays[j].name, arrays[k].name) == 0) return PVZ_E_INVALID;
    }
  }

  const int use_level = (level == PVZ_LEVEL_FROM_FILE) ? static_cast<int>(old_level) : level;

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
    const pvz_append_array &a = arrays[k];
    const std::string frame_name = ds_id + a.name + kFieldDataSuffix;
    const Dtype d = ParseDtype(a.dtype);

    bool use_shuffle = false;
    if (shuffle != PVZ_SHUFFLE_NEVER && d.itemsize > 1) {
      if (shuffle == PVZ_SHUFFLE_ALWAYS) {
        use_shuffle = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use_shuffle = AutoShuffleBeneficial(static_cast<const uint8_t *>(a.data), a.nbytes,
                                            d.itemsize, use_level);
      }
    }
    const uint8_t filter = use_shuffle ? PVZ_FILTER_SHUFFLE : PVZ_FILTER_NONE;
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
    } catch (const std::bad_alloc &) {
      return PVZ_E_NOMEM;
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

  // 5. Regenerate the file metadata. Field order matches the reference
  //    dataclass; separators are json.dumps(separators=(",", ":")).
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
    plain.push_back(PackArrayMetadata(ds_name, "|u1", {ds_new.size()}, PVZ_FILTER_NONE));
    plain.emplace_back(ds_new.begin(), ds_new.end());
    plain.push_back(PackArrayMetadata(kFileMetadataKey, "|u1", {file_new.size()},
                                      PVZ_FILTER_NONE));
    plain.emplace_back(file_new.begin(), file_new.end());
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }

  // 6. Compress. Single-threaded, matching the reference append -- which uses
  //    threads=0, unlike the writer, which sizes a pool from the payload total.
  for (std::vector<uint8_t> &payload : plain) {
    StagedFrame frame;
    frame.decomp = payload.size();
    st = CompressFrame(payload.data(), payload.size(), use_level, 0, &frame.bytes);
    if (st != PVZ_OK) return st;
    staged.push_back(std::move(frame));
  }

  // 7. Write beside the original and commit by rename, so an interrupted
  //    append cannot damage blocks that were already committed.
  const std::string tmp_path = std::string(path) + ".append.tmp";
  ScopedFile out(std::fopen(tmp_path.c_str(), "wb"));
  if (out.get() == nullptr) return PVZ_E_IO;

  std::vector<uint8_t> chunk;
  std::vector<uint8_t> trailer;
  uint64_t offset = 0;
  bool ok = true;
  for (const StagedFrame &frame : staged) {
    if (frame.copy) {
      uint64_t remaining = frame.src_end - frame.src_start;
      if (std::fseek(src.get(), static_cast<long>(frame.src_start), SEEK_SET) != 0) {
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
    return PVZ_E_IO;
  }

  // Windows will not rename onto an open handle, and the source is still open.
  src.reset(nullptr);
  if (std::rename(tmp_path.c_str(), path) != 0) {
    // Some platforms refuse a rename onto an existing file.
    if (std::remove(path) != 0 || std::rename(tmp_path.c_str(), path) != 0) {
      std::remove(tmp_path.c_str());
      return PVZ_E_IO;
    }
  }
  return PVZ_OK;
}

}  // extern "C"
