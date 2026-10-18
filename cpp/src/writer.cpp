// Writer half of the .pv container. See doc/format/container-v2.md.
//
// Byte-for-byte agreement with the reference writer constrains more than the
// layout: the file version is the highest optional feature used (3 MultiBlock
// field data, an ExplicitStructuredGrid, or a StructuredGrid or RectilinearGrid
// extent not starting at 0, else 2 fixed-width cells, else 1 if anything is
// shuffled, else 0; not cumulative), and every unsplit frame needs the same
// level AND worker count, since zstd's multi-threaded mode emits different
// bytes.
//
// A payload larger than kSubFrameBytes is written as consecutive zstd frames of
// that size, each compressed on one thread, so a reader can decompress the
// pieces in parallel: zstd cannot split one frame across cores on the way back.
// One-shot decoders consume the frames back to back, so the split needs no
// file version of its own.

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "detail.h"
#include "json_read.h"
#include "pvzstd/pvzstd.h"
#include "thumbnail.h"

using namespace pvzstd::detail;  // NOLINT(google-build-using-namespace)

namespace {

// Optimization: payloads past this are split into sub-frames of this size. Four MiB keeps a
// large array on every core of a reader while costing zstd's window nothing.
constexpr uint64_t kSubFrameBytes = 4ull << 20;

struct PendingArray {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> owned;  // holds the bytes unless `borrowed` is set
  const uint8_t *borrowed = nullptr;
  uint64_t borrowed_size = 0;
  bool is_metadata_json = false;

  // Read the payload through these, never through a member: an owning entry
  // carries its length in the vector and a borrowing one in `borrowed_size`,
  // and reading the wrong one silently yields an empty frame.
  const uint8_t *bytes() const { return borrowed != nullptr ? borrowed : owned.data(); }
  uint64_t size() const { return borrowed != nullptr ? borrowed_size : owned.size(); }
};

// Whether `json` is one brace-balanced object with nothing after it. Members are
// not validated; a reader's JSON parser still rejects a malformed one.
bool IsJsonObject(const std::string &json) {
  size_t i = 0;
  if (!pvzstd::json::SkipWs(json, &i) || json[i] != '{') return false;
  if (!pvzstd::json::SkipValue(json, &i)) return false;
  return !pvzstd::json::SkipWs(json, &i);
}

// Whether the member `key` is an array with a digit 1-9 before its first ']',
// which for the flat integer arrays json.dumps writes means a non-zero entry.
bool MemberArrayHasNonZero(const std::string &json, const char *key) {
  size_t i = pvzstd::json::FindMember(json, key);
  if (i == std::string::npos || i >= json.size() || json[i] != '[') return false;
  for (++i; i < json.size() && json[i] != ']'; ++i) {
    if (json[i] >= '1' && json[i] <= '9') return true;
  }
  return false;
}

// Sets *uses to whether a staged metadata document records MultiBlock field
// arrays, an ExplicitStructuredGrid, or a StructuredGrid or RectilinearGrid whose
// extent does not start at 0, none of which readers with a ceiling of 2 decode
// correctly. PVZSTD_E_INVALID when a metadata document fails IsJsonObject.
pvzstd_status ScanVersion3Features(const std::vector<PendingArray> &arrays, bool *uses) {
  *uses = false;
  for (const PendingArray &a : arrays) {
    if (!EndsWith(a.name, kDsMetadataSuffix)) continue;
    const std::string json(reinterpret_cast<const char *>(a.bytes()), a.size());
    if (!IsJsonObject(json)) return PVZSTD_E_INVALID;
    if (EndsWith(a.name, kMultiblockSuffix)) {
      size_t open = 0;
      size_t past = 0;
      if (pvzstd::json::MemberObjectSpan(json, "field_data_keys", &open, &past)) *uses = true;
    } else {
      std::string ds_type;
      if (!pvzstd::json::MemberString(json, "ds_type", &ds_type)) continue;
      if (ds_type == "ExplicitStructuredGrid") *uses = true;
      if ((ds_type == "StructuredGrid" || ds_type == "RectilinearGrid") &&
          MemberArrayHasNonZero(json, "offset")) {
        *uses = true;
      }
    }
  }
  return PVZSTD_OK;
}

}  // namespace

struct pvzstd_writer {
  int level = kDefaultLevel;
  int threads = PVZSTD_THREADS_AUTO;
  pvzstd_shuffle_mode shuffle = PVZSTD_SHUFFLE_NEVER;
  bool fixed_width_cells = false;
  std::vector<PendingArray> arrays;
  std::vector<uint8_t> thumbnail_png;
  uint32_t thumbnail_width = 0;
  uint32_t thumbnail_height = 0;
};

namespace {

// Shared by the copying and borrowing entry points, which differ only in `copy`.
pvzstd_status AddArray(pvzstd_writer *writer, const char *name, const char *dtype,
                       const uint64_t *shape, uint32_t ndim, const void *data, uint64_t nbytes,
                       bool copy) {
  if (writer == nullptr || name == nullptr || dtype == nullptr) return PVZSTD_E_INVALID;
  if (nbytes > 0 && data == nullptr) return PVZSTD_E_INVALID;
  // A borrowed entry keeps `nbytes` as a 64-bit length and hands it to
  // ShuffleBytes at write time, while the shuffle destination is sized from the
  // same value narrowed into a vector -- so a length past size_t writes the
  // whole array into a buffer holding its low bits. No caller can own that many
  // bytes on a target where the check is live, so it is a bad argument.
  if (ExceedsSizeT(nbytes)) return PVZSTD_E_INVALID;
  if (ndim > 0 && shape == nullptr) return PVZSTD_E_INVALID;
  if (std::strlen(dtype) > PVZSTD_DTYPE_LEN) return PVZSTD_E_INVALID;
  if (!ParseDtype(dtype).valid) return PVZSTD_E_INVALID;
  if (std::strcmp(name, kThumbnailName) == 0) return PVZSTD_E_INVALID;  // reserved

  const uint8_t *src = static_cast<const uint8_t *>(data);
  PendingArray entry;
  entry.name = name;
  entry.dtype = dtype;
  entry.shape.assign(shape, shape + ndim);
  try {
    if (copy) {
      entry.owned.assign(src, src + nbytes);
    } else {
      entry.borrowed = src;
      entry.borrowed_size = nbytes;
    }
    writer->arrays.push_back(std::move(entry));
  } catch (const std::exception &) {
    // Wider than std::bad_alloc on purpose: an oversized allocation throws
    // std::length_error, which is not a bad_alloc, so the narrower handler let
    // it out of the library.
    return PVZSTD_E_NOMEM;
  }
  return PVZSTD_OK;
}

// Compress `n` bytes as sub-frames of kSubFrameBytes, spread over `workers`
// threads with one thread per sub-frame, and write them in order. One trailer
// entry still covers the lot: a reader sees one payload that happens to be
// several frames, which ZSTD_decompress accepts as it is.
pvzstd_status WriteSubFrames(std::FILE *fp, const uint8_t *src, uint64_t n, int level, int workers,
                             uint64_t *offset) {
  const uint64_t n_sub = (n + kSubFrameBytes - 1) / kSubFrameBytes;
  if (ExceedsSizeT(n_sub)) return PVZSTD_E_INVALID;
  std::vector<std::vector<uint8_t>> frames;
  std::vector<pvzstd_status> status;
  try {
    frames.resize(static_cast<size_t>(n_sub));
    status.assign(static_cast<size_t>(n_sub), PVZSTD_OK);
  } catch (const std::exception &) {
    return PVZSTD_E_NOMEM;
  }
  int pool = workers > 1 ? workers : 1;
  if (n_sub < static_cast<uint64_t>(pool)) pool = static_cast<int>(n_sub);
  ParallelStride(pool, n_sub, [&](uint64_t k) {
    const uint64_t start = k * kSubFrameBytes;
    const uint64_t len = n - start < kSubFrameBytes ? n - start : kSubFrameBytes;
    status[static_cast<size_t>(k)] =
        CompressFrame(src + start, len, level, 0, &frames[static_cast<size_t>(k)]);
  });
  for (uint64_t k = 0; k < n_sub; ++k) {
    if (status[static_cast<size_t>(k)] != PVZSTD_OK) return status[static_cast<size_t>(k)];
    const std::vector<uint8_t> &frame = frames[static_cast<size_t>(k)];
    if (std::fwrite(frame.data(), 1, frame.size(), fp) != frame.size()) return PVZSTD_E_IO;
    *offset += frame.size();
  }
  return PVZSTD_OK;
}

}  // namespace

extern "C" {

pvzstd_status pvzstd_writer_create(pvzstd_writer **out) try {
  if (out == nullptr) return PVZSTD_E_INVALID;
  *out = new (std::nothrow) pvzstd_writer();
  return *out == nullptr ? PVZSTD_E_NOMEM : PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

void pvzstd_writer_free(pvzstd_writer *writer) try { delete writer; } catch (...) {
}

pvzstd_status pvzstd_writer_set_level(pvzstd_writer *writer, int level) try {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->level = level;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_set_threads(pvzstd_writer *writer, int n_threads) try {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->threads = n_threads;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_set_shuffle(pvzstd_writer *writer, pvzstd_shuffle_mode mode) try {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->shuffle = mode;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_set_fixed_width_cells(pvzstd_writer *writer, int enabled) try {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->fixed_width_cells = enabled != 0;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_set_thumbnail(pvzstd_writer *writer, const void *png,
                                          uint64_t nbytes) try {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  if (png == nullptr && nbytes == 0) {
    writer->thumbnail_png.clear();
    return PVZSTD_OK;
  }
  if (nbytes > PVZSTD_THUMBNAIL_MAX_BYTES) return PVZSTD_E_RANGE;
  const uint8_t *src = static_cast<const uint8_t *>(png);
  uint32_t width = 0;
  uint32_t height = 0;
  if (!PngSize(src, nbytes, &width, &height)) return PVZSTD_E_INVALID;
  try {
    writer->thumbnail_png.assign(src, src + nbytes);
  } catch (const std::exception &) {
    return PVZSTD_E_NOMEM;
  }
  writer->thumbnail_width = width;
  writer->thumbnail_height = height;
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_add_array(pvzstd_writer *writer, const char *name, const char *dtype,
                                      const uint64_t *shape, uint32_t ndim, const void *data,
                                      uint64_t nbytes) try {
  return AddArray(writer, name, dtype, shape, ndim, data, nbytes, /*copy=*/true);
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_add_array_borrowed(pvzstd_writer *writer, const char *name,
                                               const char *dtype, const uint64_t *shape,
                                               uint32_t ndim, const void *data,
                                               uint64_t nbytes) try {
  return AddArray(writer, name, dtype, shape, ndim, data, nbytes, /*copy=*/false);
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_set_ds_metadata(pvzstd_writer *writer, const char *uid,
                                            const char *json) try {
  if (writer == nullptr || uid == nullptr || json == nullptr) return PVZSTD_E_INVALID;
  const uint64_t n = std::strlen(json);
  PendingArray entry;
  entry.name = std::string(uid) + kDsMetadataSuffix;
  entry.dtype = "|u1";
  entry.shape.push_back(n);
  entry.is_metadata_json = true;
  try {
    entry.owned.assign(json, json + n);
    writer->arrays.push_back(std::move(entry));
  } catch (const std::exception &) {
    // See AddArray. No guard in front: `n` is a strlen of a string the caller
    // already holds, so it cannot name more bytes than exist.
    return PVZSTD_E_NOMEM;
  }
  return PVZSTD_OK;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

pvzstd_status pvzstd_writer_write(pvzstd_writer *writer, const char *path) try {
  if (writer == nullptr || path == nullptr) return PVZSTD_E_INVALID;
  if (writer->arrays.empty()) return PVZSTD_E_INVALID;

  // Before any compression, so a malformed metadata document fails fast.
  bool uses_version_3 = false;
  const pvzstd_status scanned = ScanVersion3Features(writer->arrays, &uses_version_3);
  if (scanned != PVZSTD_OK) return scanned;

  // 1. Resolve each array's filter, exactly as the reference writer does.
  std::vector<uint8_t> filters(writer->arrays.size(), PVZSTD_FILTER_NONE);
  bool any_filtered = false;
  for (size_t i = 0; i < writer->arrays.size(); ++i) {
    const PendingArray &a = writer->arrays[i];
    const Dtype d = ParseDtype(a.dtype.c_str());
    bool use = false;
    // No emptiness check: the reference decides from dtype, so an empty multibyte
    // array under shuffle=True is still recorded as filtered.
    if (writer->shuffle != PVZSTD_SHUFFLE_NEVER && d.itemsize > 1) {
      if (writer->shuffle == PVZSTD_SHUFFLE_ALWAYS) {
        use = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use = AutoShuffleBeneficial(a.bytes(), a.size(), d.itemsize, writer->level);
      }
    }
    if (use) {
      filters[i] = PVZSTD_FILTER_SHUFFLE;
      any_filtered = true;
    }
  }

  // 2. File version is the highest optional feature used, not a generation.
  int file_version = 0;
  if (uses_version_3) {
    file_version = 3;
  } else if (writer->fixed_width_cells) {
    file_version = 2;
  } else if (any_filtered) {
    file_version = 1;
  }

  // 3. Trailing file-metadata frame, in the reference dataclass's field order.
  const bool has_thumbnail = !writer->thumbnail_png.empty();
  std::string meta = "{\"frame_names\":[";
  if (has_thumbnail) {
    AppendJsonString(&meta, kThumbnailName);
    meta.push_back(',');  // arrays is never empty here
  }
  for (size_t i = 0; i < writer->arrays.size(); ++i) {
    if (i != 0) meta.push_back(',');
    AppendJsonString(&meta, writer->arrays[i].name);
  }
  meta += "],\"compression_level\":" + std::to_string(writer->level);
  meta += ",\"compression\":\"zstandard\"";
  meta += ",\"file_version\":" + std::to_string(file_version) + "}";

  // Total taken before the push: the reference sizes its pool from the arrays
  // only. The frame is small and the threshold is in MiB, but the two rules
  // differ exactly at a boundary.
  uint64_t total_bytes = 0;
  for (const PendingArray &a : writer->arrays) total_bytes += a.size();

  PendingArray meta_entry;
  meta_entry.name = kFileMetadataSuffix;
  meta_entry.dtype = "|u1";
  meta_entry.shape.push_back(meta.size());
  meta_entry.is_metadata_json = true;
  try {
    meta_entry.owned.assign(meta.begin(), meta.end());
  } catch (const std::exception &) {
    // See AddArray. No guard in front: `meta` is a std::string built above.
    return PVZSTD_E_NOMEM;
  }
  // Local list, not the writer's own, so writing to a second path emits one
  // metadata frame and not two.
  std::vector<const PendingArray *> emit;
  emit.reserve(writer->arrays.size() + 1);
  for (const PendingArray &a : writer->arrays) emit.push_back(&a);
  emit.push_back(&meta_entry);
  filters.push_back(PVZSTD_FILTER_NONE);  // the JSON blob is never shuffled

  // 4. Worker count, from the total computed above.
  const int workers = EffectiveWorkers(ResolveThreads(writer->threads, total_bytes));

  // Built before the file is opened, so a failed allocation leaves nothing open.
  std::vector<uint8_t> thumbnail_prefix;
  if (has_thumbnail) {
    thumbnail_prefix = ThumbnailPrefix(writer->thumbnail_png.size(), writer->thumbnail_width,
                                       writer->thumbnail_height);
  }

  // 5. Emit (header, payload) pairs, recording END offsets.
  std::FILE *fp = std::fopen(path, "wb");
  if (fp == nullptr) return PVZSTD_E_IO;

  std::vector<uint64_t> ends;
  std::vector<uint64_t> sizes;
  uint64_t offset = 0;
  pvzstd_status st = PVZSTD_OK;
  std::vector<uint8_t> header;
  std::vector<uint8_t> payload;
  std::vector<uint8_t> frame;

  // The thumbnail pair goes first, with its PNG written straight from the stored copy.
  if (has_thumbnail) {
    const uint64_t png_nbytes = writer->thumbnail_png.size();
    if (std::fwrite(thumbnail_prefix.data(), 1, thumbnail_prefix.size(), fp) !=
            thumbnail_prefix.size() ||
        std::fwrite(writer->thumbnail_png.data(), 1, png_nbytes, fp) != png_nbytes) {
      st = PVZSTD_E_IO;
    }
    offset = thumbnail_prefix.size() + png_nbytes;
    ends.push_back(kThumbnailHeaderFrameBytes);
    sizes.push_back(kThumbnailArrayHeaderBytes);
    ends.push_back(offset);
    sizes.push_back(0);  // a skippable frame decodes to nothing
  }

  for (size_t i = 0; i < emit.size() && st == PVZSTD_OK; ++i) {
    const PendingArray &a = *emit[i];
    const Dtype d = ParseDtype(a.dtype.c_str());

    header.clear();
    StoreU32(&header, static_cast<uint32_t>(a.name.size()));
    header.insert(header.end(), a.name.begin(), a.name.end());
    StoreU32(&header, static_cast<uint32_t>(a.shape.size()));
    for (const uint64_t dim : a.shape) StoreU64(&header, dim);
    for (size_t k = 0; k < PVZSTD_DTYPE_LEN; ++k) {
      header.push_back(k < a.dtype.size() ? static_cast<uint8_t>(a.dtype[k]) : ' ');
    }
    // Only when a filter is in use, keeping unfiltered frames byte-identical to
    // the legacy layout.
    if (filters[i] != PVZSTD_FILTER_NONE) header.push_back(filters[i]);

    const uint8_t *payload_ptr = a.bytes();
    uint64_t payload_len = a.size();
    if (filters[i] == PVZSTD_FILTER_SHUFFLE) {
      payload.assign(a.size(), 0);
      ShuffleBytes(a.bytes(), payload.data(), a.size(), d.itemsize);
      payload_ptr = payload.data();
      payload_len = payload.size();
    }

    const uint8_t *pieces[2] = {header.data(), payload_ptr};
    const uint64_t lengths[2] = {header.size(), payload_len};
    for (int part = 0; part < 2 && st == PVZSTD_OK; ++part) {
      if (part == 1 && lengths[part] > kSubFrameBytes) {
        st = WriteSubFrames(fp, pieces[part], lengths[part], writer->level, workers, &offset);
      } else {
        st = CompressFrame(pieces[part], lengths[part], writer->level, workers, &frame);
        if (st != PVZSTD_OK) break;
        if (!frame.empty() && std::fwrite(frame.data(), 1, frame.size(), fp) != frame.size()) {
          st = PVZSTD_E_IO;
        }
        offset += frame.size();
      }
      if (st != PVZSTD_OK) break;
      ends.push_back(offset);
      sizes.push_back(lengths[part]);
    }
  }

  // 6. Trailer: (end, decompressed_size) per frame, then the frame count.
  if (st == PVZSTD_OK) {
    std::vector<uint8_t> trailer;
    for (size_t i = 0; i < ends.size(); ++i) {
      StoreU64(&trailer, ends[i]);
      StoreU64(&trailer, sizes[i]);
    }
    StoreU64(&trailer, static_cast<uint64_t>(ends.size()));
    if (std::fwrite(trailer.data(), 1, trailer.size(), fp) != trailer.size()) st = PVZSTD_E_IO;

    // 7. Staleness CRC, patched into the thumbnail descriptor.
    if (st == PVZSTD_OK && has_thumbnail) {
      const uint32_t crc = IndexCrc(trailer.data(), ends.size());
      uint8_t crc_bytes[4];
      for (int k = 0; k < 4; ++k) crc_bytes[k] = static_cast<uint8_t>((crc >> (8 * k)) & 0xFF);
      if (std::fseek(fp, static_cast<long>(kThumbnailCrcAt), SEEK_SET) != 0 ||
          std::fwrite(crc_bytes, 1, sizeof(crc_bytes), fp) != sizeof(crc_bytes)) {
        st = PVZSTD_E_IO;
      }
    }
  }

  if (std::fclose(fp) != 0 && st == PVZSTD_OK) st = PVZSTD_E_IO;
  return st;
} catch (...) {
  return PVZSTD_E_NOMEM;
}

}  // extern "C"
