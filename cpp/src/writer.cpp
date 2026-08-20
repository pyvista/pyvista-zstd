// Writer half of the .pv container. See doc/format/container-v2.md.
//
// The goal is byte-for-byte agreement with the reference Python writer, which
// constrains more than the layout:
//
//   * frames are emitted as (header, payload) pairs, and the trailer records
//     each frame's END offset;
//   * the header's filter byte is written only when a filter is in use;
//   * the file version is the highest optional encoding used -- 2 for
//     fixed-width cells, else 1 if anything is shuffled, else 0 -- and the
//     tiers are not cumulative;
//   * every frame is compressed with the same level AND the same worker
//     count, because zstd's multi-threaded mode produces different bytes than
//     its single-threaded mode. Reproducing the layout but not the worker
//     count yields a valid file that is not the same file.

#include "pvzstd/pvzstd.h"

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kDefaultLevel = 3;
constexpr uint64_t kShuffleProbeBytes = 1u << 20;  // 1 MiB
constexpr int kMaxManualThreads = 8;
constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr uint64_t kThreadBytesPerWorker = 2ull;  // floor(MiB / 2) workers

void StoreU32(std::vector<uint8_t> *out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void StoreU64(std::vector<uint8_t> *out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

struct Dtype {
  char byteorder = '|';
  char kind = 'u';
  uint64_t itemsize = 1;
  bool valid = false;
};

Dtype ParseDtype(const char *s) {
  Dtype d;
  if (s == nullptr) return d;
  const size_t n = std::strlen(s);
  if (n < 3) return d;
  d.byteorder = s[0];
  d.kind = s[1];
  uint64_t width = 0;
  for (size_t i = 2; i < n; ++i) {
    if (s[i] < '0' || s[i] > '9') return d;
    width = width * 10 + static_cast<uint64_t>(s[i] - '0');
  }
  if (width == 0) return d;
  d.itemsize = width;
  d.valid = true;
  return d;
}

// Split into byte planes: plane p holds byte p of every element.
void ShuffleBytes(const uint8_t *src, uint8_t *dst, uint64_t nbytes, uint64_t itemsize) {
  const uint64_t n_elem = nbytes / itemsize;
  for (uint64_t plane = 0; plane < itemsize; ++plane) {
    uint8_t *out = dst + plane * n_elem;
    const uint8_t *in = src + plane;
    for (uint64_t i = 0; i < n_elem; ++i) out[i] = in[i * itemsize];
  }
}

size_t CompressedSize(const uint8_t *src, uint64_t n, int level) {
  const size_t bound = ZSTD_compressBound(static_cast<size_t>(n));
  std::vector<uint8_t> tmp(bound);
  const size_t got = ZSTD_compress(tmp.data(), tmp.size(), src, static_cast<size_t>(n), level);
  return ZSTD_isError(got) != 0 ? 0 : got;
}

// Mirrors _auto_shuffle_beneficial: trial-compress a centred sample raw and
// shuffled, keep shuffle only when strictly smaller. Reproducing this exactly
// matters -- the decision is recorded on disk, so a cheaper heuristic would
// change the bytes.
bool AutoShuffleBeneficial(const uint8_t *data, uint64_t nbytes, uint64_t itemsize, int level) {
  const uint64_t n_elem = nbytes / itemsize;
  if (n_elem == 0) return false;
  uint64_t budget = kShuffleProbeBytes / itemsize;
  if (budget == 0) budget = 1;
  const uint64_t n_sample = n_elem < budget ? n_elem : budget;
  const uint64_t start = (n_elem - n_sample) / 2;

  const uint8_t *sample = data + start * itemsize;
  const uint64_t sample_bytes = n_sample * itemsize;
  std::vector<uint8_t> shuffled(static_cast<size_t>(sample_bytes));
  ShuffleBytes(sample, shuffled.data(), sample_bytes, itemsize);

  const size_t raw_size = CompressedSize(sample, sample_bytes, level);
  const size_t shuf_size = CompressedSize(shuffled.data(), sample_bytes, level);
  if (raw_size == 0 || shuf_size == 0) return false;
  return shuf_size < raw_size;
}

// Mirrors _set_n_threads for the PVZ_THREADS_AUTO case.
int ResolveThreads(int requested, uint64_t total_bytes) {
  if (requested != PVZ_THREADS_AUTO) return requested;
  const uint64_t size_mb = total_bytes / kBytesPerMiB;
  const uint64_t n = size_mb / kThreadBytesPerWorker;
  if (n > static_cast<uint64_t>(kMaxManualThreads)) return -1;
  return static_cast<int>(n);
}

// python-zstandard maps a negative worker count to the logical CPU count.
int EffectiveWorkers(int threads) {
  if (threads >= 0) return threads;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

pvz_status CompressFrame(const uint8_t *src, uint64_t n, int level, int workers,
                         std::vector<uint8_t> *out) {
  const size_t bound = ZSTD_compressBound(static_cast<size_t>(n));
  try {
    out->resize(bound);
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }

  ZSTD_CCtx *cctx = ZSTD_createCCtx();
  if (cctx == nullptr) return PVZ_E_NOMEM;
  size_t got = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
  if (ZSTD_isError(got) == 0 && workers > 0) {
    got = ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers);
  }
  if (ZSTD_isError(got) != 0) {
    ZSTD_freeCCtx(cctx);
    return PVZ_E_ZSTD;
  }
  got = ZSTD_compress2(cctx, out->data(), out->size(), src, static_cast<size_t>(n));
  ZSTD_freeCCtx(cctx);
  if (ZSTD_isError(got) != 0) return PVZ_E_ZSTD;
  out->resize(got);
  return PVZ_OK;
}

// JSON string escaping sufficient for the identifiers this format carries,
// matching json.dumps' default (ensure_ascii, minimal separators).
void AppendJsonString(std::string *out, const std::string &s) {
  out->push_back('"');
  for (const char c : s) {
    switch (c) {
      case '"': *out += "\\\""; break;
      case '\\': *out += "\\\\"; break;
      case '\n': *out += "\\n"; break;
      case '\r': *out += "\\r"; break;
      case '\t': *out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c) & 0xFFu);
          *out += buf;
        } else {
          out->push_back(c);
        }
    }
  }
  out->push_back('"');
}

struct PendingArray {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;
  bool is_metadata_json = false;
};

}  // namespace

struct pvz_writer {
  int level = kDefaultLevel;
  int threads = PVZ_THREADS_AUTO;
  pvz_shuffle_mode shuffle = PVZ_SHUFFLE_NEVER;
  bool fixed_width_cells = false;
  std::vector<PendingArray> arrays;
};

extern "C" {

pvz_status pvz_writer_create(pvz_writer **out) {
  if (out == nullptr) return PVZ_E_INVALID;
  *out = new (std::nothrow) pvz_writer();
  return *out == nullptr ? PVZ_E_NOMEM : PVZ_OK;
}

void pvz_writer_free(pvz_writer *writer) { delete writer; }

pvz_status pvz_writer_set_level(pvz_writer *writer, int level) {
  if (writer == nullptr) return PVZ_E_INVALID;
  writer->level = level;
  return PVZ_OK;
}

pvz_status pvz_writer_set_threads(pvz_writer *writer, int n_threads) {
  if (writer == nullptr) return PVZ_E_INVALID;
  writer->threads = n_threads;
  return PVZ_OK;
}

pvz_status pvz_writer_set_shuffle(pvz_writer *writer, pvz_shuffle_mode mode) {
  if (writer == nullptr) return PVZ_E_INVALID;
  writer->shuffle = mode;
  return PVZ_OK;
}

pvz_status pvz_writer_set_fixed_width_cells(pvz_writer *writer, int enabled) {
  if (writer == nullptr) return PVZ_E_INVALID;
  writer->fixed_width_cells = enabled != 0;
  return PVZ_OK;
}

pvz_status pvz_writer_add_array(pvz_writer *writer, const char *name, const char *dtype,
                                const uint64_t *shape, uint32_t ndim, const void *data,
                                uint64_t nbytes) {
  if (writer == nullptr || name == nullptr || dtype == nullptr) return PVZ_E_INVALID;
  if (nbytes > 0 && data == nullptr) return PVZ_E_INVALID;
  if (ndim > 0 && shape == nullptr) return PVZ_E_INVALID;
  if (std::strlen(dtype) > PVZSTD_DTYPE_LEN) return PVZ_E_INVALID;
  if (!ParseDtype(dtype).valid) return PVZ_E_INVALID;

  PendingArray entry;
  entry.name = name;
  entry.dtype = dtype;
  entry.shape.assign(shape, shape + ndim);
  try {
    entry.data.assign(static_cast<const uint8_t *>(data),
                      static_cast<const uint8_t *>(data) + nbytes);
    writer->arrays.push_back(std::move(entry));
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  return PVZ_OK;
}

pvz_status pvz_writer_set_ds_metadata(pvz_writer *writer, const char *uid, const char *json) {
  if (writer == nullptr || uid == nullptr || json == nullptr) return PVZ_E_INVALID;
  const uint64_t n = std::strlen(json);
  PendingArray entry;
  entry.name = std::string(uid) + "__ds_metadata";
  entry.dtype = "|u1";
  entry.shape.push_back(n);
  entry.is_metadata_json = true;
  try {
    entry.data.assign(json, json + n);
    writer->arrays.push_back(std::move(entry));
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  return PVZ_OK;
}

pvz_status pvz_writer_write(pvz_writer *writer, const char *path) {
  if (writer == nullptr || path == nullptr) return PVZ_E_INVALID;
  if (writer->arrays.empty()) return PVZ_E_INVALID;

  // 1. Resolve each array's filter, exactly as the reference writer does.
  std::vector<uint8_t> filters(writer->arrays.size(), PVZ_FILTER_NONE);
  bool any_filtered = false;
  for (size_t i = 0; i < writer->arrays.size(); ++i) {
    const PendingArray &a = writer->arrays[i];
    const Dtype d = ParseDtype(a.dtype.c_str());
    bool use = false;
    // Note the absent emptiness check: the reference decides from dtype, so an
    // empty multibyte array under shuffle=True is still *recorded* as
    // filtered. Shuffling nothing is a no-op, but the header byte is not.
    if (writer->shuffle != PVZ_SHUFFLE_NEVER && d.itemsize > 1) {
      if (writer->shuffle == PVZ_SHUFFLE_ALWAYS) {
        use = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use = AutoShuffleBeneficial(a.data.data(), a.data.size(), d.itemsize, writer->level);
      }
    }
    if (use) {
      filters[i] = PVZ_FILTER_SHUFFLE;
      any_filtered = true;
    }
  }

  // 2. File version is the highest optional encoding used, not a generation.
  int file_version = 0;
  if (writer->fixed_width_cells) {
    file_version = 2;
  } else if (any_filtered) {
    file_version = 1;
  }

  // 3. Build the trailing file-metadata frame. Field order matches the
  //    reference dataclass, and separators are json.dumps(separators=(",", ":")).
  std::string meta = "{\"frame_names\":[";
  for (size_t i = 0; i < writer->arrays.size(); ++i) {
    if (i != 0) meta.push_back(',');
    AppendJsonString(&meta, writer->arrays[i].name);
  }
  meta += "],\"compression_level\":" + std::to_string(writer->level);
  meta += ",\"compression\":\"zstandard\"";
  meta += ",\"file_version\":" + std::to_string(file_version) + "}";

  // The reference writer sizes its worker pool from the arrays only -- it
  // computes the total before appending this metadata frame -- so the total is
  // taken here, before the push below. The frame is a few hundred bytes and
  // the threshold is in MiB, but the two rules differ exactly at a boundary,
  // which is where a "close enough" version would diverge.
  uint64_t total_bytes = 0;
  for (const PendingArray &a : writer->arrays) total_bytes += a.data.size();

  PendingArray meta_entry;
  meta_entry.name = "__pyvista_zstd_metadata";
  meta_entry.dtype = "|u1";
  meta_entry.shape.push_back(meta.size());
  meta_entry.is_metadata_json = true;
  try {
    meta_entry.data.assign(meta.begin(), meta.end());
  } catch (const std::bad_alloc &) {
    return PVZ_E_NOMEM;
  }
  writer->arrays.push_back(std::move(meta_entry));
  filters.push_back(PVZ_FILTER_NONE);  // the JSON blob is never shuffled

  // 4. Worker count, from the total computed above.
  const int workers = EffectiveWorkers(ResolveThreads(writer->threads, total_bytes));

  // 5. Emit (header, payload) pairs, recording END offsets.
  std::FILE *fp = std::fopen(path, "wb");
  if (fp == nullptr) return PVZ_E_IO;

  std::vector<uint64_t> ends;
  std::vector<uint64_t> sizes;
  uint64_t offset = 0;
  pvz_status st = PVZ_OK;
  std::vector<uint8_t> header;
  std::vector<uint8_t> payload;
  std::vector<uint8_t> frame;

  for (size_t i = 0; i < writer->arrays.size() && st == PVZ_OK; ++i) {
    const PendingArray &a = writer->arrays[i];
    const Dtype d = ParseDtype(a.dtype.c_str());

    header.clear();
    StoreU32(&header, static_cast<uint32_t>(a.name.size()));
    header.insert(header.end(), a.name.begin(), a.name.end());
    StoreU32(&header, static_cast<uint32_t>(a.shape.size()));
    for (const uint64_t dim : a.shape) StoreU64(&header, dim);
    for (size_t k = 0; k < PVZSTD_DTYPE_LEN; ++k) {
      header.push_back(k < a.dtype.size() ? static_cast<uint8_t>(a.dtype[k]) : ' ');
    }
    // Written only when a filter is in use: absence means PVZ_FILTER_NONE and
    // keeps unfiltered frames byte-identical to the legacy layout.
    if (filters[i] != PVZ_FILTER_NONE) header.push_back(filters[i]);

    const uint8_t *payload_ptr = a.data.data();
    uint64_t payload_len = a.data.size();
    if (filters[i] == PVZ_FILTER_SHUFFLE) {
      payload.assign(a.data.size(), 0);
      ShuffleBytes(a.data.data(), payload.data(), a.data.size(), d.itemsize);
      payload_ptr = payload.data();
      payload_len = payload.size();
    }

    const uint8_t *pieces[2] = {header.data(), payload_ptr};
    const uint64_t lengths[2] = {header.size(), payload_len};
    for (int part = 0; part < 2 && st == PVZ_OK; ++part) {
      st = CompressFrame(pieces[part], lengths[part], writer->level, workers, &frame);
      if (st != PVZ_OK) break;
      if (!frame.empty() && std::fwrite(frame.data(), 1, frame.size(), fp) != frame.size()) {
        st = PVZ_E_IO;
        break;
      }
      offset += frame.size();
      ends.push_back(offset);
      sizes.push_back(lengths[part]);
    }
  }

  // 6. Trailer: (end, decompressed_size) per frame, then the frame count.
  if (st == PVZ_OK) {
    std::vector<uint8_t> trailer;
    for (size_t i = 0; i < ends.size(); ++i) {
      StoreU64(&trailer, ends[i]);
      StoreU64(&trailer, sizes[i]);
    }
    StoreU64(&trailer, static_cast<uint64_t>(ends.size()));
    if (std::fwrite(trailer.data(), 1, trailer.size(), fp) != trailer.size()) st = PVZ_E_IO;
  }

  if (std::fclose(fp) != 0 && st == PVZ_OK) st = PVZ_E_IO;
  return st;
}

}  // extern "C"
