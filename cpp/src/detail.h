// Primitives shared by the writing translation units.
//
// These were the writer's private helpers until the append path needed the
// same ones. They are byte-level reproductions of specific reference-writer
// behaviours, not general utilities: the shuffle decision, the worker count,
// and the JSON escaping all decide what ends up on disk, so a second
// implementation of any of them would be a second chance to disagree.

#ifndef PVZSTD_DETAIL_H
#define PVZSTD_DETAIL_H

#include "pvzstd/pvzstd.h"

#include <zstd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace pvzstd::detail {

constexpr int kDefaultLevel = 3;
constexpr uint64_t kShuffleProbeBytes = 1u << 20;  // 1 MiB
constexpr int kMaxManualThreads = 8;
constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr uint64_t kThreadBytesPerWorker = 2ull;  // floor(MiB / 2) workers

inline void StoreU32(std::vector<uint8_t> *out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void StoreU64(std::vector<uint8_t> *out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

inline uint64_t LoadU64(const uint8_t *src) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(src[i]) << (8 * i);
  return v;
}

struct Dtype {
  char byteorder = '|';
  char kind = 'u';
  uint64_t itemsize = 1;
  bool valid = false;
};

inline Dtype ParseDtype(const char *s) {
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
inline void ShuffleBytes(const uint8_t *src, uint8_t *dst, uint64_t nbytes, uint64_t itemsize) {
  const uint64_t n_elem = nbytes / itemsize;
  for (uint64_t plane = 0; plane < itemsize; ++plane) {
    uint8_t *out = dst + plane * n_elem;
    const uint8_t *in = src + plane;
    for (uint64_t i = 0; i < n_elem; ++i) out[i] = in[i * itemsize];
  }
}

inline size_t CompressedSize(const uint8_t *src, uint64_t n, int level) {
  const size_t bound = ZSTD_compressBound(static_cast<size_t>(n));
  std::vector<uint8_t> tmp(bound);
  const size_t got = ZSTD_compress(tmp.data(), tmp.size(), src, static_cast<size_t>(n), level);
  return ZSTD_isError(got) != 0 ? 0 : got;
}

// Mirrors _auto_shuffle_beneficial: trial-compress a centred sample raw and
// shuffled, keep shuffle only when strictly smaller. Reproducing this exactly
// matters -- the decision is recorded on disk, so a cheaper heuristic would
// change the bytes.
inline bool AutoShuffleBeneficial(const uint8_t *data, uint64_t nbytes, uint64_t itemsize,
                                  int level) {
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
inline int ResolveThreads(int requested, uint64_t total_bytes) {
  if (requested != PVZ_THREADS_AUTO) return requested;
  const uint64_t size_mb = total_bytes / kBytesPerMiB;
  const uint64_t n = size_mb / kThreadBytesPerWorker;
  if (n > static_cast<uint64_t>(kMaxManualThreads)) return -1;
  return static_cast<int>(n);
}

// python-zstandard maps a negative worker count to the logical CPU count.
inline int EffectiveWorkers(int threads) {
  if (threads >= 0) return threads;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

inline pvz_status CompressFrame(const uint8_t *src, uint64_t n, int level, int workers,
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
inline void AppendJsonString(std::string *out, const std::string &s) {
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

// The array-metadata frame payload: name, shape, dtype, and -- only when a
// filter is in use -- the filter byte. Omitting the byte when it would be zero
// is what the reference writer does, and readers key on the payload length.
inline std::vector<uint8_t> PackArrayMetadata(const std::string &name, const std::string &dtype,
                                              const std::vector<uint64_t> &shape,
                                              uint8_t filter_id) {
  std::vector<uint8_t> meta;
  StoreU32(&meta, static_cast<uint32_t>(name.size()));
  meta.insert(meta.end(), name.begin(), name.end());
  StoreU32(&meta, static_cast<uint32_t>(shape.size()));
  for (const uint64_t s : shape) StoreU64(&meta, s);
  std::string dt = dtype;
  dt.resize(16, ' ');
  meta.insert(meta.end(), dt.begin(), dt.end());
  if (filter_id != 0) meta.push_back(filter_id);
  return meta;
}

}  // namespace pvzstd::detail

#endif  // PVZSTD_DETAIL_H
