// The optional raw-block thumbnail pair. See doc/format/container-v2.md, section 11.

#ifndef PVZSTD_THUMBNAIL_H
#define PVZSTD_THUMBNAIL_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "detail.h"

namespace pvzstd::detail {

constexpr char kThumbnailName[] = "__thumbnail";
constexpr char kThumbnailMagic[8] = {'P', 'V', 'Z', 'T', 'H', 'U', 'M', 'B'};
constexpr uint32_t kThumbnailVersion = 1;
constexpr uint64_t kThumbnailDescriptorBytes = 32;
// Byte offset of the staleness CRC inside the descriptor.
constexpr uint64_t kThumbnailCrcOffset = 28;
// Frame magic, descriptor byte, 4-byte content size, 3-byte block header.
constexpr uint64_t kRawFramePrefixBytes = 12;
// A raw block holds at most this many bytes, and the pair is one block per frame.
constexpr uint64_t kRawBlockMaxBytes = 128u * 1024u;
constexpr uint64_t kThumbnailMaxPngBytes = kRawBlockMaxBytes - kThumbnailDescriptorBytes;
constexpr uint8_t kPngSignature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

inline uint32_t LoadU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint32_t LoadU32BE(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

// CRC-32 with the zlib polynomial, so Python's zlib.crc32 gives the same value.
inline uint32_t Crc32(const uint8_t *data, uint64_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (uint64_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

// Width and height from a PNG's IHDR chunk, or false when the bytes are not a PNG.
inline bool PngSize(const uint8_t *png, uint64_t n, uint32_t *width, uint32_t *height) {
  if (png == nullptr || n < 33) return false;
  if (std::memcmp(png, kPngSignature, 8) != 0) return false;
  if (LoadU32BE(png + 8) != 13 || std::memcmp(png + 12, "IHDR", 4) != 0) return false;
  *width = LoadU32BE(png + 16);
  *height = LoadU32BE(png + 20);
  return *width != 0 && *height != 0;
}

// One zstd frame holding `n` bytes verbatim as a single raw block: single-segment,
// a 4-byte content size, no checksum, no dictionary. `n` must fit one block.
inline void AppendRawFrame(std::vector<uint8_t> *out, const uint8_t *src, uint64_t n) {
  StoreU32(out, 0xFD2FB528u);
  out->push_back(0xA0);  // content-size field 4 bytes, single segment
  StoreU32(out, static_cast<uint32_t>(n));
  const uint32_t block = 1u | (static_cast<uint32_t>(n) << 3);  // last block, raw
  for (int i = 0; i < 3; ++i) out->push_back(static_cast<uint8_t>((block >> (8 * i)) & 0xFF));
  out->insert(out->end(), src, src + n);
}

// The decoded descriptor of a thumbnail record.
struct ThumbnailRecord {
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t png_offset = 0;  // into the record
  uint64_t png_nbytes = 0;
  uint32_t data_crc = 0;
};

// Parse a decompressed thumbnail record. False when it is not one this build reads.
inline bool ParseThumbnailRecord(const uint8_t *rec, uint64_t n, ThumbnailRecord *out) {
  if (n < kThumbnailDescriptorBytes) return false;
  if (std::memcmp(rec, kThumbnailMagic, 8) != 0) return false;
  if (LoadU32LE(rec + 8) != kThumbnailVersion) return false;
  const uint32_t header_bytes = LoadU32LE(rec + 12);
  if (header_bytes < kThumbnailDescriptorBytes || header_bytes > n) return false;
  out->width = LoadU32LE(rec + 16);
  out->height = LoadU32LE(rec + 20);
  out->png_nbytes = LoadU32LE(rec + 24);
  out->data_crc = LoadU32LE(rec + kThumbnailCrcOffset);
  out->png_offset = header_bytes;
  return out->png_nbytes == n - header_bytes;
}

}  // namespace pvzstd::detail

#endif  // PVZSTD_THUMBNAIL_H
