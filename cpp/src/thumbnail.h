// The optional thumbnail pair. See doc/format/container-v2.md, section 11.

#ifndef PVZSTD_THUMBNAIL_H
#define PVZSTD_THUMBNAIL_H

#include <cstdint>
#include <cstring>
#include <vector>

#include "detail.h"
#include "pvzstd/pvzstd.h"

namespace pvzstd::detail {

constexpr char kThumbnailName[] = "__thumbnail";
constexpr char kThumbnailMagic[8] = {'P', 'V', 'Z', 'T', 'H', 'U', 'M', 'B'};
constexpr uint32_t kThumbnailVersion = 1;
constexpr uint64_t kThumbnailDescriptorBytes = 32;
constexpr uint32_t kSkippableMagic = 0x184D2A50u;
constexpr uint64_t kSkippablePrefixBytes = 8;        // magic and u32 length
constexpr uint64_t kRawFramePrefixBytes = 12;        // magic, descriptor, u32 size, block header
constexpr uint64_t kThumbnailArrayHeaderBytes = 43;  // see ThumbnailArrayHeader
constexpr uint64_t kThumbnailHeaderFrameBytes = kRawFramePrefixBytes + kThumbnailArrayHeaderBytes;
constexpr uint64_t kThumbnailRecordOffset = kThumbnailHeaderFrameBytes + kSkippablePrefixBytes;
constexpr uint64_t kThumbnailCrcAt = kThumbnailRecordOffset + 28;
constexpr uint8_t kPngSignature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

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

// Staleness CRC: the trailer index entries of every frame after the thumbnail pair.
inline uint32_t IndexCrc(const uint8_t *index, uint64_t n_frames) {
  return Crc32(index + 2 * kIndexEntryBytes, (n_frames - 2) * kIndexEntryBytes);
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

// The array header of the thumbnail pair: an empty |u1 array named __thumbnail.
inline std::vector<uint8_t> ThumbnailArrayHeader() {
  return PackArrayMetadata(kThumbnailName, "|u1", {0}, PVZSTD_FILTER_NONE);
}

// Every byte of the thumbnail pair before the PNG, with the staleness CRC left at zero.
inline std::vector<uint8_t> ThumbnailPrefix(uint64_t png_nbytes, uint32_t width, uint32_t height) {
  const std::vector<uint8_t> header = ThumbnailArrayHeader();
  std::vector<uint8_t> out;
  StoreU32(&out, 0xFD2FB528u);
  out.push_back(0xA0);  // 4-byte content size, single segment, no checksum
  StoreU32(&out, static_cast<uint32_t>(header.size()));
  const uint32_t block = 1u | (static_cast<uint32_t>(header.size()) << 3);  // last block, raw
  for (int i = 0; i < 3; ++i) out.push_back(static_cast<uint8_t>((block >> (8 * i)) & 0xFF));
  out.insert(out.end(), header.begin(), header.end());
  StoreU32(&out, kSkippableMagic);
  StoreU32(&out, static_cast<uint32_t>(kThumbnailDescriptorBytes + png_nbytes));
  out.insert(out.end(), kThumbnailMagic, kThumbnailMagic + 8);
  StoreU32(&out, kThumbnailVersion);
  StoreU32(&out, static_cast<uint32_t>(kThumbnailDescriptorBytes));
  StoreU32(&out, width);
  StoreU32(&out, height);
  StoreU32(&out, static_cast<uint32_t>(png_nbytes));
  StoreU32(&out, 0);
  return out;
}

// Whether a container starts with the raw header frame and skippable magic of a thumbnail pair.
inline bool StartsWithThumbnailPrefix(const uint8_t *data, uint64_t n) {
  static const std::vector<uint8_t> prefix = ThumbnailPrefix(0, 0, 0);
  const uint64_t fixed = kThumbnailHeaderFrameBytes + 4;  // header frame and skippable magic
  return n >= fixed && std::memcmp(data, prefix.data(), fixed) == 0;
}

// The decoded descriptor of a thumbnail record.
struct ThumbnailRecord {
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t png_offset = 0;  // into the record
  uint64_t png_nbytes = 0;
  uint32_t data_crc = 0;
};

// Parse a thumbnail record of `n` bytes. False when it is not one this build reads.
inline bool ParseThumbnailRecord(const uint8_t *rec, uint64_t n, ThumbnailRecord *out) {
  if (n < kThumbnailDescriptorBytes) return false;
  if (std::memcmp(rec, kThumbnailMagic, 8) != 0) return false;
  if (LoadU32(rec + 8) != kThumbnailVersion) return false;
  const uint32_t header_bytes = LoadU32(rec + 12);
  if (header_bytes < kThumbnailDescriptorBytes || header_bytes > n) return false;
  out->width = LoadU32(rec + 16);
  out->height = LoadU32(rec + 20);
  out->png_nbytes = LoadU32(rec + 24);
  out->data_crc = LoadU32(rec + 28);
  out->png_offset = header_bytes;
  if (out->png_nbytes != n - header_bytes || out->png_nbytes < sizeof(kPngSignature)) return false;
  return std::memcmp(rec + header_bytes, kPngSignature, sizeof(kPngSignature)) == 0;
}

}  // namespace pvzstd::detail

#endif  // PVZSTD_THUMBNAIL_H
