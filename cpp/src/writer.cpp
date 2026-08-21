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

#include <zstd.h>

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "detail.h"
#include "pvzstd/pvzstd.h"

// The primitives below moved to detail.h when the append path came to need the
// same ones; this brings them back into scope unqualified.
using namespace pvzstd::detail;  // NOLINT(google-build-using-namespace)

namespace {

struct PendingArray {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;
  bool is_metadata_json = false;
};

}  // namespace

struct pvzstd_writer {
  int level = kDefaultLevel;
  int threads = PVZSTD_THREADS_AUTO;
  pvzstd_shuffle_mode shuffle = PVZSTD_SHUFFLE_NEVER;
  bool fixed_width_cells = false;
  std::vector<PendingArray> arrays;
};

extern "C" {

pvzstd_status pvzstd_writer_create(pvzstd_writer **out) {
  if (out == nullptr) return PVZSTD_E_INVALID;
  *out = new (std::nothrow) pvzstd_writer();
  return *out == nullptr ? PVZSTD_E_NOMEM : PVZSTD_OK;
}

void pvzstd_writer_free(pvzstd_writer *writer) { delete writer; }

pvzstd_status pvzstd_writer_set_level(pvzstd_writer *writer, int level) {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->level = level;
  return PVZSTD_OK;
}

pvzstd_status pvzstd_writer_set_threads(pvzstd_writer *writer, int n_threads) {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->threads = n_threads;
  return PVZSTD_OK;
}

pvzstd_status pvzstd_writer_set_shuffle(pvzstd_writer *writer, pvzstd_shuffle_mode mode) {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->shuffle = mode;
  return PVZSTD_OK;
}

pvzstd_status pvzstd_writer_set_fixed_width_cells(pvzstd_writer *writer, int enabled) {
  if (writer == nullptr) return PVZSTD_E_INVALID;
  writer->fixed_width_cells = enabled != 0;
  return PVZSTD_OK;
}

pvzstd_status pvzstd_writer_add_array(pvzstd_writer *writer, const char *name, const char *dtype,
                                      const uint64_t *shape, uint32_t ndim, const void *data,
                                      uint64_t nbytes) {
  if (writer == nullptr || name == nullptr || dtype == nullptr) return PVZSTD_E_INVALID;
  if (nbytes > 0 && data == nullptr) return PVZSTD_E_INVALID;
  if (ndim > 0 && shape == nullptr) return PVZSTD_E_INVALID;
  if (std::strlen(dtype) > PVZSTD_DTYPE_LEN) return PVZSTD_E_INVALID;
  if (!ParseDtype(dtype).valid) return PVZSTD_E_INVALID;

  PendingArray entry;
  entry.name = name;
  entry.dtype = dtype;
  entry.shape.assign(shape, shape + ndim);
  try {
    entry.data.assign(static_cast<const uint8_t *>(data),
                      static_cast<const uint8_t *>(data) + nbytes);
    writer->arrays.push_back(std::move(entry));
  } catch (const std::bad_alloc &) {
    return PVZSTD_E_NOMEM;
  }
  return PVZSTD_OK;
}

pvzstd_status pvzstd_writer_set_ds_metadata(pvzstd_writer *writer, const char *uid,
                                            const char *json) {
  if (writer == nullptr || uid == nullptr || json == nullptr) return PVZSTD_E_INVALID;
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
    return PVZSTD_E_NOMEM;
  }
  return PVZSTD_OK;
}

pvzstd_status pvzstd_writer_write(pvzstd_writer *writer, const char *path) {
  if (writer == nullptr || path == nullptr) return PVZSTD_E_INVALID;
  if (writer->arrays.empty()) return PVZSTD_E_INVALID;

  // 1. Resolve each array's filter, exactly as the reference writer does.
  std::vector<uint8_t> filters(writer->arrays.size(), PVZSTD_FILTER_NONE);
  bool any_filtered = false;
  for (size_t i = 0; i < writer->arrays.size(); ++i) {
    const PendingArray &a = writer->arrays[i];
    const Dtype d = ParseDtype(a.dtype.c_str());
    bool use = false;
    // Note the absent emptiness check: the reference decides from dtype, so an
    // empty multibyte array under shuffle=True is still *recorded* as
    // filtered. Shuffling nothing is a no-op, but the header byte is not.
    if (writer->shuffle != PVZSTD_SHUFFLE_NEVER && d.itemsize > 1) {
      if (writer->shuffle == PVZSTD_SHUFFLE_ALWAYS) {
        use = true;
      } else if (d.kind == 'f' || d.kind == 'c') {
        use = AutoShuffleBeneficial(a.data.data(), a.data.size(), d.itemsize, writer->level);
      }
    }
    if (use) {
      filters[i] = PVZSTD_FILTER_SHUFFLE;
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
    return PVZSTD_E_NOMEM;
  }
  // Appended to a local list rather than to the writer's own, so writing the
  // same writer to a second path emits one metadata frame and not two. The
  // frame is generated per write because it names every frame in the file.
  std::vector<const PendingArray *> emit;
  emit.reserve(writer->arrays.size() + 1);
  for (const PendingArray &a : writer->arrays) emit.push_back(&a);
  emit.push_back(&meta_entry);
  filters.push_back(PVZSTD_FILTER_NONE);  // the JSON blob is never shuffled

  // 4. Worker count, from the total computed above.
  const int workers = EffectiveWorkers(ResolveThreads(writer->threads, total_bytes));

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
    // Written only when a filter is in use: absence means PVZSTD_FILTER_NONE and
    // keeps unfiltered frames byte-identical to the legacy layout.
    if (filters[i] != PVZSTD_FILTER_NONE) header.push_back(filters[i]);

    const uint8_t *payload_ptr = a.data.data();
    uint64_t payload_len = a.data.size();
    if (filters[i] == PVZSTD_FILTER_SHUFFLE) {
      payload.assign(a.data.size(), 0);
      ShuffleBytes(a.data.data(), payload.data(), a.data.size(), d.itemsize);
      payload_ptr = payload.data();
      payload_len = payload.size();
    }

    const uint8_t *pieces[2] = {header.data(), payload_ptr};
    const uint64_t lengths[2] = {header.size(), payload_len};
    for (int part = 0; part < 2 && st == PVZSTD_OK; ++part) {
      st = CompressFrame(pieces[part], lengths[part], writer->level, workers, &frame);
      if (st != PVZSTD_OK) break;
      if (!frame.empty() && std::fwrite(frame.data(), 1, frame.size(), fp) != frame.size()) {
        st = PVZSTD_E_IO;
        break;
      }
      offset += frame.size();
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
  }

  if (std::fclose(fp) != 0 && st == PVZSTD_OK) st = PVZSTD_E_IO;
  return st;
}

}  // extern "C"
