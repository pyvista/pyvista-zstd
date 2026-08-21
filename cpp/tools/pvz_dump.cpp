// pvz_dump -- print a deterministic description of every array in a .pv file.
//
// This is the C++ side of the conformance comparison: the Python oracle in
// tests/conformance emits the same lines from the same file, and the two must
// match byte for byte. A checksum over the decoded payload is what makes the
// comparison bit-exact rather than merely structural.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pvzstd/pvzstd.h"

namespace {

// FNV-1a, 64-bit. Chosen because it is trivial to reimplement identically on
// the other side of the comparison.
uint64_t Fnv1a(const uint8_t *data, uint64_t n) {
  uint64_t h = 1469598103934665603ull;
  for (uint64_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: pvz_dump <file.pv>\n");
    return 2;
  }

  pvz_reader *reader = nullptr;
  pvz_status st = pvz_open(argv[1], &reader);
  if (st != PVZ_OK) {
    std::fprintf(stderr, "pvz_open failed: %s\n", pvz_status_message(st));
    return 1;
  }

  const uint64_t count = pvz_array_count(reader);
  std::printf("arrays %" PRIu64 "\n", count);
  std::printf("ds_metadata %d\n", pvz_ds_metadata_json(reader) != nullptr ? 1 : 0);
  std::printf("file_metadata %d\n", pvz_file_metadata_json(reader) != nullptr ? 1 : 0);

  std::vector<uint8_t> buf;
  for (uint64_t i = 0; i < count; ++i) {
    pvz_array_info info;
    st = pvz_array_info_at(reader, i, &info);
    if (st != PVZ_OK) {
      std::fprintf(stderr, "info(%" PRIu64 ") failed: %s\n", i, pvz_status_message(st));
      pvz_close(reader);
      return 1;
    }

    buf.assign(static_cast<size_t>(info.nbytes), 0);
    st = pvz_read_array_at(reader, i, buf.empty() ? buf.data() : &buf[0], info.nbytes);
    if (st != PVZ_OK) {
      std::fprintf(stderr, "read(%s) failed: %s\n", info.name, pvz_status_message(st));
      pvz_close(reader);
      return 1;
    }

    std::printf("array %s %s filter=%u nbytes=%" PRIu64 " shape=[", info.name, info.dtype,
                static_cast<unsigned>(info.filter_id), info.nbytes);
    for (uint32_t d = 0; d < info.ndim; ++d) {
      std::printf("%s%" PRIu64, d ? "," : "", info.shape[d]);
    }
    std::printf("] fnv=%016" PRIx64 "\n", Fnv1a(buf.empty() ? nullptr : &buf[0], info.nbytes));
  }

  pvz_close(reader);
  return 0;
}
