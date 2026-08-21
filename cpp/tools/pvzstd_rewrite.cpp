// pvzstd_rewrite -- read a container with the C++ reader, write it back with
// the C++ writer.
//
// The writer's conformance harness: given a reference file and the settings that
// produced it, the output must be byte-identical. Reading first exercises both
// halves against real reference bytes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pvzstd/pvzstd.h"

namespace {

int Fail(const char *what, pvzstd_status st) {
  std::fprintf(stderr, "%s: %s\n", what, pvzstd_status_message(st));
  return 1;
}

}  // namespace

int main(int argc, char **argv) {
  constexpr int kExpectedArgs = 8;
  if (argc != kExpectedArgs) {
    std::fprintf(stderr,
                 "usage: pvzstd_rewrite <in.pv> <out.pv> <level> <threads> "
                 "<shuffle:0|1|2> <fixed_width_cells:0|1> <uid>\n");
    return 2;
  }
  const char *in_path = argv[1];
  const char *out_path = argv[2];
  const int level = std::atoi(argv[3]);
  const int threads = std::atoi(argv[4]);
  const int shuffle = std::atoi(argv[5]);
  const int fixed = std::atoi(argv[6]);
  const char *uid = argv[7];

  pvzstd_reader *reader = nullptr;
  pvzstd_status st = pvzstd_open(in_path, &reader);
  if (st != PVZSTD_OK) return Fail("pvzstd_open", st);

  pvzstd_writer *writer = nullptr;
  st = pvzstd_writer_create(&writer);
  if (st != PVZSTD_OK) {
    pvzstd_close(reader);
    return Fail("pvzstd_writer_create", st);
  }
  pvzstd_writer_set_level(writer, level);
  pvzstd_writer_set_threads(writer, threads);
  pvzstd_writer_set_shuffle(writer, static_cast<pvzstd_shuffle_mode>(shuffle));
  pvzstd_writer_set_fixed_width_cells(writer, fixed);

  const uint64_t count = pvzstd_array_count(reader);
  std::vector<uint8_t> buf;
  for (uint64_t i = 0; i < count; ++i) {
    pvzstd_array_info info;
    st = pvzstd_array_info_at(reader, i, &info);
    if (st != PVZSTD_OK) break;
    buf.assign(static_cast<size_t>(info.nbytes), 0);
    st = pvzstd_read_array_at(reader, i, buf.empty() ? buf.data() : &buf[0], info.nbytes);
    if (st != PVZSTD_OK) break;
    st = pvzstd_writer_add_array(writer, info.name, info.dtype, info.shape, info.ndim,
                                 buf.empty() ? nullptr : &buf[0], info.nbytes);
    if (st != PVZSTD_OK) break;
  }

  // The dataset-metadata frame is the last pair before the file-metadata
  // frame, so it is added after every array.
  if (st == PVZSTD_OK) {
    const char *ds_json = pvzstd_ds_metadata_json(reader);
    if (ds_json != nullptr) st = pvzstd_writer_set_ds_metadata(writer, uid, ds_json);
  }
  if (st == PVZSTD_OK) st = pvzstd_writer_write(writer, out_path);

  pvzstd_writer_free(writer);
  pvzstd_close(reader);
  if (st != PVZSTD_OK) return Fail("rewrite", st);
  return 0;
}
