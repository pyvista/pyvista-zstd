// pvzstd_stream -- drive the streaming writer from spec files, for the gate.
//
// A developer tool, not part of the shipped library. It takes one spec file
// per commit, so a run reproduces exactly the sequence of appends the
// reference side made, and the two files can be compared byte for byte.
//
// usage: pvzstd_stream [--expect=N] <container> <shuffle> <spec>...
//
//   --expect  count to close with; defaults to the number of specs. Only a
//             mismatching value is interesting, which is why it is settable:
//             the close-time count check is otherwise unreachable from here.
//   shuffle   0 never, 1 always, 2 auto
//   spec      one per commit; each line is
//             name <TAB> dtype <TAB> dtype_name <TAB> shape_csv <TAB> raw_path
//
// Each commit is reported to stderr as "commit <i> <seconds> <bytes_read>".
//
// The byte count is the delta in this process's rchar (/proc/self/io), or -1
// where that is unavailable. It is there because wall time cannot gate what
// this tool exists to check: a stream that wrongly re-reads the container on
// every commit is served from page cache, which is too cheap to separate from
// compression noise. Bytes read is the same property measured deterministically
// -- reading a growing file grows it, holding state in memory does not.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pvzstd/pvzstd.h"

namespace {

struct Spec {
  std::string name;
  std::string dtype;
  std::string dtype_name;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;
};

bool ReadWholeFile(const std::string &path, std::vector<uint8_t> *out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(n));
  return n == 0 || static_cast<bool>(f.read(reinterpret_cast<char *>(out->data()), n));
}

// Bytes this process has read, page cache included -- rchar, not read_bytes,
// because a re-read served entirely from cache is exactly the mistake being
// watched for and never reaches the device. Returns -1 where unavailable.
long long BytesRead() {
  std::ifstream io("/proc/self/io");
  if (!io) return -1;
  std::string key;
  long long value = 0;
  while (io >> key >> value) {
    if (key == "rchar:") return value;
  }
  return -1;
}

std::vector<std::string> Split(const std::string &s, char sep) {
  std::vector<std::string> parts;
  std::string item;
  std::istringstream stream(s);
  while (std::getline(stream, item, sep)) parts.push_back(item);
  return parts;
}

bool LoadSpec(const std::string &path, std::vector<Spec> *out) {
  std::ifstream spec_file(path);
  if (!spec_file) return false;
  std::string line;
  while (std::getline(spec_file, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = Split(line, '\t');
    if (fields.size() != 5) return false;
    Spec s;
    s.name = fields[0];
    s.dtype = fields[1];
    s.dtype_name = fields[2];
    if (!fields[3].empty()) {
      for (const std::string &dim : Split(fields[3], ',')) {
        s.shape.push_back(std::strtoull(dim.c_str(), nullptr, 10));
      }
    }
    if (!ReadWholeFile(fields[4], &s.data)) return false;
    out->push_back(std::move(s));
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  long long expect = -1;
  int first = 1;
  const std::string expect_flag = "--expect=";
  if (argc > 1 && std::string(argv[1]).compare(0, expect_flag.size(), expect_flag) == 0) {
    expect = std::atoll(argv[1] + expect_flag.size());
    first = 2;
  }
  argv += first - 1;
  argc -= first - 1;

  if (argc < 4) {
    std::cerr << "usage: pvzstd_stream [--expect=N] <container> <shuffle> <spec>...\n";
    return 2;
  }
  const int shuffle_code = std::atoi(argv[2]);
  if (shuffle_code < 0 || shuffle_code > 2) {
    std::cerr << "shuffle must be 0, 1 or 2\n";
    return 2;
  }

  pvzstd_stream *stream = nullptr;
  pvzstd_status st = pvzstd_stream_open(argv[1], &stream);
  if (st != PVZSTD_OK) {
    std::cerr << "pvzstd_stream_open: " << pvzstd_status_message(st) << "\n";
    return 1;
  }

  const int n_commits = argc - 3;
  for (int c = 0; c < n_commits; ++c) {
    std::vector<Spec> specs;
    if (!LoadSpec(argv[3 + c], &specs)) {
      std::cerr << "cannot read spec " << argv[3 + c] << "\n";
      pvzstd_stream_free(stream);
      return 2;
    }
    std::vector<pvzstd_append_array> arrays;
    arrays.reserve(specs.size());
    for (const Spec &s : specs) {
      pvzstd_append_array a;
      a.name = s.name.c_str();
      a.dtype = s.dtype.c_str();
      a.dtype_name = s.dtype_name.c_str();
      a.shape = s.shape.empty() ? nullptr : s.shape.data();
      a.ndim = static_cast<uint32_t>(s.shape.size());
      a.data = s.data.empty() ? nullptr : s.data.data();
      a.nbytes = s.data.size();
      arrays.push_back(a);
    }
    // Timed here rather than around the process: the gate is the cost of a
    // commit, and process start-up would swamp it.
    const long long read0 = BytesRead();
    const auto t0 = std::chrono::steady_clock::now();
    st = pvzstd_stream_append(stream, arrays.data(), arrays.size(),
                              static_cast<pvzstd_shuffle_mode>(shuffle_code));
    const auto t1 = std::chrono::steady_clock::now();
    const long long read1 = BytesRead();
    std::fprintf(stderr, "commit %d %.6f %lld\n", c, std::chrono::duration<double>(t1 - t0).count(),
                 (read0 < 0 || read1 < 0) ? -1 : read1 - read0);
    if (st != PVZSTD_OK) {
      std::cerr << "pvzstd_stream_append: " << pvzstd_status_message(st) << "\n";
      pvzstd_stream_free(stream);
      return 1;
    }
  }

  st = pvzstd_stream_close(stream, static_cast<uint64_t>(expect < 0 ? n_commits : expect));
  pvzstd_stream_free(stream);
  if (st != PVZSTD_OK) {
    std::cerr << "pvzstd_stream_close: " << pvzstd_status_message(st) << "\n";
    return 1;
  }
  return 0;
}
