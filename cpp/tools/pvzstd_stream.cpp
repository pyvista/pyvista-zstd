// pvzstd_stream -- drive the streaming writer from spec files, for the gate.
// A developer tool, not part of the shipped library.
//
// usage: pvzstd_stream [--expect=N] [--poison] <container> <shuffle> <spec>...
//
//   --expect  count to close with; defaults to the number of specs. Settable
//             so the close-time count check is reachable from here.
//   --poison  fail one commit part-way, then report what the calls after it
//             return. See PoisonMode below.
//   shuffle   0 never, 1 always, 2 auto
//   spec      one per commit; each line is
//             name <TAB> dtype <TAB> dtype_name <TAB> shape_csv <TAB> raw_path
//
// Each commit reports "commit <i> <seconds> <bytes_read>" to stderr. The byte
// count is this process's rchar delta (-1 where unavailable): a wrongly re-read
// container is served from page cache and too cheap to catch on wall time.

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

// rchar, not read_bytes: the mistake being watched for never reaches the device.
// Returns -1 where unavailable.
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

// Fail a commit after it has begun mutating the stream, then report what the
// calls after it return, as "<call> <status>" lines on stdout.
//
// The fault is an array claiming an impossible size. Validation passes, the tail
// frames are dropped, and the payload buffer is the first thing that cannot be
// had -- which is the shape of the problem: a commit that stopped part-way
// through, leaving the frame list describing frames that are not where it says.
// No privileges and no full disk needed, so it runs everywhere the suite does.
//
// The container is left damaged past its last good commit. Copy it first.
int PoisonMode(const char *path, const std::vector<Spec> &specs, pvzstd_shuffle_mode shuffle) {
  pvzstd_stream *stream = nullptr;
  pvzstd_status st = pvzstd_stream_open(path, &stream);
  if (st != PVZSTD_OK) {
    std::cerr << "pvzstd_stream_open: " << pvzstd_status_message(st) << "\n";
    return 1;
  }

  const uint64_t impossible = static_cast<uint64_t>(1) << 60;
  uint64_t shape[1] = {impossible};
  uint8_t byte = 0;
  pvzstd_append_array huge;
  huge.name = "poison";
  huge.dtype = "|u1";
  huge.dtype_name = "poison";
  huge.shape = shape;
  huge.ndim = 1;
  huge.data = &byte;
  huge.nbytes = impossible;
  // NEVER, or the auto-shuffle heuristic reads nbytes bytes of `data` first.
  std::printf("poison %d\n", pvzstd_stream_append(stream, &huge, 1, PVZSTD_SHUFFLE_NEVER));

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
  std::printf("append %d\n", pvzstd_stream_append(stream, arrays.data(), arrays.size(), shuffle));
  std::printf("commits %llu\n",
              static_cast<unsigned long long>(pvzstd_stream_commit_count(stream)));
  std::printf("close %d\n", pvzstd_stream_close(stream, 1));
  pvzstd_stream_free(stream);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  long long expect = -1;
  bool poison = false;
  int first = 1;
  const std::string expect_flag = "--expect=";
  while (argc > first) {
    const std::string arg(argv[first]);
    if (arg.compare(0, expect_flag.size(), expect_flag) == 0) {
      expect = std::atoll(argv[first] + expect_flag.size());
    } else if (arg == "--poison") {
      poison = true;
    } else {
      break;
    }
    ++first;
  }
  argv += first - 1;
  argc -= first - 1;

  if (argc < 4) {
    std::cerr << "usage: pvzstd_stream [--expect=N] [--poison] <container> <shuffle> <spec>...\n";
    return 2;
  }
  const int shuffle_code = std::atoi(argv[2]);
  if (shuffle_code < 0 || shuffle_code > 2) {
    std::cerr << "shuffle must be 0, 1 or 2\n";
    return 2;
  }

  if (poison) {
    std::vector<Spec> specs;
    if (!LoadSpec(argv[3], &specs)) {
      std::cerr << "cannot read spec " << argv[3] << "\n";
      return 2;
    }
    return PoisonMode(argv[1], specs, static_cast<pvzstd_shuffle_mode>(shuffle_code));
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
    // Around the commit, not the process, whose start-up would swamp it.
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
