// pvzstd_append -- drive pvzstd_append_arrays from a spec file, for the gate.
// A developer tool, not part of the shipped library.
//
// usage: pvzstd_append <container> <level> <shuffle> <spec>
//
//   level    compression level, or -1000 to reuse the file's
//   shuffle  0 never, 1 always, 2 auto
//   spec     one array per line, tab-separated:
//            name <TAB> dtype <TAB> dtype_name <TAB> shape_csv <TAB> raw_path
//            shape_csv is empty for a 0-d array.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::vector<std::string> Split(const std::string &s, char sep) {
  std::vector<std::string> parts;
  std::string item;
  std::istringstream stream(s);
  while (std::getline(stream, item, sep)) parts.push_back(item);
  return parts;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: pvzstd_append <container> <level> <shuffle> <spec>\n";
    return 2;
  }
  const std::string container = argv[1];
  const int level = std::atoi(argv[2]);
  const int shuffle_code = std::atoi(argv[3]);
  if (shuffle_code < 0 || shuffle_code > 2) {
    std::cerr << "shuffle must be 0, 1 or 2\n";
    return 2;
  }

  std::ifstream spec_file(argv[4]);
  if (!spec_file) {
    std::cerr << "cannot open spec " << argv[4] << "\n";
    return 2;
  }

  std::vector<Spec> specs;
  std::string line;
  while (std::getline(spec_file, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> fields = Split(line, '\t');
    if (fields.size() != 5) {
      std::cerr << "spec line needs 5 tab-separated fields, got " << fields.size() << "\n";
      return 2;
    }
    Spec s;
    s.name = fields[0];
    s.dtype = fields[1];
    s.dtype_name = fields[2];
    if (!fields[3].empty()) {
      for (const std::string &dim : Split(fields[3], ',')) {
        s.shape.push_back(std::strtoull(dim.c_str(), nullptr, 10));
      }
    }
    if (!ReadWholeFile(fields[4], &s.data)) {
      std::cerr << "cannot read raw data " << fields[4] << "\n";
      return 2;
    }
    specs.push_back(std::move(s));
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

  const pvzstd_status st =
      pvzstd_append_arrays(container.c_str(), arrays.data(), arrays.size(), level,
                           static_cast<pvzstd_shuffle_mode>(shuffle_code), nullptr, nullptr);
  if (st != PVZSTD_OK) {
    std::cerr << "pvzstd_append_arrays: " << pvzstd_status_message(st) << "\n";
    return 1;
  }
  return 0;
}
