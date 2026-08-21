// 64-bit file positioning. std::fseek/ftell use `long`, 32 bits on Windows and
// 32-bit targets, so past 2 GiB the offset wraps and the trailer is read from the
// wrong place. The POSIX branch needs _FILE_OFFSET_BITS=64, set in CMakeLists.

#ifndef PVZSTD_FILEIO_H
#define PVZSTD_FILEIO_H

#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#endif

namespace pvzstd::detail {

inline int SeekTo(std::FILE *fp, int64_t offset, int origin) {
#if defined(_WIN32)
  return _fseeki64(fp, offset, origin);
#else
  return ::fseeko(fp, static_cast<off_t>(offset), origin);
#endif
}

// Current position, or -1.
inline int64_t TellAt(std::FILE *fp) {
#if defined(_WIN32)
  return _ftelli64(fp);
#else
  return static_cast<int64_t>(::ftello(fp));
#endif
}

}  // namespace pvzstd::detail

#endif  // PVZSTD_FILEIO_H
