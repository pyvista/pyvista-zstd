// 64-bit file positioning.
//
// std::fseek and std::ftell take and return `long`, which is 32 bits on
// Windows -- 64-bit Windows included -- and on every 32-bit target. A
// container larger than 2 GiB cannot be addressed through them: the seek
// offset wraps and the tell reports a truncated size, so the trailer is read
// from the wrong place rather than reported as unreachable. These wrap the
// platform's 64-bit equivalents so an offset stays an offset.
//
// The POSIX branch relies on _FILE_OFFSET_BITS=64, which cpp/CMakeLists.txt
// defines; without it a 32-bit libc gives off_t only 32 bits and fseeko is no
// better than fseek.

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
