// 64-bit file positioning, unique staging files, and file identity.
//
// std::fseek/ftell use `long`, 32 bits on Windows and 32-bit targets, so past
// 2 GiB the offset wraps and the trailer is read from the wrong place. The POSIX
// branch needs _FILE_OFFSET_BITS=64, set in CMakeLists.

#ifndef PVZSTD_FILEIO_H
#define PVZSTD_FILEIO_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
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

// Whether a path is there, asked without opening it.
//
// Opening it is not a way to ask this. A handle on a file, even one held for an
// instant and only for reading, is enough to make another process's delete of
// that file fail on Windows, and the one file this is asked about is the append
// lock, whose holder deletes it to release it.
inline bool PathExists(const char *path) {
#if defined(_WIN32)
  return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
  struct stat st{};
  return ::stat(path, &st) == 0;
#endif
}

// The append lock for one container: a file beside it that exists only while an
// append is in flight.
//
// Exclusive creation is the primitive because it is the only form of mutual
// exclusion every target this builds for implements the same way. flock() would
// be the obvious choice and is not usable here: on the WebAssembly target it
// returns success without locking anything, so a second holder is handed the
// lock the first one thinks it has -- a guarantee stated for a target that does
// not keep it is worse than none, because it is the one nobody re-checks.
//
// The cost of doing it with a file is that a process killed mid-append leaves
// the lock behind, and every later append to that container is refused until it
// is removed. That is a visible, named, recoverable failure; the alternative it
// replaces is one writer's arrays disappearing with both writers told they
// succeeded.
enum class LockResult {
  kAcquired,
  kHeld,   // somebody else has this container
  kFailed  // the lock file could not be created at all
};

class AppendLock {
 public:
  // Retries of the exclusive create, spent only on a lock seen to be gone.
  static constexpr int kAcquireAttempts = 8;

  AppendLock() = default;
  ~AppendLock() { Release(); }
  AppendLock(const AppendLock &) = delete;
  AppendLock &operator=(const AppendLock &) = delete;

  LockResult Acquire(const std::string &container) {
    const std::string lock = container + ".append.lock";
    for (int attempt = 0; attempt < kAcquireAttempts; ++attempt) {
      // "x" fails if the file exists, and creating-if-absent is one operation:
      // testing first and creating second is the race this is here to close.
      std::FILE *fp = std::fopen(lock.c_str(), "wbx");
      if (fp != nullptr) {
        std::fclose(fp);
        path_ = lock;
        return LockResult::kAcquired;
      }
      // Held, or unwritable? Told apart by whether the lock is there rather
      // than by errno, whose value for an exclusive create onto an existing
      // file is not the same number on every target. A caller told "another
      // append holds this" about a directory it cannot write to would go
      // looking for a writer that does not exist.
      if (PathExists(lock.c_str())) return LockResult::kHeld;
      // Neither: the holder released the lock in the moment between the create
      // and the question, so this call lost a race rather than found a broken
      // directory, and the create is worth making again. Reporting the empty
      // answer as a failure is how ordinary contention turned into an I/O
      // error, which is the one outcome a losing append must never get.
      //
      // Bounded, because a create that keeps failing over a lock that is never
      // there is a directory this process cannot write to. Contention resolves
      // inside the bound: whoever is racing here either creates the file or
      // finds somebody else's.
    }
    return LockResult::kFailed;
  }

  void Release() {
    if (path_.empty()) return;
    std::remove(path_.c_str());
    path_.clear();
  }

 private:
  std::string path_;
};

// Which file something is, as opposed to which name it goes by.
//
// A rename replaces a directory entry, so a path that named one file before a
// concurrent writer committed names a different file afterwards while the
// handle already open still refers to the old one. Comparing these two answers
// is how an edit staged against the old file notices it is about to overwrite
// somebody else's.
struct FileIdentity {
  bool known = false;  // false where the filesystem could not answer
  uint64_t volume = 0;
  uint64_t file = 0;
};

// Whether two identities name the same file. An unanswered identity compares
// equal to everything on purpose: the callers use this to refuse an edit, and a
// filesystem that reports nothing useful must not turn that into a refusal of
// every edit.
inline bool SameFile(const FileIdentity &a, const FileIdentity &b) {
  if (!a.known || !b.known) return true;
  return a.volume == b.volume && a.file == b.file;
}

#if defined(_WIN32)

inline FileIdentity IdentifyHandle(HANDLE handle) {
  FileIdentity id;
  BY_HANDLE_FILE_INFORMATION info;
  if (handle != INVALID_HANDLE_VALUE && GetFileInformationByHandle(handle, &info) != 0) {
    id.known = true;
    id.volume = info.dwVolumeSerialNumber;
    id.file = (static_cast<uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
  }
  return id;
}

inline FileIdentity IdentifyOpen(std::FILE *fp) {
  if (fp == nullptr) return FileIdentity{};
  const intptr_t raw = _get_osfhandle(_fileno(fp));
  if (raw == -1) return FileIdentity{};
  return IdentifyHandle(reinterpret_cast<HANDLE>(raw));
}

inline FileIdentity IdentifyPath(const char *path) {
  // Every sharing mode: this only reads the file's identity, and refusing to
  // answer because somebody else has it open would defeat the point.
  HANDLE handle = CreateFileA(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) return FileIdentity{};
  const FileIdentity id = IdentifyHandle(handle);
  CloseHandle(handle);
  return id;
}

// Create and open a staging file beside `path`, named by the OS rather than
// derived from `path`: two callers staging an edit to one container must never
// be handed the same name, or each writes into the other's file. `model` is
// ignored here -- a new file inherits its ACL from the directory.
inline std::FILE *OpenUniqueTemp(const std::string &path, std::FILE *model, std::string *tmp_path) {
  (void)model;
  const size_t cut = path.find_last_of("\\/");
  const std::string dir = (cut == std::string::npos) ? std::string(".") : path.substr(0, cut + 1);
  char name[MAX_PATH];
  // GetTempFileName creates the file, so no second caller can be handed the
  // name it just returned.
  if (GetTempFileNameA(dir.c_str(), "zst", 0, name) == 0) return nullptr;
  std::FILE *fp = std::fopen(name, "wb");
  if (fp == nullptr) {
    std::remove(name);
    return nullptr;
  }
  *tmp_path = name;
  return fp;
}

#else

inline FileIdentity IdentifyOpen(std::FILE *fp) {
  FileIdentity id;
  struct stat st{};
  if (fp != nullptr && ::fstat(::fileno(fp), &st) == 0) {
    id.known = true;
    id.volume = static_cast<uint64_t>(st.st_dev);
    id.file = static_cast<uint64_t>(st.st_ino);
  }
  return id;
}

inline FileIdentity IdentifyPath(const char *path) {
  FileIdentity id;
  struct stat st{};
  if (::stat(path, &st) == 0) {
    id.known = true;
    id.volume = static_cast<uint64_t>(st.st_dev);
    id.file = static_cast<uint64_t>(st.st_ino);
  }
  return id;
}

// See the Windows arm. mkstemp is what makes the name unique here: it creates
// the file itself, so the name it returns is one no other caller can be given.
// A suffix built from the process id and a counter would be simpler and would
// hold on Linux, macOS and Windows -- but not on the WebAssembly target, where
// getpid() is a fixed stub value and every process would build the same suffix.
//
// `model`, when not NULL, is an open handle on the file this one will replace.
// mkstemp opens 0600, so committing the staging file without copying the mode
// across would tighten the container's permissions behind the caller's back.
inline std::FILE *OpenUniqueTemp(const std::string &path, std::FILE *model, std::string *tmp_path) {
  const std::string tmpl = path + ".append.XXXXXX";
  std::vector<char> name(tmpl.begin(), tmpl.end());
  name.push_back('\0');
  const int fd = ::mkstemp(name.data());
  if (fd < 0) return nullptr;
  struct stat st{};
  if (model != nullptr && ::fstat(::fileno(model), &st) == 0) {
    ::fchmod(fd, st.st_mode & 07777);
  }
  std::FILE *fp = ::fdopen(fd, "wb");
  if (fp == nullptr) {
    ::close(fd);
    std::remove(name.data());
    return nullptr;
  }
  *tmp_path = name.data();
  return fp;
}

#endif

}  // namespace pvzstd::detail

#endif  // PVZSTD_FILEIO_H
