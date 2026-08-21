// Whether this build has a thread runtime, and how many workers it can use.
//
// PVZSTD_NO_THREADS (from the PVZSTD_THREADS CMake option) runs the work inline
// so a target without a thread runtime still builds; results are unchanged either
// way. Compile-time because <thread> is what pulls the runtime in.

#ifndef PVZSTD_THREADS_H
#define PVZSTD_THREADS_H

#include <cstdint>
#include <vector>

#ifndef PVZSTD_NO_THREADS
#include <thread>
#endif

namespace pvzstd::detail {

#ifdef PVZSTD_NO_THREADS
constexpr bool kHasThreads = false;
#else
constexpr bool kHasThreads = true;
#endif

// Workers this build can run in parallel; 1 without a thread runtime.
inline int HardwareWorkers() {
#ifdef PVZSTD_NO_THREADS
  return 1;
#else
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
#endif
}

// Call fn(i) for every i in [0, count), striding indices over `workers` threads.
template <typename Fn>
inline void ParallelStride(int workers, uint64_t count, Fn fn) {
#ifdef PVZSTD_NO_THREADS
  (void)workers;
  for (uint64_t i = 0; i < count; ++i) fn(i);
#else
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers));
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([w, workers, count, &fn]() {
      for (uint64_t i = static_cast<uint64_t>(w); i < count; i += static_cast<uint64_t>(workers)) {
        fn(i);
      }
    });
  }
  for (std::thread &t : pool) t.join();
#endif
}

}  // namespace pvzstd::detail

#endif  // PVZSTD_THREADS_H
