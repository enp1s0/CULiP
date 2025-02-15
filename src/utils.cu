#include "utils.hpp"
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern "C" void CULiP_record_timestamp(void *tm_timestamp) {
  struct timespec *tm_ptr = (struct timespec *)tm_timestamp;
  clock_gettime(CLOCK_MONOTONIC, tm_ptr);
}

extern "C" void CULiP_print_profile_result(void *profile_result_ptr) {
  const CULiP_profile_result profile_result =
      *((CULiP_profile_result *)profile_result_ptr);

  const unsigned long elapsed_time_us =
      ((long)profile_result.end_timestamp.tv_sec -
       (long)profile_result.start_timestamp.tv_sec) *
          (long)1000000000 +
      ((long)profile_result.end_timestamp.tv_nsec -
       (long)profile_result.start_timestamp.tv_nsec);
  printf("[%s][%s] %luns\n", CULIP_RESULT_PREFIX, profile_result.function_name,
         elapsed_time_us);
  fflush(stdout);
}

extern "C" void CULiP_print_exp_stats_result(void *exp_stats_result_ptr) {
  const CULiP_exp_stats exp_stats_result =
      *((CULiP_exp_stats *)exp_stats_result_ptr);

  printf("[%s] %s: %s\n", CULIP_EXP_STATS_PREFIX, exp_stats_result.name,
         mtk::cu_exp_statistics::to_json(exp_stats_result.stats).c_str());
  fflush(stdout);
}

// TODO: Make this function non-blocking using `cuLauchHostFunc`
extern "C" void CULiP_launch_function(cudaStream_t cuda_stream,
                                      void (*fn)(void *), void *const arg) {
  cudaStreamSynchronize(cuda_stream);
  fn(arg);
  cudaStreamSynchronize(cuda_stream);
}

// Function loader
extern "C" void *CULiP_get_function_pointer(const char *const,
                                            const char *const,
                                            const char *const function_name,
                                            void **CULiP_handle_cache) {
  CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] start\n", function_name));

  // Get function pointer
  void *function_ptr = dlsym(RTLD_NEXT, function_name);
  if (function_ptr == NULL) {
    fprintf(stderr, "[CULiP ERROR] Failed to load the function %s\n",
            function_name);
    exit(1);
  }

  return function_ptr;
}

// Profiling status
extern "C" int CULiP_is_profiling_enabled(const char *env_name,
                                          const bool disable_if_set) {
  const char *value = getenv(env_name);
  if (value == NULL) {
    return disable_if_set;
  }
  if (strcmp(value, "0") == 0) {
    return disable_if_set;
  }
  return !disable_if_set;
}
