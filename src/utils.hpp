#ifndef __CUDALIBPROFILER_UTILS_HPP__
#define __CUDALIBPROFILER_UTILS_HPP__
#include <time.h>

#ifdef CULIBPROFILER_ENABLE_DEBUG_PRINT
#define CULIBPROFILER_DEBUG_PRINT(f) (f)
#else
#define CULIBPROFILER_DEBUG_PRINT(f)
#endif

// Timestamp
extern "C" void CULiP_record_timestamp(void* tm_timestamp);

// Profile result
struct CULiP_profile_result {
	// name
	enum {function_name_length = 128};
	char function_name[function_name_length] = {0};

	// tm
	struct timespec start_timestamp;
	struct timespec end_timestamp;
};

extern "C" void CULiP_print_profile_result(void* profile_result_ptr);

// Call a given function on a given stream
extern "C" void CULiP_launch_function(cudaStream_t cuda_stream, void (*fn)(void*), void* const arg);

// Function loader
extern "C" void* CULiP_get_function_pointer(const char* const library_name, const char* const env_name, const char* const function_name, void** CULiP_haldle_cache);

// Profiling status
extern "C" int CULiP_is_profiling_enabled(const char* env_name);
#endif
