#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <dlfcn.h>
#include "utils.hpp"

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
	printf("[CULiP Result][%s] %luns\n", profile_result.function_name, elapsed_time_us);
}

// TODO: Make this function non-blocking using `cuLauchHostFunc`
extern "C" void CULiP_launch_function(cudaStream_t cuda_stream, void (*fn)(void*), void* const arg) {
	cudaStreamSynchronize(cuda_stream);
	fn(arg);
	cudaStreamSynchronize(cuda_stream);
}

// Function loader
void* CULiP_get_function_pointer(const char* const env_name, const char* const function_name, void** CULiP_haldle_cache) {
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] start\n", function_name));

	// Get the real library path
	const char* library_path = getenv(env_name);
	if (library_path == NULL) {
		fprintf(stderr, "[CULiP ERROR] CULIP_CUBLAS_LIB_PATH is not set\n");
		exit(1);
	}

	// Open the library
	if (*CULiP_haldle_cache == NULL) {
		*CULiP_haldle_cache = dlopen(library_path, RTLD_NOW);
		if (*CULiP_haldle_cache == NULL) {
			fprintf(stderr, "[CULiP ERROR] Failed to load the real library %s\n", library_path);
			exit(1);
		}
		CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] %s is loaded\n", function_name, library_path));
	}

	// Get function pointer
	void* function_ptr = dlsym(*CULiP_haldle_cache, function_name);
	if (function_ptr == NULL) {
		fprintf(stderr, "[CULiP ERROR] Failed to load the function %s\n", __func__);
		exit(1);
	}

	return function_ptr;
}
