#include <stdio.h>
#include <cuda.h>
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
