#ifndef __CUDALIBPROFILER_UTILS_HPP__
#define __CUDALIBPROFILER_UTILS_HPP__
#include <time.h>

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
#endif
