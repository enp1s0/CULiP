#include "utils.hpp"

extern "C" void CULiP_record_timestamp(void* tm_timestamp) {
	struct timespec* tm_ptr = (struct timespec*)tm_timestamp;
	clock_gettime(CLOCK_MONOTONIC, tm_ptr);
}
