#include <cublas.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CULiP/cublas.hpp>
#include "utils.hpp"

extern "C" {
// dlopen cache
void* CULiP_cublas_lib_handle_cache = NULL;

// Control profiling
// 0         = Profiling
// Otherwise = Not profiling
int CULiP_profiling_control_array[CULiP_cublas_enum_length] = {0};

// Controler setter
void CULiP_profile_cublas_enable(const CULiP_cublas_control_t target_function) {
	CULiP_profiling_control_array[target_function] = 0;
}
void CULiP_profile_cublas_disable(const CULiP_cublas_control_t target_function) {
	CULiP_profiling_control_array[target_function] = 1;
}
void CULiP_profile_cublas_enable_all() {
	for (unsigned target_function = 0; target_function < CULiP_cublas_enum_length; target_function++) {
		CULiP_profiling_control_array[target_function] = 0;
	}
}
void CULiP_profile_cublas_disable_all() {
	for (unsigned target_function = 0; target_function < CULiP_cublas_enum_length; target_function++) {
		CULiP_profiling_control_array[target_function] = 1;
	}
}

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer("CULIP_CUBLAS_LIB_PATH", __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (CULiP_profiling_control_array[CULiP_cublasSgemm] == 0) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-m%d-n%d-k%d", __func__, m, n ,k);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (CULiP_profiling_control_array[CULiP_cublasSgemm] == 0) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta, double *C,
                           int ldc) {

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer("CULIP_CUBLAS_LIB_PATH", __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (CULiP_profiling_control_array[CULiP_cublasDgemm] == 0) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-m%d-n%d-k%d", __func__, m, n ,k);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (CULiP_profiling_control_array[CULiP_cublasDgemm] == 0) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const half *alpha, const half *A, int lda,
                           const half *B, int ldb, const half *beta, half *C,
                           int ldc) {

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const half*, const half*, int, const half*, int, const half*, half*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer("CULIP_CUBLAS_LIB_PATH", __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (CULiP_profiling_control_array[CULiP_cublasHgemm] == 0) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-m%d-n%d-k%d", __func__, m, n ,k);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (CULiP_profiling_control_array[CULiP_cublasHgemm] == 0) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}
} // extern "C"
