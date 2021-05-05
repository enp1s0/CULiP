#include <cublas.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CULiP/cublas.hpp>
#include "utils.hpp"

#define CULIP_CUBLAS_LIBRARY_NAME       "libcublas.so"
#define CULIP_CUBLAS_ENV_NAME           "CULIP_CUBLAS_LIB_PATH"
#define CULIP_CUBLAS_DISABLE_ENV_NAME   "CULIP_PROFILING_CUBLAS_DISABLE"

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

// cudaDataType yo string
#define CULiP_CUBLAS_COMPUTE_T_CASE_STRING(compute_type) case compute_type: return #compute_type
extern "C" const char* CULiP_get_cublasComputeType_t_string(const cublasComputeType_t compute_type) {
	switch(compute_type) {
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_16F);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_16F_PEDANTIC);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_FAST_16BF);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_FAST_16F);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_FAST_TF32);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_PEDANTIC);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32I);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32I_PEDANTIC);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_64F);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_64F_PEDANTIC);
	default:
		break;
	}
	switch((cudaDataType_t)compute_type) {
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_16BF);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_16F );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_32F );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_32I );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_64F );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_8I  );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_8U  );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_16BF);
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_16F );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_32F );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_32I );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_64F );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_8I  );
		CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_8U  );
	default:
		return "Unknown";
	}
}

// -------------------------------------------------
// cuBLAS functions
// -------------------------------------------------

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasSgemm] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
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

	if (profiling_flag) {
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
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasDgemm] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
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

	if (profiling_flag) {
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
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasHgemm] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const half*, const half*, int, const half*, int, const half*, half*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
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

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}

cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C,
                           int ldc) {
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasCgemm] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
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

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}

cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C,
                           int ldc) {
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasZgemm] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
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

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A,
                            cudaDataType_t Atype, int lda, const void *B,
                            cudaDataType_t Btype, int ldb, const void *beta,
                            void *C, cudaDataType_t Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo) {
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasGemmEx] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s-m%d-n%d-k%d", __func__, CULiP_get_cublasComputeType_t_string(computeType), m, n , k);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}
} // extern "C"
