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

extern "C" const char* CULiP_get_cublasOperation_t_string(const cublasOperation_t op) {
	switch(op) {
	case CUBLAS_OP_N:
		return "N";
	case CUBLAS_OP_T:
		return "T";
	case CUBLAS_OP_C:
		return "C";
	default:
		return "Unknown";
	}
}

// -------------------------------------------------
// GEMM
// -------------------------------------------------

// SGEMM
#define CULIP_FUNC_NAME cublasSgemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSgemm
#define CULIP_TYPE float
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// DGEMM
#define CULIP_FUNC_NAME cublasDgemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDgemm
#define CULIP_TYPE double
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// HGEMM
#define CULIP_FUNC_NAME cublasHgemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasHgemm
#define CULIP_TYPE half
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// CGEMM
#define CULIP_FUNC_NAME cublasCgemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgemm
#define CULIP_TYPE cuComplex
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// ZGEMM
#define CULIP_FUNC_NAME cublasZgemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgemm
#define CULIP_TYPE cuDoubleComplex
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

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
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-%s-m%d-n%d-k%d", __func__, CULiP_get_cublasOperation_t_string(transa), CULiP_get_cublasOperation_t_string(transb), CULiP_get_cublasComputeType_t_string(computeType), m, n , k);

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

// -------------------------------------------------
// GEMM_BATCHED
// -------------------------------------------------

// SGEMM
#define CULIP_FUNC_NAME cublasSgemmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSgemmBatched
#define CULIP_TYPE float
#include "cublas.gemm_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// DGEMM
#define CULIP_FUNC_NAME cublasDgemmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDgemmBatched
#define CULIP_TYPE double
#include "cublas.gemm_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// HGEMM
#define CULIP_FUNC_NAME cublasHgemmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasHgemmBatched
#define CULIP_TYPE half
#include "cublas.gemm_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// CGEMM
#define CULIP_FUNC_NAME cublasCgemmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgemmBatched
#define CULIP_TYPE cuComplex
#include "cublas.gemm_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// ZGEMM
#define CULIP_FUNC_NAME cublasZgemmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgemmBatched
#define CULIP_TYPE cuDoubleComplex
#include "cublas.gemm_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const void *alpha,
		const void *const Aarray[],
		cudaDataType_t Atype,
		int lda,
		const void *const Barray[],
		cudaDataType_t Btype,
		int ldb,
		const void *beta,
		void *const Carray[],
		cudaDataType_t Ctype,
		int ldc,
		int batchCount,
		cublasComputeType_t computeType,
		cublasGemmAlgo_t algo) {
	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasGemmBatchedEx] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void* const[], cudaDataType_t, int, const void* const[], cudaDataType_t, int, const void*, void* const[], cudaDataType_t, int, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-%s-m%d-n%d-k%d-batchCount%d", __func__, CULiP_get_cublasOperation_t_string(transa), CULiP_get_cublasOperation_t_string(transb), CULiP_get_cublasComputeType_t_string(computeType), m, n , k, batchCount);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}

// -------------------------------------------------
// GEMV
// -------------------------------------------------

// SGEMV
#define CULIP_FUNC_NAME cublasSgemv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSgemv
#define CULIP_TYPE float
#include "cublas.gemv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// DGEMV
#define CULIP_FUNC_NAME cublasDgemv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDgemv
#define CULIP_TYPE double
#include "cublas.gemv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// CGEMV
#define CULIP_FUNC_NAME cublasCgemv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgemv
#define CULIP_TYPE cuComplex
#include "cublas.gemv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// ZGEMV
#define CULIP_FUNC_NAME cublasZgemv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgemv
#define CULIP_TYPE cuDoubleComplex
#include "cublas.gemv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
} // extern "C"
