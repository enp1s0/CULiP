#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CULiP/cublas.hpp>
#include <cu_exp_statistics.hpp>
#include <cu_cutoff.hpp>
#include "utils.hpp"

#define CULIP_CUBLAS_LIBRARY_NAME       "libcublas.so"
#define CULIP_CUBLAS_ENV_NAME           "CULIP_CUBLAS_LIB_PATH"
#define CULIP_CUBLAS_DISABLE_ENV_NAME   "CULIP_DISABLE_CUBLAS_PROFILING"
#define CULIP_EXP_STATS_ENABLE_ENV_NAME "CULIP_ENABLE_EXP_STATS"
#define CULIP_CUTOFF_THRESHOLD_ENV_NAME "CULIP_CUTOFF_THRESHOLD"

namespace {
mtk::cu_exp_statistics::result_t exp_stats(
		const void* const ptr,
		const std::size_t offset,
		const std::size_t m,
		const std::size_t n,
		const std::size_t ld,
		const cudaStream_t cuda_stream,
		cudaDataType_t data_t
		) {
	mtk::cu_exp_statistics::result_t result;
	switch (data_t) {
	case CUDA_R_64F:
		result = mtk::cu_exp_statistics::take_matrix_statistics(reinterpret_cast<const double*>(ptr) + offset, m, n, ld, cuda_stream);
		break;
	case CUDA_R_32F:
		result = mtk::cu_exp_statistics::take_matrix_statistics(reinterpret_cast<const float*>(ptr) + offset, m, n, ld, cuda_stream);
		break;
	case CUDA_R_16F:
		result = mtk::cu_exp_statistics::take_matrix_statistics(reinterpret_cast<const half*>(ptr) + offset, m, n, ld, cuda_stream);
		break;
	case CUDA_C_64F:
		result = mtk::cu_exp_statistics::take_matrix_statistics(reinterpret_cast<const double2*>(ptr) + offset, m, n, ld, cuda_stream);
		break;
	case CUDA_C_32F:
		result = mtk::cu_exp_statistics::take_matrix_statistics(reinterpret_cast<const float2*>(ptr) + offset, m, n, ld, cuda_stream);
		break;
	case CUDA_C_16F:
		result = mtk::cu_exp_statistics::take_matrix_statistics(reinterpret_cast<const half2*>(ptr) + offset, m, n, ld, cuda_stream);
		break;
	default:
		break;
	}
	return result;
}

void cutoff(
		void* const ptr,
		const std::size_t offset,
		const std::size_t m,
		const std::size_t n,
		const std::size_t ld,
		double threshold,
		const cudaStream_t cuda_stream,
		cudaDataType_t data_t
		) {
	switch (data_t) {
	case CUDA_R_64F:
		mtk::cu_cutoff::cutoff_small_abs_values(reinterpret_cast<double*>(ptr) + offset, m, n, ld, threshold, cuda_stream);
		break;
	case CUDA_R_32F:
		mtk::cu_cutoff::cutoff_small_abs_values(reinterpret_cast<float*>(ptr) + offset, m, n, ld, threshold, cuda_stream);
		break;
	case CUDA_R_16F:
		mtk::cu_cutoff::cutoff_small_abs_values(reinterpret_cast<half*>(ptr) + offset, m, n, ld, threshold, cuda_stream);
		break;
	case CUDA_C_64F:
		mtk::cu_cutoff::cutoff_small_abs_values(reinterpret_cast<double2*>(ptr) + offset, m, n, ld, threshold, cuda_stream);
		break;
	case CUDA_C_32F:
		mtk::cu_cutoff::cutoff_small_abs_values(reinterpret_cast<float2*>(ptr) + offset, m, n, ld, threshold, cuda_stream);
		break;
	case CUDA_C_16F:
		mtk::cu_cutoff::cutoff_small_abs_values(reinterpret_cast<half2*>(ptr) + offset, m, n, ld, threshold, cuda_stream);
		break;
	default:
		break;
	}
}
} // unnamed namespace

extern "C" {
// dlopen cache
void* CULiP_cublas_lib_handle_cache = NULL;

// Control profiling
// 0         = Profiling
// Otherwise = Not profiling
int CULiP_profiling_control_array[CULiP_cublas_enum_length] = {0};

// Controler setter
void CULiP_enable_cublas_profiling(const CULiP_cublas_control_t target_function) {
	CULiP_profiling_control_array[target_function] = 0;
}
void CULiP_disable_cublas_profiling(const CULiP_cublas_control_t target_function) {
	CULiP_profiling_control_array[target_function] = 1;
}
void CULiP_enable_cublas_all_profiling() {
	for (unsigned target_function = 0; target_function < CULiP_cublas_enum_length; target_function++) {
		CULiP_profiling_control_array[target_function] = 0;
	}
}
void CULiP_disable_cublas_all_profiling() {
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

extern "C" const char* CULiP_get_cublasFillMode_t_string(const cublasFillMode_t mode) {
	switch(mode) {
	case CUBLAS_FILL_MODE_FULL:
		return "FULL";
	case CUBLAS_FILL_MODE_LOWER:
		return "LOWER";
	case CUBLAS_FILL_MODE_UPPER:
		return "UPPER";
	default:
		return "Unknown";
	}
}

extern "C" const char* CULiP_get_cublasSideMode_t_string(const cublasSideMode_t mode) {
	switch(mode) {
	case CUBLAS_SIDE_LEFT:
		return "LEFT";
	case CUBLAS_SIDE_RIGHT:
		return "RIGHT";
	default:
		return "Unknown";
	}
}

extern "C" const char* CULiP_get_cublasDiagType_t_string(const cublasDiagType_t type) {
	switch(type) {
	case CUBLAS_DIAG_NON_UNIT:
		return "NON_UNIT";
	case CUBLAS_DIAG_UNIT:
		return "UNIT";
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
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
	const int cutoff_flag = (CULiP_profiling_control_array[CULiP_cublasGemmEx] == 0) && CULiP_is_profiling_enabled(CULIP_CUTOFF_THRESHOLD_ENV_NAME, false);
	if (cutoff_flag) {
		double threshold;
		try {
			const auto env_str = getenv(CULIP_CUTOFF_THRESHOLD_ENV_NAME);
			threshold	= std::stod(env_str);

			cudaStream_t cuda_stream;
			cublasGetStream(handle, &cuda_stream);
			cutoff(const_cast<void*>(A), 0, (transa == CUBLAS_OP_N ? m : k), (transa == CUBLAS_OP_N ? k : m), lda, threshold, cuda_stream, Atype);
			cutoff(const_cast<void*>(B), 0, (transb == CUBLAS_OP_N ? k : n), (transb == CUBLAS_OP_N ? n : k), ldb, threshold, cuda_stream, Btype);
		} catch(const std::exception& e) {
			CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Warning] invalid threshold (%s)\n", env_str));
		}
	}

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
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-%s-%s-%s-%s-m%d-n%d-k%d", __func__,
				CULiP_get_cublasOperation_t_string(transa), CULiP_get_cublasOperation_t_string(transb),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Atype),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Btype),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Ctype),
				CULiP_get_cublasComputeType_t_string(computeType), m, n , k);

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

	const int exp_stats_flag = (CULiP_profiling_control_array[CULiP_cublasGemmEx] == 0) && CULiP_is_profiling_enabled(CULIP_EXP_STATS_ENABLE_ENV_NAME, false);
	if (exp_stats_flag) {
		cudaStream_t cuda_stream;
		cublasGetStream(handle, &cuda_stream);
		CULiP_exp_stats a_stats;
		CULiP_exp_stats b_stats;
		snprintf(a_stats.name, a_stats.name_length - 1, "A");
		snprintf(b_stats.name, b_stats.name_length - 1, "B");
		a_stats.stats = exp_stats(A, 0, (transa == CUBLAS_OP_N ? m : k), (transa == CUBLAS_OP_N ? k : m), lda, cuda_stream, Atype);
		b_stats.stats = exp_stats(B, 0, (transb == CUBLAS_OP_N ? k : n), (transb == CUBLAS_OP_N ? n : k), ldb, cuda_stream, Btype);
		mtk::cu_exp_statistics::to_json(a_stats.stats);
		mtk::cu_exp_statistics::to_json(b_stats.stats);
		CULiP_launch_function(cuda_stream, &CULiP_print_exp_stats_result, (void*)&a_stats);
		CULiP_launch_function(cuda_stream, &CULiP_print_exp_stats_result, (void*)&b_stats);
	}

	return result;
#endif
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
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
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
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-%s-%s-%s-%s-m%d-n%d-k%d-batchCount%d", __func__,
				CULiP_get_cublasOperation_t_string(transa), CULiP_get_cublasOperation_t_string(transb),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Atype),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Btype),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Ctype),
				CULiP_get_cublasComputeType_t_string(computeType), m, n , k, batchCount);

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
#endif
}

// -------------------------------------------------
// GEMM_STRIDED_BATCHED
// -------------------------------------------------

// SGEMM
#define CULIP_FUNC_NAME cublasSgemmStridedBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSgemmStridedBatched
#define CULIP_TYPE float
#include "cublas.gemm_strided_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// DGEMM
#define CULIP_FUNC_NAME cublasDgemmStridedBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDgemmStridedBatched
#define CULIP_TYPE double
#include "cublas.gemm_strided_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// HGEMM
#define CULIP_FUNC_NAME cublasHgemmStridedBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasHgemmStridedBatched
#define CULIP_TYPE half
#include "cublas.gemm_strided_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// CGEMM
#define CULIP_FUNC_NAME cublasCgemmStridedBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgemmStridedBatched
#define CULIP_TYPE cuComplex
#include "cublas.gemm_strided_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// ZGEMM
#define CULIP_FUNC_NAME cublasZgemmStridedBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgemmBatched
#define CULIP_TYPE cuDoubleComplex
#include "cublas.gemm_strided_batched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const void *alpha,
		const void *A,
		cudaDataType_t Atype,
		int lda,
		long long int strideA,
		const void *const B,
		cudaDataType_t Btype,
		int ldb,
		long long int strideB,
		const void *beta,
		void *const C,
		cudaDataType_t Ctype,
		int ldc,
		long long int strideC,
		int batchCount,
		cublasComputeType_t computeType,
		cublasGemmAlgo_t algo) {
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
	const int cutoff_flag = (CULiP_profiling_control_array[CULiP_cublasGemmStridedBatchedEx] == 0) && CULiP_is_profiling_enabled(CULIP_CUTOFF_THRESHOLD_ENV_NAME, false);
	if (cutoff_flag) {
		double threshold;
		try {
			const auto env_str = getenv(CULIP_CUTOFF_THRESHOLD_ENV_NAME);
			threshold	= std::stod(env_str);

			cudaStream_t cuda_stream;
			cublasGetStream(handle, &cuda_stream);
			for (std::uint32_t i = 0; i < batchCount; i++) {
				cutoff(const_cast<void*>(A), i * strideA, (transa == CUBLAS_OP_N ? m : k), (transa == CUBLAS_OP_N ? k : m), lda, threshold, cuda_stream, Atype);
				cutoff(const_cast<void*>(B), i * strideB, (transb == CUBLAS_OP_N ? k : n), (transb == CUBLAS_OP_N ? n : k), ldb, threshold, cuda_stream, Btype);
			}
		} catch(const std::exception& e) {
			CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Warning] invalid threshold (%s)\n", env_str));
		}
	}

	const int profiling_flag = (CULiP_profiling_control_array[CULiP_cublasGemmStridedBatchedEx] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, long long int, const void*, cudaDataType_t, int, long long int, const void*, void*, cudaDataType_t, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-%s-%s-%s-%s-m%d-n%d-k%d-batchCount%d", __func__,
				CULiP_get_cublasOperation_t_string(transa), CULiP_get_cublasOperation_t_string(transb),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Atype),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Btype),
				CULiP_get_cublasComputeType_t_string((cublasComputeType_t)Ctype),
				CULiP_get_cublasComputeType_t_string(computeType), m, n , k, batchCount);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	const int exp_stats_flag = (CULiP_profiling_control_array[CULiP_cublasGemmStridedBatchedEx] == 0) && CULiP_is_profiling_enabled(CULIP_EXP_STATS_ENABLE_ENV_NAME, false);
	if (exp_stats_flag) {
		cudaStream_t cuda_stream;
		cublasGetStream(handle, &cuda_stream);
		CULiP_exp_stats a_stats;
		CULiP_exp_stats b_stats;
		snprintf(a_stats.name, a_stats.name_length - 1, "A");
		snprintf(b_stats.name, b_stats.name_length - 1, "B");
		for (std::uint32_t i = 0; i < batchCount; i++) {
			a_stats.stats += exp_stats(A, i * strideA, (transa == CUBLAS_OP_N ? m : k), (transa == CUBLAS_OP_N ? k : m), lda, cuda_stream, Atype);
			b_stats.stats += exp_stats(B, i * strideB, (transb == CUBLAS_OP_N ? k : n), (transb == CUBLAS_OP_N ? n : k), ldb, cuda_stream, Btype);
		}
		mtk::cu_exp_statistics::to_json(a_stats.stats);
		mtk::cu_exp_statistics::to_json(b_stats.stats);
		CULiP_launch_function(cuda_stream, &CULiP_print_exp_stats_result, (void*)&a_stats);
		CULiP_launch_function(cuda_stream, &CULiP_print_exp_stats_result, (void*)&b_stats);
	}

	return result;
#endif
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

// -------------------------------------------------
// GBMV
// -------------------------------------------------

// SGBMV
#define CULIP_FUNC_NAME cublasSgbmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSgbmv
#define CULIP_TYPE float
#include "cublas.gbmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// DGBMV
#define CULIP_FUNC_NAME cublasDgbmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDgbmv
#define CULIP_TYPE double
#include "cublas.gbmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// CGBMV
#define CULIP_FUNC_NAME cublasCgbmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgbmv
#define CULIP_TYPE cuComplex
#include "cublas.gbmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// ZGBMV
#define CULIP_FUNC_NAME cublasZgbmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgbmv
#define CULIP_TYPE cuDoubleComplex
#include "cublas.gbmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// GER
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSger
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSger
#define CULIP_TYPE float
#include "cublas.ger.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDger
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDger
#define CULIP_TYPE double
#include "cublas.ger.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCgerc
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgerc
#define CULIP_TYPE cuComplex
#include "cublas.ger.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCgeru
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgeru
#define CULIP_TYPE cuComplex
#include "cublas.ger.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZgerc
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgerc
#define CULIP_TYPE cuDoubleComplex
#include "cublas.ger.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZgeru
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgeru
#define CULIP_TYPE cuDoubleComplex
#include "cublas.ger.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// SBMV
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSsbmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSsbmv
#define CULIP_TYPE float
#include "cublas.sbmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDsbmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDsbmv
#define CULIP_TYPE double
#include "cublas.sbmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// SPMV
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSspmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSspmv
#define CULIP_TYPE float
#include "cublas.spmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDspmv
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDspmv
#define CULIP_TYPE double
#include "cublas.spmv.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// SYRK
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSsyrk
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSsyrk
#define CULIP_TYPE float
#include "cublas.syrk.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDsyrk
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDsyrk
#define CULIP_TYPE double
#include "cublas.syrk.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCsyrk
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCsyrk
#define CULIP_TYPE cuComplex
#include "cublas.syrk.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZsyrk
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZsyrk
#define CULIP_TYPE cuDoubleComplex
#include "cublas.syrk.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// SYR2K
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSsyr2k
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSsyr2k
#define CULIP_TYPE float
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDsyr2k
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDsyr2k
#define CULIP_TYPE double
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCsyr2k
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCsyr2k
#define CULIP_TYPE cuComplex
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZsyr2k
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZsyr2k
#define CULIP_TYPE cuDoubleComplex
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// SYRKX
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSsyrkx
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSsyrkx
#define CULIP_TYPE float
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDsyrkx
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDsyrkx
#define CULIP_TYPE double
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCsyrkx
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCsyrkx
#define CULIP_TYPE cuComplex
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZsyrkx
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZsyrkx
#define CULIP_TYPE cuDoubleComplex
#include "cublas.syr2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// GEMM3M
// -------------------------------------------------
// CGEMM
#define CULIP_FUNC_NAME cublasCgemm3m
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCgemm3m
#define CULIP_TYPE cuComplex
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// ZGEMM
#define CULIP_FUNC_NAME cublasZgemm3m
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZgemm3m
#define CULIP_TYPE cuDoubleComplex
#include "cublas.gemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// SYMM
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasSsymm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasSsymm
#define CULIP_TYPE float
#include "cublas.symm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDsymm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDsymm
#define CULIP_TYPE double
#include "cublas.symm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCsymm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCsymm
#define CULIP_TYPE cuComplex
#include "cublas.symm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZsymm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZsymm
#define CULIP_TYPE cuDoubleComplex
#include "cublas.symm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// TRMM
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasStrmm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasStrmm
#define CULIP_TYPE float
#include "cublas.trmm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDtrmm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDtrmm
#define CULIP_TYPE double
#include "cublas.trmm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCtrmm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCtrmm
#define CULIP_TYPE cuComplex
#include "cublas.trmm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZtrmm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZtrmm
#define CULIP_TYPE cuDoubleComplex
#include "cublas.trmm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// TRSM
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasStrsm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasStrsm
#define CULIP_TYPE float
#include "cublas.trsm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDtrsm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDtrsm
#define CULIP_TYPE double
#include "cublas.trsm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCtrsm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCtrsm
#define CULIP_TYPE cuComplex
#include "cublas.trsm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZtrsm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZtrsm
#define CULIP_TYPE cuDoubleComplex
#include "cublas.trsm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// TRSM Batched
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasStrsmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasStrsmBatched
#define CULIP_TYPE float
#include "cublas.trsmBatched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasDtrsmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasDtrsmBatched
#define CULIP_TYPE double
#include "cublas.trsmBatched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasCtrsmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCtrsmBatched
#define CULIP_TYPE cuComplex
#include "cublas.trsmBatched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZtrsmBatched
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZtrsmBatched
#define CULIP_TYPE cuDoubleComplex
#include "cublas.trsmBatched.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// HERK
// -------------------------------------------------
#define CULIP_FUNC_NAME cublasCherk
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCherk
#define CULIP_TYPE cuComplex
#define CULIP_REAL_TYPE float
#include "cublas.herk.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
#undef CULIP_REAL_TYPE

#define CULIP_FUNC_NAME cublasZherk
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZherk
#define CULIP_TYPE cuDoubleComplex
#define CULIP_REAL_TYPE double
#include "cublas.herk.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
#undef CULIP_REAL_TYPE

// -------------------------------------------------
// HEMM
// -------------------------------------------------

#define CULIP_FUNC_NAME cublasChemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasChemm
#define CULIP_TYPE cuComplex
#include "cublas.hemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

#define CULIP_FUNC_NAME cublasZhemm
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZhemm
#define CULIP_TYPE cuDoubleComplex
#include "cublas.hemm.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE

// -------------------------------------------------
// HER2K
// -------------------------------------------------
#define CULIP_FUNC_NAME cublasCher2k
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCher2k
#define CULIP_TYPE cuComplex
#define CULIP_REAL_TYPE float
#include "cublas.her2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
#undef CULIP_REAL_TYPE

#define CULIP_FUNC_NAME cublasZher2k
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZher2k
#define CULIP_TYPE cuDoubleComplex
#define CULIP_REAL_TYPE double
#include "cublas.her2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
#undef CULIP_REAL_TYPE

// -------------------------------------------------
// HERKX
// -------------------------------------------------
#define CULIP_FUNC_NAME cublasCherkx
#define CULIP_FUNC_ENUM_NAME CULiP_cublasCherkx
#define CULIP_TYPE cuComplex
#define CULIP_REAL_TYPE float
#include "cublas.her2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
#undef CULIP_REAL_TYPE

#define CULIP_FUNC_NAME cublasZherkx
#define CULIP_FUNC_ENUM_NAME CULiP_cublasZherkx
#define CULIP_TYPE cuDoubleComplex
#define CULIP_REAL_TYPE double
#include "cublas.her2k.template.h"
#undef CULIP_FUNC_NAME
#undef CULIP_FUNC_ENUM_NAME
#undef CULIP_TYPE
#undef CULIP_REAL_TYPE

} // extern "C"
