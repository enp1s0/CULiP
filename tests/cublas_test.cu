#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>
#include <CULiP/cublas.hpp>

struct op_gemm;
struct op_gemmEx;

template <class T, class Op>
cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const T *alpha, const T *A, int lda,
                           const T *B, int ldb, const T *beta, T *C,
                           int ldc);
// -----------------------------------------------------
// op_gemm
// -----------------------------------------------------
#define GEMM_OP_GEMM(short_type, type)\
template <>\
cublasStatus_t gemm<type , op_gemm>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A, int lda,\
                           const type *B, int ldb, const type *beta, type *C,\
                           int ldc) {\
	return cublas##short_type##gemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_GEMM(H, half);
GEMM_OP_GEMM(S, float);
GEMM_OP_GEMM(D, double);
GEMM_OP_GEMM(C, cuComplex);
GEMM_OP_GEMM(Z, cuDoubleComplex);

#define GEMM_OP_GEMM_EX(cuda_data_type, type)\
template <>\
cublasStatus_t gemm<type , op_gemmEx>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A, int lda,\
                           const type *B, int ldb, const type *beta, type *C,\
                           int ldc) {\
	return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);\
}
GEMM_OP_GEMM_EX(CUDA_R_16F, half);
GEMM_OP_GEMM_EX(CUDA_R_32F, float);
GEMM_OP_GEMM_EX(CUDA_R_64F, double);
GEMM_OP_GEMM_EX(CUDA_C_32F, cuComplex);
GEMM_OP_GEMM_EX(CUDA_C_64F, cuDoubleComplex);

// -------------
// GemmBatched
// -------------
template <class T, class Op>
cublasStatus_t gemm_batched(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const T *alpha, const T *A[], int lda,
                           const T *B[], int ldb, const T *beta, T *C[],
                           int ldc, int batchCount);
// -----------------------------------------------------
// op_gemm
// -----------------------------------------------------
#define GEMM_BATCHED_OP_GEMM(short_type, type)\
template <>\
cublasStatus_t gemm_batched<type , op_gemm>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A[], int lda,\
                           const type *B[], int ldb, const type *beta, type *C[],\
                           int ldc, int batchCount) {\
	return cublas##short_type##gemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);\
}
GEMM_BATCHED_OP_GEMM(H, half);
GEMM_BATCHED_OP_GEMM(S, float);
GEMM_BATCHED_OP_GEMM(D, double);
GEMM_BATCHED_OP_GEMM(C, cuComplex);
GEMM_BATCHED_OP_GEMM(Z, cuDoubleComplex);
// -----------------------------------------------------
// op_gemmEx
// -----------------------------------------------------
#define GEMM_BATCHED_OP_GEMMEX(cuda_data_type, type)\
template <>\
cublasStatus_t gemm_batched<type , op_gemmEx>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A[], int lda,\
                           const type *B[], int ldb, const type *beta, type *C[],\
                           int ldc, int batchCount) {\
	return cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, reinterpret_cast<const void**>(A), cuda_data_type, lda, reinterpret_cast<const void**>(B), cuda_data_type, ldb, beta, reinterpret_cast<void**>(C), cuda_data_type, ldc, batchCount, cuda_data_type, CUBLAS_GEMM_DEFAULT);\
}
GEMM_BATCHED_OP_GEMMEX(CUDA_R_16F, half);
GEMM_BATCHED_OP_GEMMEX(CUDA_R_32F, float);
GEMM_BATCHED_OP_GEMMEX(CUDA_R_64F, double);
GEMM_BATCHED_OP_GEMMEX(CUDA_C_32F, cuComplex);
GEMM_BATCHED_OP_GEMMEX(CUDA_C_64F, cuDoubleComplex);


template <class T>
T convert(const double a) {return static_cast<T>(a);}
template <> cuComplex       convert<cuComplex      >(const double a) {return make_float2(a, 0);}
template <> cuDoubleComplex convert<cuDoubleComplex>(const double a) {return make_double2(a, 0);}

template <class T, class Op>
void gemm_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta  = convert<T>(0);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gemm<T, Op>(
			cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			n, n, n,
			&alpha,
			mat_a, n,
			mat_b, n,
			&beta,
			mat_c, n
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(mat_b);
	cudaFree(mat_c);
}

void test_all() {
	gemm_test<double         , op_gemm  >();
	gemm_test<float          , op_gemm  >();
	gemm_test<half           , op_gemm  >();
	gemm_test<cuComplex      , op_gemm  >();
	gemm_test<cuDoubleComplex, op_gemm  >();
	gemm_test<double         , op_gemmEx>();
	gemm_test<float          , op_gemmEx>();
	gemm_test<half           , op_gemmEx>();
	gemm_test<cuComplex      , op_gemmEx>();
	gemm_test<cuDoubleComplex, op_gemmEx>();
}

int main(){
	std::printf("Without profiling\n");
	CULiP_profile_cublas_disable_all();
	test_all();
	std::printf("-------\n");

	std::printf("With profiling\n");
	CULiP_profile_cublas_enable_all();
	test_all();
	std::printf("-------\n");
}
