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
template <>
cublasStatus_t gemm<float , op_gemm>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template <>
cublasStatus_t gemm<double, op_gemm>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta, double *C,
                           int ldc) {
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template <>
cublasStatus_t gemm<half  , op_gemm>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const half *alpha, const half *A, int lda,
                           const half *B, int ldb, const half *beta, half *C,
                           int ldc) {
	return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template <>
cublasStatus_t gemm<cuComplex, op_gemm>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C,
                           int ldc) {
	return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
// -----------------------------------------------------
// op_gemmEx
// -----------------------------------------------------
template <>
cublasStatus_t gemm<float , op_gemmEx>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
	return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
template <>
cublasStatus_t gemm<double, op_gemmEx>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta, double *C,
                           int ldc) {
	return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, beta, C, CUDA_R_64F, ldc, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
}
template <>
cublasStatus_t gemm<half  , op_gemmEx>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const half *alpha, const half *A, int lda,
                           const half *B, int ldb, const half *beta, half *C,
                           int ldc) {
	return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, beta, C, CUDA_R_16F, ldc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
}

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
	gemm_test<double, op_gemm  >();
	gemm_test<float , op_gemm  >();
	gemm_test<half  , op_gemm  >();
	gemm_test<cuComplex, op_gemm  >();
	gemm_test<double, op_gemmEx>();
	gemm_test<float , op_gemmEx>();
	gemm_test<half  , op_gemmEx>();
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
