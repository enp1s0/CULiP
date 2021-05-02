#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>

template <class T>
cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const T *alpha, const T *A, int lda,
                           const T *B, int ldb, const T *beta, T *C,
                           int ldc);
template <>
cublasStatus_t gemm<float >(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template <>
cublasStatus_t gemm<double>(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta, double *C,
                           int ldc) {
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template <>
cublasStatus_t gemm<half  >(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const half *alpha, const half *A, int lda,
                           const half *B, int ldb, const half *beta, half *C,
                           int ldc) {
	return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <class T>
void gemm_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = static_cast<T>(1);
	const auto beta  = static_cast<T>(0);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gemm<T>(
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

int main(){
	gemm_test<double>();
	gemm_test<float >();
	gemm_test<half  >();
}
