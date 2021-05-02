#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>

void sgemm_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = 1.f;
	const auto beta  = 0.f;

	float* mat_a;
	float* mat_b;
	float* mat_c;

	cudaMalloc(&mat_a, sizeof(float) * n * n);
	cudaMalloc(&mat_b, sizeof(float) * n * n);
	cudaMalloc(&mat_c, sizeof(float) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	cublasSgemm(
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
	sgemm_test();
}
