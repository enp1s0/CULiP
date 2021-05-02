#include <cublas.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <iostream>
#include <culibprofiler/cublas.hpp>

extern "C" {
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
	return CUBLAS_STATUS_SUCCESS;
}
}
