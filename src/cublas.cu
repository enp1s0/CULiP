#include <cublas.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <culibprofiler/cublas.hpp>

#ifdef CULIBPROFILER_ENABLE_DEBUG_PRINT
#define CULIBPROFILER_DEBUG_PRINT(f) (f)
#else
#define CULIBPROFILER_DEBUG_PRINT(f)
#endif

extern "C" {
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] start\n", __func__));

	// Get the real library path
	const char* cublas_lib_path = getenv("CLP_CUBLAS_LIB_PATH");
	if (cublas_lib_path == NULL) {
		fprintf(stderr, "[CULiP ERROR] CLP_CUBLAS_LIB_PATH is not set\n");
		exit(1);
	}
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] %s is loaded\n", __func__, cublas_lib_path));

	// Open the library
	void* cublas_lib_handle = dlopen(cublas_lib_path, RTLD_NOW);
	if (cublas_lib_handle == NULL) {
		fprintf(stderr, "[CULiP ERROR] Failed to load the real library %s\n", cublas_lib_path);
		exit(1);
	}

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
	*(void**)(&cublas_lib_func) = dlsym(cublas_lib_handle, __func__);
	if (cublas_lib_func == NULL) {
		fprintf(stderr, "[CULiP ERROR] Failed to load the function %s\n", __func__);
		exit(1);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	return result;
}
} // extern "C"
