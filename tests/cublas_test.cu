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

// -----------------------------------------------------
// op_gemmEx
// -----------------------------------------------------
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
                           const T *alpha, const T **A, int lda,
                           const T **B, int ldb, const T *beta, T **C,
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

// -------------
// Gemv
// -------------
template <class T>
cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const T *alpha, const T *A, int lda,
                           const T *x, int incx, const T *beta, T *y,
                           int incy);
#define GEMM_OP_GEMV(short_type, type)\
template <>\
cublasStatus_t gemv<type>(cublasHandle_t handle, cublasOperation_t trans,\
                           int m, int n,\
                           const type *alpha, const type *A, int lda,\
                           const type *x, int incx, const type *beta, type *y,\
                           int incy) {\
	return cublas##short_type##gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);\
}
GEMM_OP_GEMV(S, float);
GEMM_OP_GEMV(D, double);
GEMM_OP_GEMV(C, cuComplex);
GEMM_OP_GEMV(Z, cuDoubleComplex);

// -------------
// Gbmv
// -------------
template <class T>
cublasStatus_t gbmv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, int kl, int ku,
                           const T *alpha, const T *A, int lda,
                           const T *x, int incx, const T *beta, T *y,
                           int incy);
#define GEMM_OP_GBMV(short_type, type)\
template <>\
cublasStatus_t gbmv<type>(cublasHandle_t handle, cublasOperation_t trans,\
                           int m, int n, int kl, int ku,\
                           const type *alpha, const type *A, int lda,\
                           const type *x, int incx, const type *beta, type *y,\
                           int incy) {\
	return cublas##short_type##gbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);\
}
GEMM_OP_GBMV(S, float);
GEMM_OP_GBMV(D, double);
GEMM_OP_GBMV(C, cuComplex);
GEMM_OP_GBMV(Z, cuDoubleComplex);

// -------------
// Syrk
// -------------
template <class T>
cublasStatus_t syrk(cublasHandle_t handle,
		                       cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const T *alpha, const T *A, int lda,
                           const T *beta , T *C, int ldc
                           );
#define GEMM_OP_SYRK(short_type, type)\
template <>\
cublasStatus_t syrk<type>(cublasHandle_t handle, cublasFillMode_t uplo,\
		                       cublasOperation_t trans,\
                           int n, int k,\
                           const type *alpha, const type *A, int lda,\
                           const type *beta, type *C, int ldc\
                           ) {\
	return cublas##short_type##syrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);\
}
GEMM_OP_SYRK(S, float);
GEMM_OP_SYRK(D, double);
GEMM_OP_SYRK(C, cuComplex);
GEMM_OP_SYRK(Z, cuDoubleComplex);

// -------------
// Syr2k
// -------------
template <class T>
cublasStatus_t syr2k(cublasHandle_t handle,
		                       cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const T *alpha,
													 const T *A, int lda,
													 const T *B, int ldb,
                           const T *beta , T *C, int ldc
                           );
#define GEMM_OP_SYR2K(short_type, type)\
template <>\
cublasStatus_t syr2k<type>(cublasHandle_t handle, cublasFillMode_t uplo,\
		                       cublasOperation_t trans,\
                           int n, int k,\
                           const type *alpha, \
													 const type *A, int lda,\
													 const type *B, int ldb,\
                           const type *beta, type *C, int ldc\
                           ) {\
	return cublas##short_type##syr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_SYR2K(S, float);
GEMM_OP_SYR2K(D, double);
GEMM_OP_SYR2K(C, cuComplex);
GEMM_OP_SYR2K(Z, cuDoubleComplex);

// -------------
// Symm
// -------------
template <class T>
cublasStatus_t symm(cublasHandle_t handle, cublasSideMode_t size,
		                       cublasFillMode_t uplo,
                           int m, int n,
                           const T *alpha,
													 const T *A, int lda,
													 const T *B, int ldb,
                           const T *beta , T *C, int ldc
                           );
#define GEMM_OP_SYMM(short_type, type)\
template <>\
cublasStatus_t symm<type>(cublasHandle_t handle, cublasSideMode_t side,\
		                       cublasFillMode_t uplo,\
                           int m, int n,\
                           const type *alpha, \
													 const type *A, int lda,\
													 const type *B, int ldb,\
                           const type *beta, type *C, int ldc\
                           ) {\
	return cublas##short_type##symm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_SYMM(S, float);
GEMM_OP_SYMM(D, double);
GEMM_OP_SYMM(C, cuComplex);
GEMM_OP_SYMM(Z, cuDoubleComplex);

// -------------
// Gemm3m
// -------------
template <class T>
cublasStatus_t gemm3m(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const T *alpha, const T *A, int lda,
                           const T *B, int ldb, const T *beta, T *C,
                           int ldc);
// -----------------------------------------------------
// op_gemm
// -----------------------------------------------------
#define GEMM_OP_GEMM3M(short_type, type)\
template <>\
cublasStatus_t gemm3m<type>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A, int lda,\
                           const type *B, int ldb, const type *beta, type *C,\
                           int ldc) {\
	return cublas##short_type##gemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_GEMM3M(C, cuComplex);
GEMM_OP_GEMM3M(Z, cuDoubleComplex);


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

template <class T, class Op>
void gemm_batched_test() {
	const int n = 1lu << 7;
	const int batch_size = 1u << 10;
	const auto alpha = convert<T>(1);
	const auto beta  = convert<T>(0);

	T** mat_a_array;
	T** mat_b_array;
	T** mat_c_array;

	cudaMallocHost(&mat_a_array, sizeof(T*) * batch_size);
	cudaMallocHost(&mat_b_array, sizeof(T*) * batch_size);
	cudaMallocHost(&mat_c_array, sizeof(T*) * batch_size);

	for (unsigned i = 0; i < batch_size; i++) {
		T* ptr;
		cudaMalloc(&ptr, sizeof(T) * n * n);
		mat_a_array[i] = ptr;
		cudaMalloc(&ptr, sizeof(T) * n * n);
		mat_b_array[i] = ptr;
		cudaMalloc(&ptr, sizeof(T) * n * n);
		mat_c_array[i] = ptr;
	}

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gemm_batched<T, Op>(
			cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			n, n, n,
			&alpha,
			(const T**)mat_a_array, n,
			(const T**)mat_b_array, n,
			&beta,
			mat_c_array, n,
			batch_size
			);

	cublasDestroy(cublas_handle);

	for (unsigned i = 0; i < batch_size; i++) {
		cudaFree(mat_a_array[i]);
		cudaFree(mat_b_array[i]);
		cudaFree(mat_c_array[i]);
	}
	cudaFreeHost(mat_a_array);
	cudaFreeHost(mat_b_array);
	cudaFreeHost(mat_c_array);
}

template <class T>
void gemv_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta  = convert<T>(0);

	T* mat_a;
	T* vec_x;
	T* vec_y;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&vec_x, sizeof(T) * n);
	cudaMalloc(&vec_y, sizeof(T) * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gemv<T>(
			cublas_handle,
			CUBLAS_OP_N,
			n, n,
			&alpha,
			mat_a, n,
			vec_x, 1,
			&beta,
			vec_y, 1
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(vec_x);
	cudaFree(vec_y);
}

template <class T>
void gbmv_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta  = convert<T>(0);

	T* mat_a;
	T* vec_x;
	T* vec_y;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&vec_x, sizeof(T) * n);
	cudaMalloc(&vec_y, sizeof(T) * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gbmv<T>(
			cublas_handle,
			CUBLAS_OP_N,
			n, n, n / 10, n / 10,
			&alpha,
			mat_a, n,
			vec_x, 1,
			&beta,
			vec_y, 1
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(vec_x);
	cudaFree(vec_y);
}

template <class T>
void syrk_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta  = convert<T>(0);

	T* mat_a;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	syrk<T>(
			cublas_handle,
			CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N,
			n, n,
			&alpha,
			mat_a, n,
			&beta,
			mat_c, n
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(mat_c);
}

template <class T>
void syr2k_test() {
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

	syr2k<T>(
			cublas_handle,
			CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N,
			n, n,
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

template <class T>
void symm_test() {
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

	symm<T>(
			cublas_handle,
			CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
			n, n,
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

template <class T>
void gemm3m_test() {
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

	gemm3m<T>(
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

	gemm_batched_test<double         , op_gemm  >();
	gemm_batched_test<float          , op_gemm  >();
	gemm_batched_test<half           , op_gemm  >();
	gemm_batched_test<cuComplex      , op_gemm  >();
	gemm_batched_test<cuDoubleComplex, op_gemm  >();
	gemm_batched_test<double         , op_gemmEx>();
	gemm_batched_test<float          , op_gemmEx>();
	gemm_batched_test<half           , op_gemmEx>();
	gemm_batched_test<cuComplex      , op_gemmEx>();
	gemm_batched_test<cuDoubleComplex, op_gemmEx>();

	gemv_test<double         >();
	gemv_test<float          >();
	gemv_test<cuComplex      >();
	gemv_test<cuDoubleComplex>();

	gbmv_test<double         >();
	gbmv_test<float          >();
	gbmv_test<cuComplex      >();
	gbmv_test<cuDoubleComplex>();

	syrk_test<double         >();
	syrk_test<float          >();
	syrk_test<cuComplex      >();
	syrk_test<cuDoubleComplex>();

	symm_test<double         >();
	symm_test<float          >();
	symm_test<cuComplex      >();
	symm_test<cuDoubleComplex>();

	syr2k_test<double         >();
	syr2k_test<float          >();
	syr2k_test<cuComplex      >();
	syr2k_test<cuDoubleComplex>();

	gemm3m_test<cuComplex      >();
	gemm3m_test<cuDoubleComplex>();
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
