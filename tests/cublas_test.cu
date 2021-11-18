#include <iostream>
#include <cublas.h>
#include <cublas_v2.h>
#include <CULiP/cublas.hpp>

template <class T>
struct get_real_type {
	using type = void;
};
template <> struct get_real_type<cuComplex> {using type = float;};
template <> struct get_real_type<cuDoubleComplex> {using type = double;};

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
// GemmStridedBatched
// -------------
template <class T, class Op>
cublasStatus_t gemm_strided_batched(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const T *alpha, const T *A, int lda, long long int strideA,
                           const T *B, int ldb, long long int strideB, const T *beta, T *C,
                           int ldc, long long int strideC, int batchCount);
// -----------------------------------------------------
// op_gemm
// -----------------------------------------------------
#define GEMM_STRIDED_BATCHED_OP_GEMM(short_type, type)\
template <>\
cublasStatus_t gemm_strided_batched<type , op_gemm>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A, int lda, long long int strideA,\
                           const type *B, int ldb, long long int strideB, const type *beta, type *C,\
                           int ldc, long long int strideC, int batchCount) {\
	return cublas##short_type##gemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);\
}
GEMM_STRIDED_BATCHED_OP_GEMM(H, half);
GEMM_STRIDED_BATCHED_OP_GEMM(S, float);
GEMM_STRIDED_BATCHED_OP_GEMM(D, double);
GEMM_STRIDED_BATCHED_OP_GEMM(C, cuComplex);
GEMM_STRIDED_BATCHED_OP_GEMM(Z, cuDoubleComplex);
// -----------------------------------------------------
// op_gemmEx
// -----------------------------------------------------
#define GEMM_STRIDED_BATCHED_OP_GEMMEX(cuda_data_type, type)\
template <>\
cublasStatus_t gemm_strided_batched<type , op_gemmEx>(cublasHandle_t handle, cublasOperation_t transa,\
                           cublasOperation_t transb, int m, int n, int k,\
                           const type *alpha, const type *A, int lda, long long int strideA,\
                           const type *B, int ldb, long long int strideB, const type *beta, type *C,\
                           int ldc, long long int strideC, int batchCount) {\
	return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, reinterpret_cast<const void*>(A), cuda_data_type, lda, strideA, reinterpret_cast<const void*>(B), cuda_data_type, ldb, strideB, beta, reinterpret_cast<void*>(C), cuda_data_type, ldc, strideC, batchCount, cuda_data_type, CUBLAS_GEMM_DEFAULT);\
}
GEMM_STRIDED_BATCHED_OP_GEMMEX(CUDA_R_16F, half);
GEMM_STRIDED_BATCHED_OP_GEMMEX(CUDA_R_32F, float);
GEMM_STRIDED_BATCHED_OP_GEMMEX(CUDA_R_64F, double);
GEMM_STRIDED_BATCHED_OP_GEMMEX(CUDA_C_32F, cuComplex);
GEMM_STRIDED_BATCHED_OP_GEMMEX(CUDA_C_64F, cuDoubleComplex);

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
// Ger
// -------------
template <class T>
cublasStatus_t ger(cublasHandle_t handle,
                           int m, int n,
                           const T *alpha,
                           const T *x, int incx, const T *y,
                           int incy, T *A, int lda);
#define GEMM_OP_GER(short_type, type)\
template <>\
cublasStatus_t ger<type>(cublasHandle_t handle,\
                           int m, int n,\
                           const type *alpha, \
                           const type *x, int incx, const type *y,\
                           int incy, type* A, int lda) {\
	return cublas##short_type##ger(handle, m, n, alpha, x, incx, y, incy, A, lda);\
}
GEMM_OP_GER(S, float);
GEMM_OP_GER(D, double);

template <class T>
cublasStatus_t gerc(cublasHandle_t handle,
                           int m, int n,
                           const T *alpha,
                           const T *x, int incx, const T *y,
                           int incy, T *A, int lda);
#define GEMM_OP_GERC(short_type, type)\
template <>\
cublasStatus_t gerc<type>(cublasHandle_t handle,\
                           int m, int n,\
                           const type *alpha, \
                           const type *x, int incx, const type *y,\
                           int incy, type* A, int lda) {\
	return cublas##short_type##gerc(handle, m, n, alpha, x, incx, y, incy, A, lda);\
}
GEMM_OP_GERC(C, cuComplex);
GEMM_OP_GERC(Z, cuDoubleComplex);

template <class T>
cublasStatus_t geru(cublasHandle_t handle,
                           int m, int n,
                           const T *alpha,
                           const T *x, int incx, const T *y,
                           int incy, T *A, int lda);
#define GEMM_OP_GERU(short_type, type)\
template <>\
cublasStatus_t geru<type>(cublasHandle_t handle,\
                           int m, int n,\
                           const type *alpha, \
                           const type *x, int incx, const type *y,\
                           int incy, type* A, int lda) {\
	return cublas##short_type##geru(handle, m, n, alpha, x, incx, y, incy, A, lda);\
}
GEMM_OP_GERU(C, cuComplex);
GEMM_OP_GERU(Z, cuDoubleComplex);

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
// Syrkx
// -------------
template <class T>
cublasStatus_t syrkx(cublasHandle_t handle,
		                       cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const T *alpha,
													 const T *A, int lda,
													 const T *B, int ldb,
                           const T *beta , T *C, int ldc
                           );
#define GEMM_OP_SYRKX(short_type, type)\
template <>\
cublasStatus_t syrkx<type>(cublasHandle_t handle, cublasFillMode_t uplo,\
		                       cublasOperation_t trans,\
                           int n, int k,\
                           const type *alpha, \
													 const type *A, int lda,\
													 const type *B, int ldb,\
                           const type *beta, type *C, int ldc\
                           ) {\
	return cublas##short_type##syrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_SYRKX(S, float);
GEMM_OP_SYRKX(D, double);
GEMM_OP_SYRKX(C, cuComplex);
GEMM_OP_SYRKX(Z, cuDoubleComplex);

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
// Trmm
// -------------
template <class T>
cublasStatus_t trmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const T *alpha,
													 const T *A, int lda,
													 const T *B, int ldb,
                           T *C, int ldc
                           );
#define GEMM_OP_TRMM(short_type, type)\
template <>\
cublasStatus_t trmm<type>(cublasHandle_t handle, \
                           cublasSideMode_t side, cublasFillMode_t uplo, \
                           cublasOperation_t trans, cublasDiagType_t diag, \
                           int m, int n,\
                           const type *alpha, \
													 const type *A, int lda,\
													 const type *B, int ldb,\
                           type *C, int ldc\
                           ) {\
	return cublas##short_type##trmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);\
}
GEMM_OP_TRMM(S, float);
GEMM_OP_TRMM(D, double);
GEMM_OP_TRMM(C, cuComplex);
GEMM_OP_TRMM(Z, cuDoubleComplex);

// -------------
// Trsm
// -------------
template <class T>
cublasStatus_t trsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const T *alpha,
													 const T *A, int lda,
                           T *B, int ldb
                           );
#define GEMM_OP_TRSM(short_type, type)\
template <>\
cublasStatus_t trsm<type>(cublasHandle_t handle, \
                           cublasSideMode_t side, cublasFillMode_t uplo, \
                           cublasOperation_t trans, cublasDiagType_t diag, \
                           int m, int n,\
                           const type *alpha, \
													 const type *A, int lda,\
                           type *B, int ldb\
                           ) {\
	return cublas##short_type##trsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);\
}
GEMM_OP_TRSM(S, float);
GEMM_OP_TRSM(D, double);
GEMM_OP_TRSM(C, cuComplex);
GEMM_OP_TRSM(Z, cuDoubleComplex);

// -------------
// Trsm batched
// -------------
template <class T>
cublasStatus_t trsm_batched(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const T *alpha,
													 const T *const A[], int lda,
                           T * const B[], int ldb,
													 int batchCount
                           );
#define GEMM_OP_TRSM_BATCHED(short_type, type)\
template <>\
cublasStatus_t trsm_batched<type>(cublasHandle_t handle, \
                           cublasSideMode_t side, cublasFillMode_t uplo, \
                           cublasOperation_t trans, cublasDiagType_t diag, \
                           int m, int n,\
                           const type *alpha, \
													 const type * const A[], int lda,\
                           type * const B[], int ldb,\
													 int batchCount\
                           ) {\
	return cublas##short_type##trsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);\
}
GEMM_OP_TRSM_BATCHED(S, float);
GEMM_OP_TRSM_BATCHED(D, double);
GEMM_OP_TRSM_BATCHED(C, cuComplex);
GEMM_OP_TRSM_BATCHED(Z, cuDoubleComplex);

// -----------------------------------------------------
// hemm
// -----------------------------------------------------
template <class T>
cublasStatus_t hemm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           int m, int n,
                           const T *alpha, const T *A, int lda,
                           const T *B, int ldb, const T *beta, T *C,
                           int ldc);
#define GEMM_OP_HEMM(short_type, type)\
template <>\
cublasStatus_t hemm<type>(cublasHandle_t handle,\
                           cublasSideMode_t side, cublasFillMode_t uplo, \
                           int m, int n, \
                           const type *alpha, const type *A, int lda,\
                           const type *B, int ldb, const type *beta, type *C,\
                           int ldc) {\
	return cublas##short_type##hemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_HEMM(C, cuComplex);
GEMM_OP_HEMM(Z, cuDoubleComplex);

// -----------------------------------------------------
// herk
// -----------------------------------------------------
template <class T, class RT>
cublasStatus_t herk(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int m, int n,
                           const RT *alpha, const T *A, int lda,
                           const RT *beta, T *C,
                           int ldc);
#define GEMM_OP_HERK(short_type, type, real_type)\
template <>\
cublasStatus_t herk<type, real_type>(cublasHandle_t handle,\
                           cublasFillMode_t uplo, cublasOperation_t trans, \
                           int m, int n, \
                           const real_type *alpha, const type *A, int lda,\
                           const real_type *beta, type *C,\
                           int ldc) {\
	return cublas##short_type##herk(handle, uplo, trans, m, n, alpha, A, lda, beta, C, ldc);\
}
GEMM_OP_HERK(C, cuComplex, float);
GEMM_OP_HERK(Z, cuDoubleComplex, double);

// -----------------------------------------------------
// her2k
// -----------------------------------------------------
template <class T, class RT>
cublasStatus_t her2k(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int m, int n,
                           const T *alpha,
                           const T *A, int lda,
                           const T *B, int ldb,
                           const RT *beta, T *C,
                           int ldc);
#define GEMM_OP_HER2K(short_type, type, real_type)\
template <>\
cublasStatus_t her2k<type, real_type>(cublasHandle_t handle,\
                           cublasFillMode_t uplo, cublasOperation_t trans, \
                           int m, int n, \
                           const type *alpha,\
                           const type *A, int lda,\
                           const type *B, int ldb,\
                           const real_type *beta, type *C,\
                           int ldc) {\
	return cublas##short_type##her2k(handle, uplo, trans, m, n, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_HER2K(C, cuComplex, float);
GEMM_OP_HER2K(Z, cuDoubleComplex, double);

// -----------------------------------------------------
// herkx
// -----------------------------------------------------
template <class T, class RT>
cublasStatus_t herkx(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int m, int n,
                           const T *alpha,
                           const T *A, int lda,
                           const T *B, int ldb,
                           const RT *beta, T *C,
                           int ldc);
#define GEMM_OP_HERKX(short_type, type, real_type)\
template <>\
cublasStatus_t herkx<type, real_type>(cublasHandle_t handle,\
                           cublasFillMode_t uplo, cublasOperation_t trans, \
                           int m, int n, \
                           const type *alpha,\
                           const type *A, int lda,\
                           const type *B, int ldb,\
                           const real_type *beta, type *C,\
                           int ldc) {\
	return cublas##short_type##herkx(handle, uplo, trans, m, n, alpha, A, lda, B, ldb, beta, C, ldc);\
}
GEMM_OP_HERKX(C, cuComplex, float);
GEMM_OP_HERKX(Z, cuDoubleComplex, double);

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

template <class T, class Op>
void gemm_strided_batched_test() {
	const int n = 1lu << 7;
	const int batch_size = 1u << 10;
	const auto alpha = convert<T>(1);
	const auto beta  = convert<T>(0);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMallocHost(&mat_a, sizeof(T) * n * n * batch_size);
	cudaMallocHost(&mat_b, sizeof(T) * n * n * batch_size);
	cudaMallocHost(&mat_c, sizeof(T) * n * n * batch_size);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gemm_strided_batched<T, Op>(
			cublas_handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			n, n, n,
			&alpha,
			mat_a, n, n * n,
			mat_b, n, n * n,
			&beta,
			mat_c, n, n * n,
			batch_size
			);

	cublasDestroy(cublas_handle);
	cudaFreeHost(mat_a);
	cudaFreeHost(mat_b);
	cudaFreeHost(mat_c);
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
void ger_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);

	T* mat_a;
	T* vec_x;
	T* vec_y;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&vec_x, sizeof(T) * n);
	cudaMalloc(&vec_y, sizeof(T) * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	ger<T>(
			cublas_handle,
			n, n,
			&alpha,
			vec_x, 1,
			vec_y, 1,
			mat_a, n
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(vec_x);
	cudaFree(vec_y);
}

template <class T>
void gerc_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);

	T* mat_a;
	T* vec_x;
	T* vec_y;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&vec_x, sizeof(T) * n);
	cudaMalloc(&vec_y, sizeof(T) * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	gerc<T>(
			cublas_handle,
			n, n,
			&alpha,
			vec_x, 1,
			vec_y, 1,
			mat_a, n
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(vec_x);
	cudaFree(vec_y);
}

template <class T>
void geru_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);

	T* mat_a;
	T* vec_x;
	T* vec_y;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&vec_x, sizeof(T) * n);
	cudaMalloc(&vec_y, sizeof(T) * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	geru<T>(
			cublas_handle,
			n, n,
			&alpha,
			vec_x, 1,
			vec_y, 1,
			mat_a, n
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
void syrkx_test() {
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

	syrkx<T>(
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
void trmm_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	trmm<T>(
			cublas_handle,
			CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
			n, n,
			&alpha,
			mat_a, n,
			mat_b, n,
			mat_c, n
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(mat_b);
	cudaFree(mat_c);
}

template <class T>
void trsm_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);

	T* mat_a;
	T* mat_b;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	trsm<T>(
			cublas_handle,
			CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
			n, n,
			&alpha,
			mat_a, n,
			mat_b, n
			);

	cublasDestroy(cublas_handle);
	cudaFree(mat_a);
	cudaFree(mat_b);
}

template <class T>
void trsm_batched_test() {
	const std::size_t n = 1lu << 8;
	const std::size_t batch_size = 1lu << 5;
	const auto alpha = convert<T>(1);

	T** mat_a_array;
	T** mat_b_array;

	cudaMallocHost(&mat_a_array, sizeof(T*) * batch_size);
	cudaMallocHost(&mat_b_array, sizeof(T*) * batch_size);

	for (unsigned i = 0; i < batch_size; i++) {
		T* ptr;
		cudaMalloc(&ptr, sizeof(T) * n * n);
		mat_a_array[i] = ptr;
		cudaMalloc(&ptr, sizeof(T) * n * n);
		mat_b_array[i] = ptr;
	}

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	trsm_batched<T>(
			cublas_handle,
			CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
			n, n,
			&alpha,
			mat_a_array, n,
			mat_b_array, n,
			batch_size
			);

	cublasDestroy(cublas_handle);

	for (unsigned i = 0; i < batch_size; i++) {
		cudaFree(mat_a_array[i]);
		cudaFree(mat_b_array[i]);
	}
	cudaFreeHost(mat_a_array);
	cudaFreeHost(mat_b_array);
}

template <class T>
void hemm_test() {
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta = convert<T>(0);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	hemm<T>(
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
void herk_test() {
	using real_type = typename get_real_type<T>::type;
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<real_type>(1);
	const auto beta = convert<real_type>(0);

	T* mat_a;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	herk<T>(
			cublas_handle,
			CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
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
void her2k_test() {
	using real_type = typename get_real_type<T>::type;
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta = convert<real_type>(0);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	her2k<T>(
			cublas_handle,
			CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
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
void herkx_test() {
	using real_type = typename get_real_type<T>::type;
	const std::size_t n = 1lu << 10;
	const auto alpha = convert<T>(1);
	const auto beta = convert<real_type>(0);

	T* mat_a;
	T* mat_b;
	T* mat_c;

	cudaMalloc(&mat_a, sizeof(T) * n * n);
	cudaMalloc(&mat_b, sizeof(T) * n * n);
	cudaMalloc(&mat_c, sizeof(T) * n * n);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	herkx<T>(
			cublas_handle,
			CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
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
	gemv_test<double         >();
	gemv_test<float          >();
	gemv_test<cuComplex      >();
	gemv_test<cuDoubleComplex>();

	gbmv_test<double         >();
	gbmv_test<float          >();
	gbmv_test<cuComplex      >();
	gbmv_test<cuDoubleComplex>();

	ger_test<double         >();
	ger_test<float          >();
	gerc_test<cuComplex      >();
	gerc_test<cuDoubleComplex>();
	geru_test<cuComplex      >();
	geru_test<cuDoubleComplex>();

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

	gemm_strided_batched_test<double         , op_gemm  >();
	gemm_strided_batched_test<float          , op_gemm  >();
	gemm_strided_batched_test<half           , op_gemm  >();
	gemm_strided_batched_test<cuComplex      , op_gemm  >();
	gemm_strided_batched_test<cuDoubleComplex, op_gemm  >();
	gemm_strided_batched_test<double         , op_gemmEx>();
	gemm_strided_batched_test<float          , op_gemmEx>();
	gemm_strided_batched_test<half           , op_gemmEx>();
	gemm_strided_batched_test<cuComplex      , op_gemmEx>();
	gemm_strided_batched_test<cuDoubleComplex, op_gemmEx>();

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

	syrkx_test<double         >();
	syrkx_test<float          >();
	syrkx_test<cuComplex      >();
	syrkx_test<cuDoubleComplex>();

	trmm_test<double         >();
	trmm_test<float          >();
	trmm_test<cuComplex      >();
	trmm_test<cuDoubleComplex>();

	trsm_test<double         >();
	trsm_test<float          >();
	trsm_test<cuComplex      >();
	trsm_test<cuDoubleComplex>();

	trsm_batched_test<double         >();
	trsm_batched_test<float          >();
	trsm_batched_test<cuComplex      >();
	trsm_batched_test<cuDoubleComplex>();

	hemm_test<cuComplex      >();
	hemm_test<cuDoubleComplex>();

	herk_test<cuComplex      >();
	herk_test<cuDoubleComplex>();

	her2k_test<cuComplex      >();
	her2k_test<cuDoubleComplex>();

	herkx_test<cuComplex      >();
	herkx_test<cuDoubleComplex>();

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
