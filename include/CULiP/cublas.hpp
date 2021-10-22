#ifndef __CULIP_CUBLAS_HPP__
#define __CULIP_CUBLAS_HPP__

enum CULiP_cublas_control_t {
	CULiP_cublasDgemm,
	CULiP_cublasSgemm,
	CULiP_cublasHgemm,
	CULiP_cublasCgemm,
	CULiP_cublasZgemm,
	CULiP_cublasGemmEx,
	CULiP_cublasCgemm3m,
	CULiP_cublasZgemm3m,
	CULiP_cublasDgemmBatched,
	CULiP_cublasSgemmBatched,
	CULiP_cublasHgemmBatched,
	CULiP_cublasCgemmBatched,
	CULiP_cublasZgemmBatched,
	CULiP_cublasGemmBatchedEx,
	CULiP_cublasDgemv,
	CULiP_cublasSgemv,
	CULiP_cublasCgemv,
	CULiP_cublasZgemv,
	CULiP_cublasDgbmv,
	CULiP_cublasSgbmv,
	CULiP_cublasCgbmv,
	CULiP_cublasZgbmv,
	CULiP_cublasDsyrk,
	CULiP_cublasSsyrk,
	CULiP_cublasCsyrk,
	CULiP_cublasZsyrk,
	CULiP_cublasDsymm,
	CULiP_cublasSsymm,
	CULiP_cublasCsymm,
	CULiP_cublasZsymm,
	CULiP_cublasDsyr2k,
	CULiP_cublasSsyr2k,
	CULiP_cublasCsyr2k,
	CULiP_cublasZsyr2k,
	CULiP_cublasDsyrkx,
	CULiP_cublasSsyrkx,
	CULiP_cublasCsyrkx,
	CULiP_cublasZsyrkx,
	CULiP_cublasDtrmm,
	CULiP_cublasStrmm,
	CULiP_cublasCtrmm,
	CULiP_cublasZtrmm,
	CULiP_cublasDtrsm,
	CULiP_cublasStrsm,
	CULiP_cublasCtrsm,
	CULiP_cublasZtrsm,
	CULiP_cublas_enum_length
};

extern "C" void CULiP_profile_cublas_enable(const CULiP_cublas_control_t target_function);
extern "C" void CULiP_profile_cublas_disable(const CULiP_cublas_control_t target_function);
extern "C" void CULiP_profile_cublas_enable_all();
extern "C" void CULiP_profile_cublas_disable_all();
#endif
