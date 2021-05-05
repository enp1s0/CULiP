#ifndef __CULIP_CUBLAS_HPP__
#define __CULIP_CUBLAS_HPP__

enum CULiP_cublas_control_t {
	CULiP_cublasDgemm = 0,
	CULiP_cublasSgemm = 1,
	CULiP_cublasHgemm = 2,
	CULiP_cublasCgemm = 3,
	CULiP_cublasZgemm = 4,
	CULiP_cublasGemmEx = 5,
	CULiP_cublas_enum_length
};

extern "C" void CULiP_profile_cublas_enable(const CULiP_cublas_control_t target_function);
extern "C" void CULiP_profile_cublas_disable(const CULiP_cublas_control_t target_function);
extern "C" void CULiP_profile_cublas_enable_all();
extern "C" void CULiP_profile_cublas_disable_all();
#endif
