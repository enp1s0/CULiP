cublasStatus_t CULIP_FUNC_NAME(cublasHandle_t handle,
                           cublasFillMode_t uplo, int n, int k,
                           const CULIP_TYPE *alpha, const CULIP_TYPE *A, int lda,
                           const CULIP_TYPE *x, int incx, const CULIP_TYPE *beta, CULIP_TYPE *y,
                           int incy) {
	const int profiling_flag = (CULiP_profiling_control_array[CULIP_FUNC_ENUM_NAME] == 0) && CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

	// Get the function pointer
	cublasStatus_t (*cublas_lib_func)(cublasHandle_t, cublasFillMode_t, int, int, const CULIP_TYPE*, const CULIP_TYPE*, int, const CULIP_TYPE*, int, const CULIP_TYPE*, CULIP_TYPE*, int);
	*(void**)(&cublas_lib_func) = CULiP_get_function_pointer(CULIP_CUBLAS_LIBRARY_NAME, CULIP_CUBLAS_ENV_NAME, __func__, &CULiP_cublas_lib_handle_cache);

	cudaStream_t cuda_stream;
	struct CULiP_profile_result profile_result;

	if (profiling_flag) {
		// Get current cuda stream
		cublasGetStream(handle, &cuda_stream);

		// Profile result structure
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s-n%d-k%d", __func__, CULiP_get_cublasFillMode_t_string(uplo), n, k);

		// Record start rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.start_timestamp);
	}

	// Call the function
	const cublasStatus_t result = (*cublas_lib_func)(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
	CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

	if (profiling_flag) {
		// Record end rimestamp
		CULiP_launch_function(cuda_stream, &CULiP_record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		CULiP_launch_function(cuda_stream, &CULiP_print_profile_result, (void*)&profile_result);
	}

	return result;
}
