cublasStatus_t CULIP_FUNC_NAME(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const CULIP_TYPE *alpha, const CULIP_TYPE *A,
                               int lda, const CULIP_TYPE *B, int ldb,
                               const CULIP_REAL_TYPE *beta, CULIP_TYPE *C,
                               int ldc) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  const int profiling_flag =
      (CULiP_cublas_profiling_control_array[CULIP_FUNC_ENUM_NAME] == 0) &&
      CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

  // Get the function pointer
  cublasStatus_t (*cublas_lib_func)(
      cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,
      const CULIP_TYPE *, const CULIP_TYPE *, int, const CULIP_TYPE *, int,
      const CULIP_REAL_TYPE *, CULIP_TYPE *, int);
  *(void **)(&cublas_lib_func) = CULiP_get_function_pointer(__func__);

  cudaStream_t cuda_stream;
  struct CULiP_profile_result profile_result;

  if (profiling_flag) {
    // Get current cuda stream
    cublasGetStream(handle, &cuda_stream);

    // Profile result structure
    snprintf(profile_result.function_name,
             profile_result.function_name_length - 1, "%s-%s-%s-n%d-k%d",
             __func__, CULiP_get_cublasFillMode_t_string(uplo),
             CULiP_get_cublasOperation_t_string(trans), n, k);

    // Record start rimestamp
    CULiP_launch_function(cuda_stream, &CULiP_record_timestamp,
                          (void *)&profile_result.start_timestamp);
  }

  // Call the function
  const cublasStatus_t result = (*cublas_lib_func)(
      handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

  if (profiling_flag) {
    // Record end rimestamp
    CULiP_launch_function(cuda_stream, &CULiP_record_timestamp,
                          (void *)&profile_result.end_timestamp);

    // Print result
    CULiP_launch_function(cuda_stream, &CULiP_print_profile_result,
                          (void *)&profile_result);
  }

  return result;
#endif
}
