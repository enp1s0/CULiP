cublasStatus_t CULIP_FUNC_NAME(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               const CULIP_TYPE *alpha,
                               const CULIP_TYPE *const Aarray[], int lda,
                               const CULIP_TYPE *const Barray[], int ldb,
                               const CULIP_TYPE *beta,
                               CULIP_TYPE *const Carray[], int ldc,
                               int batchCount) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  const int profiling_flag =
      (CULiP_cublas_profiling_control_array[CULIP_FUNC_ENUM_NAME] == 0) &&
      CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

  // Get the function pointer
  cublasStatus_t (*cublas_lib_func)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const CULIP_TYPE *const, const CULIP_TYPE *const[], int,
      const CULIP_TYPE *const[], int, const CULIP_TYPE *const,
      CULIP_TYPE *const[], int, int);
  *(void **)(&cublas_lib_func) = CULiP_get_function_pointer(__func__);

  cudaStream_t cuda_stream;
  struct CULiP_profile_result profile_result;

  if (profiling_flag) {
    // Get current cuda stream
    cublasGetStream(handle, &cuda_stream);

    // Profile result structure
    snprintf(profile_result.function_name,
             profile_result.function_name_length - 1,
             "%s-%s%s-m%d-n%d-k%d-batchCount%d", __func__,
             CULiP_get_cublasOperation_t_string(transa),
             CULiP_get_cublasOperation_t_string(transb), m, n, k, batchCount);

    // Record start rimestamp
    CULiP_launch_function(cuda_stream, &CULiP_record_timestamp,
                          (void *)&profile_result.start_timestamp);
  }

  // Call the function
  const cublasStatus_t result =
      (*cublas_lib_func)(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                         Barray, ldb, beta, Carray, ldc, batchCount);
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
