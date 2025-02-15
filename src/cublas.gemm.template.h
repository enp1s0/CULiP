cublasStatus_t CULIP_FUNC_NAME(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               const CULIP_TYPE *alpha, const CULIP_TYPE *A,
                               int lda, const CULIP_TYPE *B, int ldb,
                               const CULIP_TYPE *beta, CULIP_TYPE *C, int ldc) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  const int cutoff_flag =
      (CULiP_cublas_profiling_control_array[CULIP_FUNC_ENUM_NAME] == 0) &&
      CULiP_is_profiling_enabled(CULIP_CUTOFF_THRESHOLD_ENV_NAME, false);
  if (cutoff_flag) {
    double threshold;
    std::string env_str;
    try {
      env_str = getenv(CULIP_CUTOFF_THRESHOLD_ENV_NAME);
      threshold = std::stod(env_str);

      cudaStream_t cuda_stream;
      cublasGetStream(handle, &cuda_stream);
      mtk::cu_cutoff::cutoff_small_abs_values(
          const_cast<CULIP_TYPE *>(A), (transa == CUBLAS_OP_N ? m : k),
          (transa == CUBLAS_OP_N ? k : m), lda, threshold, cuda_stream);
      mtk::cu_cutoff::cutoff_small_abs_values(
          const_cast<CULIP_TYPE *>(B), (transb == CUBLAS_OP_N ? k : n),
          (transb == CUBLAS_OP_N ? n : k), ldb, threshold, cuda_stream);
    } catch (const std::exception &e) {
      CULIBPROFILER_DEBUG_PRINT(
          printf("[CULiP Warning] invalid threshold (%s)\n", env_str.c_str()));
    }
  }

  const int profiling_flag =
      (CULiP_cublas_profiling_control_array[CULIP_FUNC_ENUM_NAME] == 0) &&
      CULiP_is_profiling_enabled(CULIP_CUBLAS_DISABLE_ENV_NAME);

  // Get the function pointer
  cublasStatus_t (*cublas_lib_func)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const CULIP_TYPE *, const CULIP_TYPE *, int, const CULIP_TYPE *, int,
      const CULIP_TYPE *, CULIP_TYPE *, int);
  *(void **)(&cublas_lib_func) = CULiP_get_function_pointer(__func__);

  cudaStream_t cuda_stream;
  struct CULiP_profile_result profile_result;

  if (profiling_flag) {
    // Get current cuda stream
    cublasGetStream(handle, &cuda_stream);

    // Profile result structure
    snprintf(profile_result.function_name,
             profile_result.function_name_length - 1, "%s-%s%s-m%d-n%d-k%d",
             __func__, CULiP_get_cublasOperation_t_string(transa),
             CULiP_get_cublasOperation_t_string(transb), m, n, k);

    // Record start rimestamp
    CULiP_launch_function(cuda_stream, &CULiP_record_timestamp,
                          (void *)&profile_result.start_timestamp);
  }

  // Call the function
  const cublasStatus_t result = (*cublas_lib_func)(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CULIBPROFILER_DEBUG_PRINT(printf("[CULiP Debug][%s] executed\n", __func__));

  if (profiling_flag) {
    // Record end rimestamp
    CULiP_launch_function(cuda_stream, &CULiP_record_timestamp,
                          (void *)&profile_result.end_timestamp);

    // Print result
    CULiP_launch_function(cuda_stream, &CULiP_print_profile_result,
                          (void *)&profile_result);
  }

  const int exp_stats_flag =
      (CULiP_cublas_profiling_control_array[CULIP_FUNC_ENUM_NAME] == 0) &&
      CULiP_is_profiling_enabled(CULIP_EXP_STATS_ENABLE_ENV_NAME, false);
  if (exp_stats_flag) {
    cudaStream_t cuda_stream;
    cublasGetStream(handle, &cuda_stream);
    CULiP_exp_stats a_stats;
    CULiP_exp_stats b_stats;
    snprintf(a_stats.name, a_stats.name_length - 1, "A");
    snprintf(b_stats.name, b_stats.name_length - 1, "B");
    a_stats.stats = mtk::cu_exp_statistics::take_matrix_statistics(
        A, (transa == CUBLAS_OP_N ? m : k), (transa == CUBLAS_OP_N ? k : m),
        lda, cuda_stream);
    b_stats.stats = mtk::cu_exp_statistics::take_matrix_statistics(
        B, (transb == CUBLAS_OP_N ? k : n), (transb == CUBLAS_OP_N ? n : k),
        ldb, cuda_stream);
    mtk::cu_exp_statistics::to_json(a_stats.stats);
    mtk::cu_exp_statistics::to_json(b_stats.stats);
    CULiP_launch_function(cuda_stream, &CULiP_print_exp_stats_result,
                          (void *)&a_stats);
    CULiP_launch_function(cuda_stream, &CULiP_print_exp_stats_result,
                          (void *)&b_stats);
  }

  return result;
#endif
}
