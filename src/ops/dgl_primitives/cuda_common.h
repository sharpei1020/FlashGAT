/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef DGL_RUNTIME_CUDA_CUDA_COMMON_H_
#define DGL_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <string>

namespace dgl
{
  namespace runtime
  {

    template <typename T>
    inline bool is_zero(T size) { return size == 0; }

    template <>
    inline bool is_zero<dim3>(dim3 size)
    {
      return size.x == 0 || size.y == 0 || size.z == 0;
    }

#define CUDA_DRIVER_CALL(x)                                           \
  {                                                                   \
    CUresult result = x;                                              \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) \
    {                                                                 \
      const char *msg;                                                \
      cuGetErrorName(result, &msg);                                   \
      LOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;   \
    }                                                                 \
  }

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)          \
  {                                                                         \
    if (!dgl::runtime::is_zero((nblks)) && !dgl::runtime::is_zero((nthrs))) \
    {                                                                       \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__);       \
      cudaError_t e = cudaGetLastError();                                   \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)              \
          << "CUDA kernel launch error: " << cudaGetErrorString(e);         \
    }                                                                       \
  }

#define CUBLAS_CALL(func)                                       \
  {                                                             \
    cublasStatus_t e = (func);                                  \
    CHECK(e == CUBLAS_STATUS_SUCCESS) << "CUBLAS ERROR: " << e; \
  }

#define CUSPARSE_CALL(func)                                         \
  {                                                                 \
    cusparseStatus_t e = (func);                                    \
    CHECK(e == CUSPARSE_STATUS_SUCCESS) << "CUSPARSE ERROR: " << e; \
  }

    /*
 * \brief Cast data type to cudaDataType_t.
 */
    template <typename T>
    struct cuda_dtype
    {
      static constexpr cudaDataType_t value = CUDA_R_32F;
    };

    template <>
    struct cuda_dtype<half>
    {
      static constexpr cudaDataType_t value = CUDA_R_16F;
    };

    template <>
    struct cuda_dtype<float>
    {
      static constexpr cudaDataType_t value = CUDA_R_32F;
    };

    template <>
    struct cuda_dtype<double>
    {
      static constexpr cudaDataType_t value = CUDA_R_64F;
    };

    /*
 * \brief Cast index data type to cusparseIndexType_t.
 */
    template <typename T>
    struct cusparse_idtype
    {
      static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
    };

    template <>
    struct cusparse_idtype<int32_t>
    {
      static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
    };

    template <>
    struct cusparse_idtype<int64_t>
    {
      static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_64I;
    };

  } // namespace runtime
} // namespace dgl
#endif // DGL_RUNTIME_CUDA_CUDA_COMMON_H_
