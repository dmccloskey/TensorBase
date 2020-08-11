#ifndef TENSORBASE_ASSERTGPU_H
#define TENSORBASE_ASSERTGPU_H

#if COMPILE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { TensorBaseAssert::gpuAssert((ans), __FILE__, __LINE__); }
#define gpuCheckEqual(lhs, rhs) { TensorBaseAssert::gpuAssertEqual((lhs), (rhs), __FILE__, __LINE__); }
#define gpuCheckNotEqual(lhs, rhs) { TensorBaseAssert::gpuAssertNotEqual((lhs), (rhs), __FILE__, __LINE__); }
#define gpuCheckLessThan(lhs, rhs) { TensorBaseAssert::gpuAssertLessThan((lhs), (rhs), __FILE__, __LINE__); }
#define gpuCheckGreaterThan(lhs, rhs) { TensorBaseAssert::gpuAssertGreaterThan((lhs), (rhs), __FILE__, __LINE__); }
#define gpuCheck(ans) { TensorBaseAssert::gpuAssertBool((ans), __FILE__, __LINE__); }

namespace TensorBaseAssert
{
  /// Check for Gpu errors
  inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
  {
    if (code != cudaSuccess)
    {
      std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
    }
  }

  /// Check Gpu values
  template<typename TensorTLHS, typename TensorTRHS>
  inline void gpuAssertEqual(const TensorTLHS& lhs, const TensorTRHS& rhs, const char* file, int line, bool abort = false)
  {
    if (!(lhs == static_cast<TensorTLHS>(rhs)))
    {
      std::cout << "GPUassert: " << file << " " << line << " " << lhs << " == " << rhs << std::endl;
      if (abort) exit(1);
    }
  }
  template<typename TensorTLHS, typename TensorTRHS>
  inline void gpuAssertNotEqual(const TensorTLHS& lhs, const TensorTRHS& rhs, const char* file, int line, bool abort = false)
  {
    if (!(lhs != static_cast<TensorTLHS>(rhs)))
    {
      std::cout << "GPUassert: " << file << " " << line << " " << lhs << " != " << rhs << std::endl;
      if (abort) exit(1);
    }
  }
  template<typename TensorTLHS, typename TensorTRHS>
  inline void gpuAssertLessThan(const TensorTLHS& lhs, const TensorTRHS& rhs, const char* file, int line, bool abort = false)
  {
    if (!(lhs < static_cast<TensorTLHS>(rhs)))
    {
      std::cout << "GPUassert: " << file << " " << line << " " << lhs << " < " << rhs << std::endl;
      if (abort) exit(1);
    }
  }
  template<typename TensorTLHS, typename TensorTRHS>
  inline void gpuAssertGreaterThan(const TensorTLHS& lhs, const TensorTRHS& rhs, const char* file, int line, bool abort = false)
  {
    if (!(lhs > static_cast<TensorTLHS>(rhs)))
    {
      std::cout << "GPUassert: " << file << " " << line << " " << lhs << " > " << rhs << std::endl;
      if (abort) exit(1);
    }
  }
  template<typename TensorT>
  inline void gpuAssertBool(const TensorT& ans, const char* file, int line, bool abort = false)
  {
    if (!ans)
    {
      std::cout << "GPUassert: " << file << " " << line << " " << ans << std::endl;
      if (abort) exit(1);
    }
  }
};
#endif
#endif //TENSORBASE_ASSERTGPU_H