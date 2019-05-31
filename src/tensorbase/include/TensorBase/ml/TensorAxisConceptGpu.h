/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPTGPU_H
#define TENSORBASE_TENSORAXISCONCEPTGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorAxisGpu.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxisConcept.h>

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpu<int>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpu<float>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpu<double>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpu<char>, Eigen::GpuDevice>);
#endif
#endif //TENSORBASE_TENSORAXISCONCEPTGPU_H