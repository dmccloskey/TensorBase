/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPTGPU_H
#define TENSORBASE_TENSORAXISCONCEPTGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorAxisGpu.h> // NOTE: this MUST be declared in a seperate file

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxisConcept.h>

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpuPrimitiveT<int>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpuPrimitiveT<float>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpuPrimitiveT<double>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpuPrimitiveT<char>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu8, char, Eigen::GpuDevice>);
#endif
#endif //TENSORBASE_TENSORAXISCONCEPTGPU_H