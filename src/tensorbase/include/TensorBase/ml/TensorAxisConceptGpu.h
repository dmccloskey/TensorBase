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

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<int, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<float, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<double, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<char, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArrayGpu8<char>, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArrayGpu32<char>, Eigen::GpuDevice>, Eigen::GpuDevice>);
#if LARGE_GPU_ARRAY
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArrayGpu128<char>, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArrayGpu512<char>, Eigen::GpuDevice>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArrayGpu2048<char>, Eigen::GpuDevice>, Eigen::GpuDevice>);
#endif
#endif
#endif //TENSORBASE_TENSORAXISCONCEPTGPU_H