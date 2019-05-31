/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPTGPU_H
#define TENSORBASE_TENSORTABLECONCEPTGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorTableGpu.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>

CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<int, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<float, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<double, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<int, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<float, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<double, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<int, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<float, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<double, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<int, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<float, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<double, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpu<char, 4>, Eigen::GpuDevice>);
#endif
#endif //TENSORBASE_TENSORTABLECONCEPTGPU_H