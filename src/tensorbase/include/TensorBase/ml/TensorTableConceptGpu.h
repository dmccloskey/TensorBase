/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPTGPU_H
#define TENSORBASE_TENSORTABLECONCEPTGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorTableGpuPrimitiveT.h>
#include <TensorBase/ml/TensorTableGpuClassT.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>

CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<int, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<float, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<double, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<int, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<float, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<double, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<int, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<float, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<double, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<int, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<float, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<double, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuPrimitiveT<char, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 4>, Eigen::GpuDevice>);
#endif
#endif //TENSORBASE_TENSORTABLECONCEPTGPU_H