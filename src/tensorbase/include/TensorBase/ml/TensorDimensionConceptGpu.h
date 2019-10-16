/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONCONCEPTGPU_H
#define TENSORBASE_TENSORDIMENSIONCONCEPTGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorDimensionGpu.h> // NOTE: this MUST be declared in a seperate file

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorDimensionConcept.h>

CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuPrimitiveT<int>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuPrimitiveT<float>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuPrimitiveT<double>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuPrimitiveT<char>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu8, char>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu32, char>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu128, char>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu512, char>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu2048, char>, Eigen::GpuDevice>);
#endif
#endif //TENSORBASE_TENSORDIMENSIONCONCEPTGPU_H