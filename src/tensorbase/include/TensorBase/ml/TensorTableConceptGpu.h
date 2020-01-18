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
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 4>, Eigen::GpuDevice>);

CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu8<char>, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu8<char>, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu8<char>, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu8<char>, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu32<char>, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu32<char>, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu32<char>, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu32<char>, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu128<char>, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu128<char>, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu128<char>, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu128<char>, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu512<char>, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu512<char>, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu512<char>, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu512<char>, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu2048<char>, Eigen::GpuDevice, 1>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu2048<char>, Eigen::GpuDevice, 2>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu2048<char>, Eigen::GpuDevice, 3>, Eigen::GpuDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArrayGpu2048<char>, Eigen::GpuDevice, 4>, Eigen::GpuDevice>);
#endif
#endif //TENSORBASE_TENSORTABLECONCEPTGPU_H