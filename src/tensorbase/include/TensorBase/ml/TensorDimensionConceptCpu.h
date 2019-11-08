/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONCONCEPTCPU_H
#define TENSORBASE_TENSORDIMENSIONCONCEPTCPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorDimensionConcept.h>
#include <TensorBase/ml/TensorDimensionCpu.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<int>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<float>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<double>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<char>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<TensorBase::TensorArray8<char>>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<TensorBase::TensorArray32<char>>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<TensorBase::TensorArray128<char>>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<TensorBase::TensorArray512<char>>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionCpu<TensorBase::TensorArray2048<char>>, Eigen::ThreadPoolDevice>);
#endif //TENSORBASE_TENSORDIMENSIONCONCEPTCPU_H