/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONCONCEPTDEFAULTDEVICE_H
#define TENSORBASE_TENSORDIMENSIONCONCEPTDEFAULTDEVICE_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorDimensionConcept.h>
#include <TensorBase/ml/TensorDimensionCpu.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<int>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<float>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<double>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<char>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray8<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray32<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray128<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray512<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray2048<char>>, Eigen::DefaultDevice>);
#endif //TENSORBASE_TENSORDIMENSIONCONCEPTDEFAULTDEVICE_H