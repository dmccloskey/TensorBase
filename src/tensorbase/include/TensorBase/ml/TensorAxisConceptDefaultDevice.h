/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPTDEFAULTDEVICE_H
#define TENSORBASE_TENSORAXISCONCEPTDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxisConcept.h>
#include <TensorBase/ml/TensorAxisDefaultDevice.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<int>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<float>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<double>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<char>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray8<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray32<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray128<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray512<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray2048<char>>, Eigen::DefaultDevice>);
#endif //TENSORBASE_TENSORAXISCONCEPTDEFAULTDEVICE_H