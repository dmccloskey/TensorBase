/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPTDEFAULTDEVICE_H
#define TENSORBASE_TENSORAXISCONCEPTDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxisConcept.h>
#include <TensorBase/ml/TensorAxisDefaultDevice.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<int, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<float, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<double, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<char, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray8<char>, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray32<char>, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray128<char>, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray512<char>, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray2048<char>, Eigen::DefaultDevice>, Eigen::DefaultDevice>);
#endif //TENSORBASE_TENSORAXISCONCEPTDEFAULTDEVICE_H