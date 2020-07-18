/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPTCPU_H
#define TENSORBASE_TENSORAXISCONCEPTCPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxisConcept.h>
#include <TensorBase/ml/TensorAxisCpu.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<int>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<float>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<double>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<char>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<TensorBase::TensorArray8<char>>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<TensorBase::TensorArray32<char>>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<TensorBase::TensorArray128<char>>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<TensorBase::TensorArray512<char>>, Eigen::ThreadPoolDevice>);
//CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<TensorBase::TensorArray2048<char>>, Eigen::ThreadPoolDevice>);

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<int, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<float, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<double, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<char, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray8<char>, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray32<char>, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray128<char>, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray512<char>, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxis<TensorBase::TensorArray2048<char>, Eigen::ThreadPoolDevice>, Eigen::ThreadPoolDevice>);
#endif //TENSORBASE_TENSORAXISCONCEPTCPU_H