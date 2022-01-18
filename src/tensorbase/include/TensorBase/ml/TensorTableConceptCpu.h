/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPTCPU_H
#define TENSORBASE_TENSORTABLECONCEPTCPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <TensorBase/ml/TensorTableCpu.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::ThreadPoolDevice, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::ThreadPoolDevice, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::ThreadPoolDevice, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::ThreadPoolDevice, 4>, Eigen::ThreadPoolDevice>);
#endif //TENSORBASE_TENSORTABLECONCEPTCPU_H