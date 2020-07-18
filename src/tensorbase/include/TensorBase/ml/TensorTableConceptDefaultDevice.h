/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPTDEFAULTDEVICE_H
#define TENSORBASE_TENSORTABLECONCEPTDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<int, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<float, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<double, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<char, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray8<char>, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray32<char>, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray128<char>, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray512<char>, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::DefaultDevice, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::DefaultDevice, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::DefaultDevice, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTable<TensorBase::TensorArray2048<char>, Eigen::DefaultDevice, 4>, Eigen::DefaultDevice>);

#endif //TENSORBASE_TENSORTABLECONCEPTDEFAULTDEVICE_H