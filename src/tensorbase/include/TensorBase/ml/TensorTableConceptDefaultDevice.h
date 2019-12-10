/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPTDEFAULTDEVICE_H
#define TENSORBASE_TENSORTABLECONCEPTDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray32<char>, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray32<char>, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray32<char>, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray32<char>, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray128<char>, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray128<char>, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray128<char>, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray128<char>, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray512<char>, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray512<char>, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray512<char>, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray512<char>, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray2048<char>, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray2048<char>, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray2048<char>, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray2048<char>, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_DYNAMIC_INIT(TensorTableConceptDefaultDevice);
#endif //TENSORBASE_TENSORTABLECONCEPTDEFAULTDEVICE_H