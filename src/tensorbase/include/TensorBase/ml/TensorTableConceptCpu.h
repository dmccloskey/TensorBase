/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPTCPU_H
#define TENSORBASE_TENSORTABLECONCEPTCPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <TensorBase/ml/TensorTableCpu.h>

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray32<char>, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray32<char>, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray32<char>, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray32<char>, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray128<char>, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray128<char>, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray128<char>, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray128<char>, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray512<char>, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray512<char>, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray512<char>, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray512<char>, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray2048<char>, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray2048<char>, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray2048<char>, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray2048<char>, 4>, Eigen::ThreadPoolDevice>);
#endif //TENSORBASE_TENSORTABLECONCEPTCPU_H