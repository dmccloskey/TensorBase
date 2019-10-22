/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONCONCEPTCPU_H
#define TENSORBASE_TENSORDIMENSIONCONCEPTCPU_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorDimensionConcept.h>
#include <TensorBase/ml/TensorDimensionCpu.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

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