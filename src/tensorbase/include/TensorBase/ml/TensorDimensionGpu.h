/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONGPU_H
#define TENSORBASE_TENSORDIMENSIONGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorDataGpu.h>

#include <TensorBase/ml/TensorDimension.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT>
  class TensorDimensionGpu : public TensorDimension<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorDimensionGpu() = default;  ///< Default constructor
    TensorDimensionGpu(const std::string& name) { setName(name); };
    TensorDimensionGpu(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) { setName(name); setLabels(labels); };
    ~TensorDimensionGpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataGpu<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<TensorT, Eigen::GpuDevice>>(this));
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpu<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpu<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpu<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpu<char>);
#endif
#endif //TENSORBASE_TENSORDIMENSIONGPU_H