/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONCPU_H
#define TENSORBASE_TENSORDIMENSIONCPU_H

#include <TensorBase/ml/TensorDimension.h>
#include <TensorBase/ml/TensorDataCpu.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT>
  class TensorDimensionCpu : public TensorDimension<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorDimensionCpu() = default;  ///< Default constructor
    TensorDimensionCpu(const std::string& name) : TensorDimension(name) {};
    TensorDimensionCpu(const std::string& name, const size_t& n_labels) : TensorDimension(name, n_labels) { this->setLabels(); };
    TensorDimensionCpu(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) : TensorDimension(name) { this->setLabels(labels); };
    ~TensorDimensionCpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_ = std::make_shared<TensorDataCpu<TensorT, 1>>(TensorDataCpu<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    void setLabels() override {
      Eigen::array<Eigen::Index, 1> dimensions;
      dimensions.at(0) = this->n_labels_;
      this->labels_ = std::make_shared<TensorDataCpu<TensorT, 1>>(TensorDataCpu<TensorT, 1>(dimensions));
      this->labels_->setData();
    };
    bool loadLabelsBinary(const std::string& filename, Eigen::ThreadPoolDevice& device) override {
      this->setDataStatus(true, false);
      Eigen::Tensor<TensorT, 1> data((int)this->n_labels_);
      DataFile::loadDataBinary<TensorT, 1>(filename + ".td", data);
      this->getLabels() = data;
      this->syncHAndDData(device); // H to D
      return true;
    };
    bool storeLabelsBinary(const std::string& filename, Eigen::ThreadPoolDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::storeDataBinary<TensorT, 1>(filename + ".td", this->getLabels());
      this->setDataStatus(false, true);
      return true;
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<TensorT, Eigen::ThreadPoolDevice>>(this));
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<TensorBase::TensorArray8<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<TensorBase::TensorArray32<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<TensorBase::TensorArray128<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<TensorBase::TensorArray512<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<TensorBase::TensorArray2048<char>>);

#endif //TENSORBASE_TENSORDIMENSIONCPU_H