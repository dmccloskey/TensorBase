/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONDEFAULTDEVICE_H
#define TENSORBASE_TENSORDIMENSIONDEFAULTDEVICE_H

#include <TensorBase/ml/TensorDimension.h>
#include <TensorBase/ml/TensorDataDefaultDevice.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT>
  class TensorDimensionDefaultDevice : public TensorDimension<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorDimensionDefaultDevice() = default;  ///< Default constructor
    TensorDimensionDefaultDevice(const std::string& name): TensorDimension(name) {};
    TensorDimensionDefaultDevice(const std::string& name, const size_t& n_labels) : TensorDimension(name, n_labels) { this->setLabels(); };
    TensorDimensionDefaultDevice(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) : TensorDimension(name) { this->setLabels(labels); };
    ~TensorDimensionDefaultDevice() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_ = std::make_shared<TensorDataDefaultDevice<TensorT, 1>>(TensorDataDefaultDevice<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    void setLabels() override {
      Eigen::array<Eigen::Index, 1> dimensions;
      dimensions.at(0) = this->n_labels_;
      this->labels_ = std::make_shared<TensorDataDefaultDevice<TensorT, 1>>(TensorDataDefaultDevice<TensorT, 1>(dimensions));
      this->labels_->setData();
    };
    bool loadLabelsBinary(const std::string& filename, Eigen::DefaultDevice& device) override {
      this->setDataStatus(true, false);
      Eigen::Tensor<TensorT, 1> data((int)this->n_labels_);
      DataFile::loadDataBinary<TensorT, 1>(filename + ".td", data);
      this->getLabels() = data;
      this->syncHAndDData(device); // H to D
      return true;
    };
    bool storeLabelsBinary(const std::string& filename, Eigen::DefaultDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::storeDataBinary<TensorT, 1>(filename + ".td", this->getLabels());
      this->setDataStatus(false, true);
      return true;
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<TensorT, Eigen::DefaultDevice>>(this));
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray8<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray32<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray128<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray512<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray2048<char>>);

#endif //TENSORBASE_TENSORDIMENSIONDEFAULTDEVICE_H