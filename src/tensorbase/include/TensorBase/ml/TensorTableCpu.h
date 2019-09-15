/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECPU_H
#define TENSORBASE_TENSORTABLECPU_H

#include <TensorBase/ml/TensorTable.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT, int TDim>
  class TensorTableCpu : public TensorTable<TensorT, Eigen::ThreadPoolDevice, TDim>
  {
  public:
    TensorTableCpu() = default;
    TensorTableCpu(const std::string& name) { this->setName(name); };
    ~TensorTableCpu() = default;
    void setAxes() override;
    void initData() override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorTable<TensorT, Eigen::ThreadPoolDevice, TDim>>(this));
    }
  };

  template<typename TensorT, int TDim>
  void TensorTableCpu<TensorT, TDim>::initData() {
    this->data_.reset(new TensorDataCpu<TensorT, TDim>(this->getDimensions()));
  }
};

// Cereal registration of TensorTs: float, int, char, double and TDims: 1, 2, 3, 4
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<int, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<float, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<double, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<int, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<float, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<double, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<int, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<float, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<double, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<int, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<float, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<double, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableCpu<char, 4>);
#endif //TENSORBASE_TENSORTABLECPU_H