/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECPU_H
#define TENSORBASE_TENSORTABLECPU_H

#include <TensorBase/ml/TensorTable.h>

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
  };

  template<typename TensorT, int TDim>
  void TensorTableCpu<TensorT, TDim>::initData() {
    this->getData().reset(new TensorDataCpu<TensorT, TDim>(this->getDimensions()));
  }
};
#endif //TENSORBASE_TENSORTABLECPU_H