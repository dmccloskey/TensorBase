#ifndef TENSORBASE_GRAPHGENERATORSDEFAULTDEVICE_H
#define TENSORBASE_GRAPHGENERATORSDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphGenerators.h>
#include <TensorBase/ml/TensorDataDefaultDevice.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  class KroneckerGraphGeneratorDefaultDevice: public KroneckerGraphGenerator<LabelsT, TensorT, Eigen::DefaultDevice> {
  public:
    using KroneckerGraphGenerator<LabelsT, TensorT, Eigen::DefaultDevice>::KroneckerGraphGenerator;
    /// allocate memory for the kronecker graph
    void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, const int& M, Eigen::DefaultDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorDefaultDevice<LabelsT, TensorT>::initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, const int& M, Eigen::DefaultDevice& device) const
  {
    TensorDataDefaultDevice<LabelsT, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(indices_tmp);
    TensorDataDefaultDevice<TensorT, 2> weights_tmp(Eigen::array<Eigen::Index, 2>({ M, 1 }));
    weights_tmp.setData();
    weights_tmp.syncHAndDData(device);
    weights = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(weights_tmp);
  }
}
#endif //TENSORBASE_GRAPHGENERATORSDEFAULTDEVICE_H