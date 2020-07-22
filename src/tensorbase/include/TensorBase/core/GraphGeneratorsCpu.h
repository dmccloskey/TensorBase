#ifndef TENSORBASE_GRAPHGENERATORSCPU_H
#define TENSORBASE_GRAPHGENERATORSCPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphGenerators.h>
#include <TensorBase/ml/TensorDataCpu.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  class KroneckerGraphGeneratorCpu: public KroneckerGraphGenerator<LabelsT, TensorT, Eigen::ThreadPoolDevice> {
  public:
    using KroneckerGraphGenerator<LabelsT, TensorT, Eigen::ThreadPoolDevice>::KroneckerGraphGenerator;
    /// allocate memory for the kronecker graph
    void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& weights, const int& M, Eigen::ThreadPoolDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorCpu<LabelsT, TensorT>::initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& weights, const int& M, Eigen::ThreadPoolDevice& device) const
  {
    TensorDataCpu<LabelsT, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataCpu<LabelsT, 2>>(indices_tmp);
    TensorDataCpu<TensorT, 2> weights_tmp(Eigen::array<Eigen::Index, 2>({ M, 1 }));
    weights_tmp.setData();
    weights_tmp.syncHAndDData(device);
    weights = std::make_shared<TensorDataCpu<TensorT, 2>>(weights_tmp);
  }
}
#endif //TENSORBASE_GRAPHGENERATORSCPU_H