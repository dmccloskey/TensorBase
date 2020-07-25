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
  protected:
    void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& weights, const int& M, Eigen::ThreadPoolDevice& device) const override;
    void initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 2>>& indices_float, const int& M, Eigen::ThreadPoolDevice& device) const override;
    void initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 1>>& node_or_link_ids, const int& N, Eigen::ThreadPoolDevice& device) const override;
    void getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 1>>& node_ids, Eigen::ThreadPoolDevice& device) const override;
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
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorCpu<LabelsT, TensorT>::initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 2>>& indices_float, const int& M, Eigen::ThreadPoolDevice& device) const
  {
    TensorDataCpu<float, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices_float = std::make_shared<TensorDataCpu<float, 2>>(indices_tmp);
  }  
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorCpu<LabelsT, TensorT>::initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 1>>& node_or_link_ids, const int& N, Eigen::ThreadPoolDevice& device) const
  {
    TensorDataCpu<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ N }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    node_or_link_ids = std::make_shared<TensorDataCpu<LabelsT, 1>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorCpu<LabelsT, TensorT>::getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 1>>& node_ids, Eigen::ThreadPoolDevice& device) const
  {
    // Sort a copy of the data
    TensorDataCpu<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ span }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    auto indices_tmp_ptr = std::make_shared<TensorDataCpu<LabelsT, 1>>(indices_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_tmp_values(indices_tmp_ptr->getDataPointer().get(), indices_tmp_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_values(indices->getDataPointer().get(), indices->getTensorSize());
    indices_tmp_values.device(device) = indices_values.slice(Eigen::array<Eigen::Index, 1>({ offset }), Eigen::array<Eigen::Index, 1>({ span }));
    indices_tmp_ptr->sort("ASC", device);

    // Allocate memory
    TensorDataCpu<LabelsT, 1> unique_tmp(indices_tmp_ptr->getDimensions());
    unique_tmp.setData();
    unique_tmp.syncHAndDData(device);
    TensorDataCpu<int, 1> count_tmp(indices_tmp_ptr->getDimensions());
    count_tmp.setData();
    count_tmp.syncHAndDData(device);
    TensorDataCpu<int, 1> n_runs_tmp(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_runs_tmp.setData();
    n_runs_tmp.syncHAndDData(device);

    // Move over the memory
    std::shared_ptr<TensorData<LabelsT, Eigen::ThreadPoolDevice, 1>> unique = std::make_shared<TensorDataCpu<LabelsT, 1>>(unique_tmp);
    std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> count = std::make_shared<TensorDataCpu<int, 1>>(count_tmp);
    std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> n_runs = std::make_shared<TensorDataCpu<int, 1>>(n_runs_tmp);

    // Run the algorithm
    indices_tmp_ptr->runLengthEncode(unique, count, n_runs, device);

    // Resize the unique results
    n_runs->syncHAndDData(device); // d to h
    //if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
    //  assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    //}
    unique->setDimensions(Eigen::array<Eigen::Index, 1>({ n_runs->getData()(0) }));

    // Copy over the results
    TensorDataCpu<LabelsT, 1> node_ids_tmp(Eigen::array<Eigen::Index, 1>({ n_runs->getData()(0) }));
    node_ids_tmp.setData();
    node_ids_tmp.syncHAndDData(device);
    node_ids = std::make_shared<TensorDataCpu<LabelsT, 1>>(node_ids_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> node_ids_values(node_ids->getDataPointer().get(), node_ids->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> unique_values(unique->getDataPointer().get(), unique->getDimensions());
    node_ids_values.device(device) = unique_values;
  }

}
#endif //TENSORBASE_GRAPHGENERATORSCPU_H