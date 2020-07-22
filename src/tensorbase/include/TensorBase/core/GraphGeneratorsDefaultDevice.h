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
  protected:
    void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, const int& M, Eigen::DefaultDevice& device) const override;
    void initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>>& indices_float, const int& M, Eigen::DefaultDevice& device) const override;
    void initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_or_link_ids, const int& N, Eigen::DefaultDevice& device) const override;
    void getUniqueIds(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, Eigen::DefaultDevice& device) const override;
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
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorDefaultDevice<LabelsT, TensorT>::initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>>& indices_float, const int& M, Eigen::DefaultDevice& device) const
  {
    TensorDataDefaultDevice<float, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices_float = std::make_shared<TensorDataDefaultDevice<float, 2>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorDefaultDevice<LabelsT, TensorT>::initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_or_link_ids, const int& N, Eigen::DefaultDevice& device) const
  {
    TensorDataDefaultDevice<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ N }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    node_or_link_ids = std::make_shared<TensorDataDefaultDevice<LabelsT, 1>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorDefaultDevice<LabelsT, TensorT>::getUniqueIds(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, Eigen::DefaultDevice& device) const
  {
    // Sort a copy of the data
    TensorDataDefaultDevice<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ (int)indices->getTensorSize() }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    auto indices_tmp_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 1>>(indices_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_tmp_values(indices_tmp_ptr->getDataPointer().get(), indices_tmp_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_values(indices->getDataPointer().get(), indices->getTensorSize());
    indices_tmp_values.device(device) = indices_values;
    indices_tmp_ptr->sort("ASC", device);

    // Allocate memory
    TensorDataDefaultDevice<LabelsT, 1> unique_tmp(indices_tmp_ptr->getDimensions());
    unique_tmp.setData();
    unique_tmp.syncHAndDData(device);
    TensorDataDefaultDevice<int, 1> count_tmp(indices_tmp_ptr->getDimensions());
    count_tmp.setData();
    count_tmp.syncHAndDData(device);
    TensorDataDefaultDevice<int, 1> n_runs_tmp(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_runs_tmp.setData();
    n_runs_tmp.syncHAndDData(device);

    // Move over the memory
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> unique = std::make_shared<TensorDataDefaultDevice<LabelsT, 1>>(unique_tmp);
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> count = std::make_shared<TensorDataDefaultDevice<int, 1>>(count_tmp);
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> n_runs = std::make_shared<TensorDataDefaultDevice<int, 1>>(n_runs_tmp);

    // Run the algorithm
    indices_tmp_ptr->runLengthEncode(unique, count, n_runs, device);

    // Resize the unique results
    n_runs->syncHAndDData(device); // d to h
    //if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
    //  assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    //}
    unique->setDimensions(Eigen::array<Eigen::Index, 1>({ n_runs->getData()(0) }));

    // Copy over the results
    TensorDataDefaultDevice<LabelsT, 1> node_ids_tmp(Eigen::array<Eigen::Index, 1>({ n_runs->getData()(0) }));
    node_ids_tmp.setData();
    node_ids_tmp.syncHAndDData(device);
    node_ids = std::make_shared<TensorDataDefaultDevice<LabelsT, 1>>(node_ids_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> node_ids_values(node_ids->getDataPointer().get(), node_ids->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> unique_values(unique->getDataPointer().get(), unique->getDimensions());
    node_ids_values.device(device) = unique_values;
  }
}
#endif //TENSORBASE_GRAPHGENERATORSDEFAULTDEVICE_H