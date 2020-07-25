#ifndef TENSORBASE_GRAPHGENERATORSGPU_H
#define TENSORBASE_GRAPHGENERATORSGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphGenerators.h>
#include <TensorBase/ml/TensorDataGpu.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  class KroneckerGraphGeneratorGpu: public KroneckerGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using KroneckerGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice>::KroneckerGraphGenerator;
  protected:
    void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, const int& M, Eigen::GpuDevice& device) const override;
    void initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>>& indices_float, const int& M, Eigen::GpuDevice& device) const override;
    void initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_or_link_ids, const int& N, Eigen::GpuDevice& device) const override;
    void getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_ids, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, const int& M, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<LabelsT, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(indices_tmp);
    TensorDataGpuPrimitiveT<TensorT, 2> weights_tmp(Eigen::array<Eigen::Index, 2>({ M, 1 }));
    weights_tmp.setData();
    weights_tmp.syncHAndDData(device);
    weights = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(weights_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>>& indices_float, const int& M, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<float, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices_float = std::make_shared<TensorDataGpuPrimitiveT<float, 2>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_or_link_ids, const int& N, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ N }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    node_or_link_ids = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 1>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_ids, Eigen::GpuDevice& device) const
  {
    // Sort a copy of the data
    TensorDataGpuPrimitiveT<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ span }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    auto indices_tmp_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 1>>(indices_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_tmp_values(indices_tmp_ptr->getDataPointer().get(), indices_tmp_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_values(indices->getDataPointer().get(), indices->getTensorSize());
    indices_tmp_values.device(device) = indices_values.slice(Eigen::array<Eigen::Index, 1>({ offset }), Eigen::array<Eigen::Index, 1>({ span }));
    indices_tmp_ptr->sort("ASC", device);

    // Allocate memory
    TensorDataGpuPrimitiveT<LabelsT, 1> unique_tmp(indices_tmp_ptr->getDimensions());
    unique_tmp.setData();
    unique_tmp.syncHAndDData(device);
    TensorDataGpuPrimitiveT<int, 1> count_tmp(indices_tmp_ptr->getDimensions());
    count_tmp.setData();
    count_tmp.syncHAndDData(device);
    TensorDataGpuPrimitiveT<int, 1> n_runs_tmp(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_runs_tmp.setData();
    n_runs_tmp.syncHAndDData(device);

    // Move over the memory
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> unique = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 1>>(unique_tmp);
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> count = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(count_tmp);
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> n_runs = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(n_runs_tmp);

    // Run the algorithm
    indices_tmp_ptr->runLengthEncode(unique, count, n_runs, device);

    // Resize the unique results
    n_runs->syncHAndDData(device); // d to h
    if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    }
    unique->setDimensions(Eigen::array<Eigen::Index, 1>({ n_runs->getData()(0) }));

    // Copy over the results
    TensorDataGpuPrimitiveT<LabelsT, 1> node_ids_tmp(Eigen::array<Eigen::Index, 1>({ n_runs->getData()(0) }));
    node_ids_tmp.setData();
    node_ids_tmp.syncHAndDData(device);
    node_ids = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 1>>(node_ids_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> node_ids_values(node_ids->getDataPointer().get(), node_ids->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> unique_values(unique->getDataPointer().get(), unique->getDimensions());
    node_ids_values.device(device) = unique_values;
  }
}
#endif
#endif //TENSORBASE_GRAPHGENERATORSGPU_H