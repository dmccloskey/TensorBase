#ifndef TENSORBASE_GRAPHALGORITHMSGPU_H
#define TENSORBASE_GRAPHALGORITHMSGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphAlgorithms.h>
#include <TensorBase/ml/TensorDataGpu.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  struct IndicesAndWeightsToAdjacencyMatrixGpu: IndicesAndWeightsToAdjacencyMatrix<LabelsT, TensorT, Eigen::GpuDevice> {
    void initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& adjacency, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void IndicesAndWeightsToAdjacencyMatrixGpu<LabelsT, TensorT>::initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& adjacency, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<TensorT, 2> adjacency_tmp(Eigen::array<Eigen::Index, 2>({ n_nodes, n_nodes }));
    adjacency_tmp.setData();
    adjacency_tmp.syncHAndDData(device);
    adjacency = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(adjacency_tmp);
  }
  template<typename LabelsT, typename TensorT>
  struct BreadthFirstSearchGpu : public BreadthFirstSearch<LabelsT, TensorT, Eigen::GpuDevice> {
    void initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& tree, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void BreadthFirstSearchGpu<LabelsT, TensorT>::initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& tree, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<TensorT, 2> tree_tmp(Eigen::array<Eigen::Index, 2>({ n_nodes, n_nodes + 1 }));
    tree_tmp.setData();
    tree_tmp.syncHAndDData(device);
    tree = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(tree_tmp);
  }
  template<typename LabelsT, typename TensorT>
  struct SingleSourceShortestPathGpu : public SingleSourceShortestPath<LabelsT, TensorT, Eigen::GpuDevice> {
    void initPathLengthsPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& path_lengths, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SingleSourceShortestPathGpu<LabelsT, TensorT>::initPathLengthsPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& path_lengths, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<TensorT, 1> path_lengths_tmp(Eigen::array<Eigen::Index, 1>({ n_nodes }));
    path_lengths_tmp.setData();
    path_lengths_tmp.syncHAndDData(device);
    path_lengths = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(path_lengths_tmp);
  }
}
#endif
#endif //TENSORBASE_GRAPHALGORITHMSGPU_H