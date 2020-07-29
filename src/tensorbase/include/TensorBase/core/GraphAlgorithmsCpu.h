#ifndef TENSORBASE_GRAPHALGORITHMSCPU_H
#define TENSORBASE_GRAPHALGORITHMSCPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphAlgorithms.h>
#include <TensorBase/ml/TensorDataCpu.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  struct IndicesAndWeightsToAdjacencyMatrixCpu: IndicesAndWeightsToAdjacencyMatrix<LabelsT, TensorT, Eigen::ThreadPoolDevice> {
    void initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& adjacency, Eigen::ThreadPoolDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void IndicesAndWeightsToAdjacencyMatrixCpu<LabelsT, TensorT>::initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& adjacency, Eigen::ThreadPoolDevice& device) const
  {
    TensorDataCpu<TensorT, 2> adjacency_tmp(Eigen::array<Eigen::Index, 2>({ n_nodes, n_nodes }));
    adjacency_tmp.setData();
    adjacency_tmp.syncHAndDData(device);
    adjacency = std::make_shared<TensorDataCpu<TensorT, 2>>(adjacency_tmp);
  }
  template<typename LabelsT, typename TensorT>
  struct BreadthFirstSearchCpu : public BreadthFirstSearch<LabelsT, TensorT, Eigen::ThreadPoolDevice> {
    void initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& tree, Eigen::ThreadPoolDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void BreadthFirstSearchCpu<LabelsT, TensorT>::initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& tree, Eigen::ThreadPoolDevice& device) const
  {
    TensorDataCpu<TensorT, 2> tree_tmp(Eigen::array<Eigen::Index, 2>({ n_nodes, n_nodes + 1 }));
    tree_tmp.setData();
    tree_tmp.syncHAndDData(device);
    tree = std::make_shared<TensorDataCpu<TensorT, 2>>(tree_tmp);
  }
  template<typename LabelsT, typename TensorT>
  struct SingleSourceShortestPathCpu : public SingleSourceShortestPath<LabelsT, TensorT, Eigen::ThreadPoolDevice> {
    void initPathLengthsPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 1>>& path_lengths, Eigen::ThreadPoolDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SingleSourceShortestPathCpu<LabelsT, TensorT>::initPathLengthsPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 1>>& path_lengths, Eigen::ThreadPoolDevice& device) const
  {
    TensorDataCpu<TensorT, 1> path_lengths_tmp(Eigen::array<Eigen::Index, 1>({ n_nodes }));
    path_lengths_tmp.setData();
    path_lengths_tmp.syncHAndDData(device);
    path_lengths = std::make_shared<TensorDataCpu<TensorT, 1>>(path_lengths_tmp);
  }
}
#endif //TENSORBASE_GRAPHALGORITHMSCPU_H