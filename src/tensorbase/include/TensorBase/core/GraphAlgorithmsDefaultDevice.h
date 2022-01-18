#ifndef TENSORBASE_GRAPHALGORITHMSDEFAULTDEVICE_H
#define TENSORBASE_GRAPHALGORITHMSDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphAlgorithms.h>
#include <TensorBase/ml/TensorDataDefaultDevice.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  struct IndicesAndWeightsToAdjacencyMatrixDefaultDevice: IndicesAndWeightsToAdjacencyMatrix<LabelsT, TensorT, Eigen::DefaultDevice> {
    void initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& adjacency, Eigen::DefaultDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void IndicesAndWeightsToAdjacencyMatrixDefaultDevice<LabelsT, TensorT>::initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& adjacency, Eigen::DefaultDevice& device) const
  {
    TensorDataDefaultDevice<TensorT, 2> adjacency_tmp(Eigen::array<Eigen::Index, 2>({ n_nodes, n_nodes }));
    adjacency_tmp.setData();
    adjacency_tmp.syncHAndDData(device);
    adjacency = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(adjacency_tmp);
  }
  template<typename LabelsT, typename TensorT>
  struct BreadthFirstSearchDefaultDevice : public BreadthFirstSearch<LabelsT, TensorT, Eigen::DefaultDevice> {
    void initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& tree, Eigen::DefaultDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void BreadthFirstSearchDefaultDevice<LabelsT, TensorT>::initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& tree, Eigen::DefaultDevice& device) const
  {
    TensorDataDefaultDevice<TensorT, 2> tree_tmp(Eigen::array<Eigen::Index, 2>({ n_nodes, n_nodes + 1 }));
    tree_tmp.setData();
    tree_tmp.syncHAndDData(device);
    tree = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(tree_tmp);
  }
  template<typename LabelsT, typename TensorT>
  struct SingleSourceShortestPathDefaultDevice : public SingleSourceShortestPath<LabelsT, TensorT, Eigen::DefaultDevice> {
    void initPathLengthsPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& path_lengths, Eigen::DefaultDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SingleSourceShortestPathDefaultDevice<LabelsT, TensorT>::initPathLengthsPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& path_lengths, Eigen::DefaultDevice& device) const
  {
    TensorDataDefaultDevice<TensorT, 1> path_lengths_tmp(Eigen::array<Eigen::Index, 1>({ n_nodes }));
    path_lengths_tmp.setData();
    path_lengths_tmp.syncHAndDData(device);
    path_lengths = std::make_shared<TensorDataDefaultDevice<TensorT, 1>>(path_lengths_tmp);
  }
}
#endif //TENSORBASE_GRAPHALGORITHMSDEFAULTDEVICE_H