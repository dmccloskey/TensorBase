#ifndef TENSORBASE_GRAPHALGORITHMS_H
#define TENSORBASE_GRAPHALGORITHMS_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include <random>
#include <TensorBase/ml/TensorData.h>

namespace TensorBase
{
  /*
  @class Struct for converting a list of Sparse indices and weights to an adjacency matrix
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  struct IndicesAndWeightsToAdjacencyMatrix {
    /*
    @brief Generate the Adjecency matrix representation from a sparse list of node in and out indices pairs and
      weights.

    @param[in] node_ids 1D Tensor of unique node ids
    @param[in] indices 2D Tensor of link node_in and node_out pairs with dimensions 0 = M links and dimensions 1 = 2
    @param[in] weights 2D Tensor of weights with dimension 0 = M links and dimension 1 = 1
    @param[out] adjacency 2D adjacency Tensor with dimension 0 = N nodes and dimension 1 = N nodes 
      where the out nodes are represented along dimensions 0 and the in nodes are represented along dimensions 1
    */
    void operator()(const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, const std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& adjacency, DeviceT& device) const;
    virtual void initAdjacencyPtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& adjacency, DeviceT& device) = 0;
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void IndicesAndWeightsToAdjacencyMatrix<LabelsT, TensorT, DeviceT>::operator()(const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, const std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& adjacency, DeviceT& device) const {
    assert(weights->getDimensions().at(0) == indices->getDimensions().at(0));
    
    // 1. Build the incidence matrices Ein and Eout of type TensorT without self loops
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_values(weights->getDataPointer().get(), weights->getDimensions());
    auto weights_bcast = weights_values.broadcast(Eigen::array<Eigen::Index, 2>({ 1, node_ids->getTensorSize() })).eval();
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values(indices->getDataPointer().get(), indices->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> e_in_out_values(node_ids->getDataPointer().get(), 1, node_ids->getTensorSize());
    auto e_in_out_bcast = e_in_out_values.broadcast(Eigen::array<Eigen::Index, 2>({ indices->getDimensions().at(0), 1})).eval();
    auto indices_bcast_in = indices_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ indices->getDimensions().at(0), 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, node_ids->getTensorSize() })).eval();
    auto indices_bcast_out = indices_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 1 }), Eigen::array<Eigen::Index, 2>({ indices->getDimensions().at(0), 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, node_ids->getTensorSize() })).eval();
    // Ein will be of TensorT with 1s for incidences
    auto e_in = (e_in_out_bcast == indices_bcast_in && indices_bcast_in != indices_bcast_out).select(weights_bcast.constant(TensorT(1)), weights_bcast.constant(TensorT(0))).eval();
    // Eout will be of TensorT with weights for incidences
    auto e_out = (e_in_out_bcast == indices_bcast_out && indices_bcast_in != indices_bcast_out).select(weights_bcast, weights_bcast.constant(TensorT(0))).eval();

    // 2. Construct the adjacency matrix A = Eout.T * Ein
    initAdjacencyPtr(node_ids->getTensorSize(), adjacency, device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> adjacency_values(adjacency->getDataPointer().get(), adjacency->getDimensions()); 
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    adjacency_values.device(device) = e_out.shuffle(Eigen::array<Eigen::Index, 2>({1,0})).contract(e_in, product_dims);
  }

  /*
  @class Struct for performing a breadth first search (BFS) from a starting node
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  struct BFS {
    /*
    @brief Perform a BFS search starting from root and return a Tree of the search

    @param[in] root Node ID to use as the root
    @param[in] node_ids 1D Tensor of unique node ids
    @param[in] adjacency 2D adjacency Tensor with dimension 0 = N nodes and dimension 1 = N nodes
      where the out nodes are represented along dimensions 0 and the in nodes are represented along dimensions 1
    @param[out] tree 2D adjacency Tensor of the search tree starting from the root with dimension 0 = N nodes and dimension 1 = N nodes + 1
      where the nodes encountered during the search are recored as vectors along dimension 1
    */
    void operator()(const LabelsT root, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, const std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& adjacency, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& tree, DeviceT& device) const;
    virtual void initTreePtr(const int& n_nodes, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& tree, DeviceT& device) = 0;
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  void BFS<LabelsT, TensorT, DeviceT>::operator()(const LabelsT root, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, const std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& adjacency, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& tree, DeviceT& device) const {
    // 1. construct the root vector
    initTreePtr(node_ids->getTensorSize(), tree, device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> tree_values(tree->getDataPointer().get(), tree->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> node_ids_values(node_ids->getDataPointer().get(), 1, node_ids->getTensorSize());
    auto tree_values_slice = tree_values.slice(Eigen::array<Eigen::Index, 2>({ 0,0 }), Eigen::array<Eigen::Index, 2>({ node_ids->getTensorSize(), 1 }));
    tree_values.slice(Eigen::array<Eigen::Index, 2>({ 0,0 }), Eigen::array<Eigen::Index, 2>({ node_ids->getTensorSize(), 1 })).device(device) = (
      node_ids_values == node_ids_values.constant(root)).select(tree_values_slice.constant(TensorT(root)), tree_values_slice.constant(TensorT(0)));

    // 2. iteratively run A.T * v and update the tree
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> adjacency_values(adjacency->getDataPointer().get(), adjacency->getDimensions());
    for (int i = 0; i < node_ids->getTensorSize(); ++i) {
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
      auto tree_values_slice = tree_values.slice(Eigen::array<Eigen::Index, 2>({ 0,i }), Eigen::array<Eigen::Index, 2>({ node_ids->getTensorSize(), 1 })).eval();
      tree_values.slice(Eigen::array<Eigen::Index, 2>({ 0,i+1 }), Eigen::array<Eigen::Index, 2>({ node_ids->getTensorSize(), 1 })).device(device) = adjacency_values.shuffle(Eigen::array<Eigen::Index, 2>({ 1,0 })).contract(tree_values_slice, product_dims);
    }
  }
}
#endif //TENSORBASE_GRAPHALGORITHMS_H