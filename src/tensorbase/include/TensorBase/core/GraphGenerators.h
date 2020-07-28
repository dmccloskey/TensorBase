#ifndef TENSORBASE_GRAPHGENERATORS_H
#define TENSORBASE_GRAPHGENERATORS_H

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
  @class Base class for all graph generators
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class GraphGenerator {
  public:
    virtual ~GraphGenerator() = default;
    /*
    @brief Extract the list of unique Node IDs from the graph

    @param[in] offset The starting point to extract out the unique nodes and link ids
    @param[in] span The length to extract out the unique nodes and link ids
    @param[in] indices Node indices of the Kronecker graph
    @param[out] node_ids Ordered and Unique list of node ids
    @param[out] link_ids Ordered and Unique list of link ids

    @returns A 1D list of nodes
    */
    void getNodeAndLinkIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& link_ids, DeviceT& device) const;
  protected:
    /// allocate memory for the kronecker graph
    virtual void initIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, const int& M, DeviceT& device) const = 0;
    /// allocate memory for the IDs
    virtual void initIDs(std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> & node_or_link_ids, const int& N, DeviceT & device) const = 0;
    /// determine the unique ids
    virtual void getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, DeviceT& device) const = 0;

  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void GraphGenerator<LabelsT, TensorT, DeviceT>::getNodeAndLinkIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& link_ids, DeviceT& device) const
  {
    // Allocate memory for the link ids
    this->initIDs(link_ids, span, device);

    // Make the link ids
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> link_ids_values(link_ids->getDataPointer().get(), link_ids->getDimensions());
    link_ids_values.device(device) = link_ids_values.constant(1).cumsum(0) + link_ids_values.constant(offset - 1);

    // Make the node ids
    getUniqueIds(offset, span, indices, node_ids, device);
  }

  /*
  @class Class for generating a kronecker graph according to the specifications of the Graph500 Benchmark

  References:
    https://graph500.org/
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class KroneckerGraphGenerator: public virtual GraphGenerator<LabelsT, TensorT, DeviceT> {
  public:
    KroneckerGraphGenerator() = default;
    ~KroneckerGraphGenerator() = default;
    /*
    @brief Generate the kronecker graph with a specified scale and edge factor
      where the number of vertices N = 2^scale and the number of edges M = N*edge_factor.
      The algorithm will generate a small number of duplicate links, self-links, and isolated links
      that must be accounted for in downstream algorithsm

    @param[in] scale
    @param[in] edge_factor
    @param[out] indices 2D Tensor of Edge in/out indices where size of dimension 0 = M and size of dimension 1 = 2
      with dimension 1 entries are node_in, node_out, and link IDs, respectively.
    */
    void makeKroneckerGraph(const int& scale, const int& edge_factor, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, DeviceT& device) const;
  protected:
    // allocate temporary memory for the kronecker graph
    virtual void initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, DeviceT, 2>>& indices_float, const int& M, DeviceT& device) const = 0;
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void KroneckerGraphGenerator<LabelsT, TensorT, DeviceT>::makeKroneckerGraph(const int& scale, const int& edge_factor, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, DeviceT& device) const
  {
    int N = std::pow(2,scale);// Set number of vertices.
    int M = edge_factor * N; // Set number of edges.
    double A = 0.57, B = 0.19, C = 0.19; // Set initiator probabilities.

    // Create index arrays and weights.
    this->initIndicesAndWeights(indices, weights, M, device);

    // Create the temporary data structures
    std::shared_ptr<TensorData<float, DeviceT, 2>> indices_float;
    initKroneckerGraphTmpData(indices_float, M, device);
    Eigen::TensorMap<Eigen::Tensor<float, 2>> indices_float_values(indices_float->getDataPointer().get(), indices_float->getDimensions());
    indices_float_values.device(device) = indices_float_values.constant(1);

    // Generate weights
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_values(weights->getDataPointer().get(), weights->getDimensions());
    weights_values.device(device) = weights_values.random().abs().clip(TensorT(0), TensorT(1));

    // Loop over each order of bit.
    double ab = A + B;
    double c_norm = C/(1 - (A + B));
    double a_norm = A/(A + B);
    Eigen::array<Eigen::Index, 2> offset_1 = { 0,0 };
    Eigen::array<Eigen::Index, 2> span_1 = { M, 1 };
    Eigen::array<Eigen::Index, 2> offset_2 = { 0, 1 };
    Eigen::array<Eigen::Index, 2> span_2 = { M, 1 };
    for (int ib = 0; ib < scale; ++ib) {
      // Compare with probabilities and set bits of indices.
      auto ii_bit = indices_float_values.slice(offset_1, span_1).random().abs().clip(float(0), float(1)) > indices_float_values.slice(offset_1, span_1).constant(float(ab));
      auto not_ii_bit = (ii_bit).select(ii_bit.constant(0), ii_bit.constant(1));
      auto jj_bit = indices_float_values.slice(offset_2, span_2).random().abs().clip(float(0), float(1)) > (
        indices_float_values.slice(offset_2, span_2).constant(float(c_norm)) * ii_bit + indices_float_values.slice(offset_2, span_2).constant(float(a_norm)) * not_ii_bit);
      indices_float_values.slice(offset_1, span_1).device(device) = indices_float_values.slice(offset_1, span_1) + indices_float_values.slice(offset_1, span_1).constant(float(std::pow(2, ib))) * ii_bit;
      indices_float_values.slice(offset_2, span_2).device(device) = indices_float_values.slice(offset_2, span_2) + indices_float_values.slice(offset_2, span_2).constant(float(std::pow(2, ib))) * jj_bit;
    }
    //std::cout << "bit assignment\n" << indices_float_values << std::endl;

    // Copy the current indices values
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    std::shared_ptr<TensorData<float, DeviceT, 2>> indices_copy = indices_float->copy(device);
    indices_copy->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<float, 2>> indices_values_copy(indices_copy->getDataPointer().get(), indices_copy->getDimensions());

    // allocate temporary memory
    float* tmp_data;
    if (typeid(device).name() != typeid(Eigen::GpuDevice).name()) {
      tmp_data = new float[M*M];
    }
#if COMPILE_WITH_CUDA
    else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
      size_t bytes = M * M * sizeof(float);
      assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
    }
#endif

    // Permute vertex labels (in)
    {
      Eigen::TensorMap<Eigen::Tensor<float, 3>> vertex_permutation(indices_float->getDataPointer().get(), M, 1, 1);
      auto v_perm_bcast = vertex_permutation.broadcast(Eigen::array<Eigen::Index, 3>({ 1, M, 1 })).eval();
      auto v_perm_rand = v_perm_bcast.random().eval();
      auto v_perm_min_0 = v_perm_rand.minimum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).eval();
      auto v_perm_max_1 = v_perm_rand.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).eval();
      Eigen::TensorMap<Eigen::Tensor<float, 2>> v_perm_ind_a(tmp_data, M, M);
      v_perm_ind_a.device(device) = (v_perm_rand.chip(0, 2) == v_perm_max_1 || v_perm_rand.chip(0, 2) == v_perm_min_0).select(v_perm_bcast.chip(0, 2).constant(1), v_perm_bcast.chip(0, 2).constant(0)).eval();
      auto v_perm_ind_b = (v_perm_ind_a.cumsum(0).eval() > v_perm_ind_a.constant(1)).select(v_perm_ind_a.constant(0), v_perm_ind_a);
      auto v_perm_ind = (v_perm_ind_b.cumsum(1).eval() > v_perm_ind_b.constant(1)).select(v_perm_ind_b.constant(0), v_perm_ind_b);
      indices_float_values.slice(offset_1, span_1).device(device) = v_perm_ind.contract(indices_float_values.slice(offset_1, span_1).eval(), product_dims);
    }

    // Permute vertex labels (out)
    {
      Eigen::TensorMap<Eigen::Tensor<float, 3>> vertex_permutation(indices_float->getDataPointer().get(), M, 1, 1);
      auto v_perm_bcast = vertex_permutation.broadcast(Eigen::array<Eigen::Index, 3>({ 1, M, 1 })).eval();
      auto v_perm_rand = v_perm_bcast.random().eval();
      auto v_perm_min_0 = v_perm_rand.minimum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).eval();
      auto v_perm_max_1 = v_perm_rand.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).eval();
      Eigen::TensorMap<Eigen::Tensor<float, 2>> v_perm_ind_a(tmp_data, M, M);
      v_perm_ind_a.device(device) = (v_perm_rand.chip(0, 2) == v_perm_max_1 || v_perm_rand.chip(0, 2) == v_perm_min_0).select(v_perm_bcast.chip(0, 2).constant(1), v_perm_bcast.chip(0, 2).constant(0)).eval();
      auto v_perm_ind_b = (v_perm_ind_a.cumsum(0).eval() > v_perm_ind_a.constant(1)).select(v_perm_ind_a.constant(0), v_perm_ind_a);
      auto v_perm_ind = (v_perm_ind_b.cumsum(1).eval() > v_perm_ind_b.constant(1)).select(v_perm_ind_b.constant(0), v_perm_ind_b);
      indices_float_values.slice(offset_2, span_2).device(device) = v_perm_ind.contract(indices_float_values.slice(offset_2, span_2).eval(), product_dims);
    }
    indices_float_values.device(device) = (indices_float_values == indices_float_values.constant(0)).select(indices_values_copy, indices_float_values);

    // Permute the edge list
    {
      Eigen::TensorMap<Eigen::Tensor<float, 3>> vertex_permutation(indices_float->getDataPointer().get(), M, 1, 1);
      auto v_perm_bcast = vertex_permutation.broadcast(Eigen::array<Eigen::Index, 3>({ 1, M, 1 })).eval();
      auto v_perm_rand = v_perm_bcast.random().eval();
      auto v_perm_min_0 = v_perm_rand.minimum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).eval();
      auto v_perm_max_1 = v_perm_rand.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).eval();
      Eigen::TensorMap<Eigen::Tensor<float, 2>> v_perm_ind_a(tmp_data, M, M);
      v_perm_ind_a.device(device) = (v_perm_rand.chip(0, 2) == v_perm_max_1 || v_perm_rand.chip(0, 2) == v_perm_min_0).select(v_perm_bcast.chip(0, 2).constant(1), v_perm_bcast.chip(0, 2).constant(0)).eval();
      auto v_perm_ind_b = (v_perm_ind_a.cumsum(0).eval() > v_perm_ind_a.constant(1)).select(v_perm_ind_a.constant(0), v_perm_ind_a);
      auto v_perm_ind = (v_perm_ind_b.cumsum(1).eval() > v_perm_ind_b.constant(1)).select(v_perm_ind_b.constant(0), v_perm_ind_b);
      indices_float_values.slice(offset_1, span_1).device(device) = v_perm_ind.contract(indices_float_values.slice(offset_1, span_1).eval(), product_dims);
      indices_float_values.slice(offset_2, span_2).device(device) = v_perm_ind.contract(indices_float_values.slice(offset_2, span_2).eval(), product_dims);
    }
    indices_float_values.device(device) = (indices_float_values == indices_float_values.constant(0)).select(indices_values_copy, indices_float_values);

    // Adjust to zero-based labels.
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values(indices->getDataPointer().get(), indices->getDimensions());
    indices_values.device(device) = indices_float_values.cast<LabelsT>() - indices_values.constant(LabelsT(1));

    // deallocate temporary memory
    if (typeid(device).name() != typeid(Eigen::GpuDevice).name()) {
      delete[] tmp_data;
    }
#if COMPILE_WITH_CUDA
    else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
      assert(cudaFree(tmp_data) == cudaSuccess);
    }
#endif
  }

  /*
  @class Class for generating a binary tree
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class BinaryTreeGraphGenerator : public virtual GraphGenerator<LabelsT, TensorT, DeviceT> {
  public:
    BinaryTreeGraphGenerator() = default;
    ~BinaryTreeGraphGenerator() = default;
    /*
    @brief Generate the binary tree with a certain depth

    @param[in] depth
    @param[in] edge_factor
    @param[out] indices 2D Tensor of Edge in/out indices where size of dimension 0 = M and size of dimension 1 = 2
      with dimension 1 entries are node_in, node_out, and link IDs, respectively.
    @param[out] weights
    */
    void makeBinaryTree(const int& depth, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, DeviceT& device) const;
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void BinaryTreeGraphGenerator<LabelsT, TensorT, DeviceT>::makeBinaryTree(const int & depth, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, DeviceT & device) const
  {
    const int n_nodes = std::pow(2, depth) + 1;
    const int n_links = std::pow(2, depth);

    // initialize the ptrs
    this->initIndicesAndWeights(indices, weights, n_links, device);

    // make the input node indices
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values_tmp(indices->getDataPointer().get(), indices->getDimensions().at(1), indices->getDimensions().at(0)/2);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values(indices->getDataPointer().get(), indices->getDimensions());
    indices_values.device(device) = indices_values.constant(LabelsT(0));
    auto indices_init = indices_values_tmp.constant(LabelsT(1)).cumsum(1) - indices_values_tmp.constant(LabelsT(1));
    auto indices_reshape = indices_init.reshape(Eigen::array<Eigen::Index, 1>({ indices->getDimensions().at(1) * indices->getDimensions().at(0) / 2 })).eval();
    indices_values.chip(0, 1).device(device) = indices_reshape;

    // make the output node indices
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_count(indices->getDataPointer().get(), indices->getDimensions().at(0));
    auto indices_count_cumsum = indices_count.constant(LabelsT(1)).cumsum(0) - indices_count.constant(LabelsT(1));
    auto indices_mod = indices_count_cumsum - (indices_count_cumsum.constant(2) * (indices_count_cumsum / indices_count_cumsum.constant(2))).eval(); // a mod n = a - (n * int(a/n))
    indices_values.chip(1, 1).device(device) = (indices_mod == indices_mod.constant(LabelsT(0))).select(
      indices_mod.constant(LabelsT(2)) * indices_values.chip(0, 1) + indices_mod.constant(LabelsT(1)), // left child = 2i + 1
      indices_mod.constant(LabelsT(2)) * indices_values.chip(0, 1) + indices_mod.constant(LabelsT(2))); // right child = 2i + 2

    // make the random weights
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_values(weights->getDataPointer().get(), weights->getDimensions());
    weights_values.device(device) = weights_values.random().abs().clip(TensorT(0), TensorT(1));    
  }
}
#endif //TENSORBASE_GRAPHGENERATORS_H