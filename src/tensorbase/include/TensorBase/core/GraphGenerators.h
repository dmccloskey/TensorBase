#ifndef TENSORBASE_GRAPHGENERATORS_H
#define TENSORBASE_GRAPHGENERATORS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include <random>
#include <TensorBase/ml/TensorData.h>

namespace TensorBase
{
  /*
  @class Class for generating a kronecker graph according to the specifications of the Graph500 Benchmark

  References:
    https://graph500.org/
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class KroneckerGraphGenerator {
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
    // allocate memory for the kronecker graph
    virtual void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, const int& M, DeviceT& device) const = 0;
    ///*
    //@brief Extract the list of unique Node IDs from the graph

    //@param[in] indices Node indices of the Kronecker graph
    //@param[out] node_ids Ordered and Unique list of node ids
    //@param[out] link_ids Ordered and Unique list of link ids

    //@returns A 1D list of nodes
    //*/
    //void getNodeAndLinkIds(const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& link_ids,DeviceT& device);
    ///// allocate memory for the IDs
    //virtual void initIDs(std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_or_link_ids, const int& N, DeviceT& device) const = 0;
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void KroneckerGraphGenerator<LabelsT, TensorT, DeviceT>::makeKroneckerGraph(const int& scale, const int& edge_factor, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, DeviceT& device) const
  {
    int N = std::pow(2,scale);// Set number of vertices.
    int M = edge_factor * N; // Set number of edges.
    double A = 0.57, B = 0.19, C = 0.19; // Set initiator probabilities.

    // Create index arrays and weights.
    initKroneckerGraph(indices, weights, M, device);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values(indices->getDataPointer().get(), indices->getDimensions());
    indices_values.device(device) = indices_values.constant(LabelsT(1));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weights_values(weights->getDataPointer().get(), weights->getDimensions());

    // Generate weights
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
      auto ii_bit = indices_values.slice(offset_1, span_1).cast<float>().random().abs().clip(float(0), float(1)) > indices_values.slice(offset_1, span_1).cast<float>().constant(float(ab));
      auto not_ii_bit = (ii_bit).select(ii_bit.constant(0), ii_bit.constant(1));
      auto jj_bit = indices_values.slice(offset_2, span_2).cast<float>().random().abs().clip(float(0), float(1)) > (
        indices_values.slice(offset_2, span_2).cast<float>().constant(float(c_norm)) * ii_bit.cast<float>() + indices_values.slice(offset_2, span_2).cast<float>().constant(float(a_norm)) * not_ii_bit.cast<float>() );
      indices_values.slice(offset_1, span_1).device(device) = indices_values.slice(offset_1, span_1) + indices_values.slice(offset_1, span_1).constant(LabelsT(std::pow(2, ib))) * ii_bit.cast<LabelsT>();
      indices_values.slice(offset_2, span_2).device(device) = indices_values.slice(offset_2, span_2) + indices_values.slice(offset_2, span_2).constant(LabelsT(std::pow(2, ib))) * jj_bit.cast<LabelsT>();
    }

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> indices_copy = indices->copy(device);
    indices_copy->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values_copy(indices_copy->getDataPointer().get(), indices_copy->getDimensions());
    // Permute vertex labels (in)
    {
      Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> vertex_permutation(indices->getDataPointer().get(), M, 1, 1);
      auto v_perm_bcast = vertex_permutation.broadcast(Eigen::array<Eigen::Index, 3>({ 1, M, 1 })).cast<float>().eval();
      auto v_perm_rand = v_perm_bcast.random().eval();
      auto v_perm_min_0 = v_perm_rand.minimum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).eval();
      auto v_perm_max_1 = v_perm_rand.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).eval();
      auto v_perm_ind_a = (v_perm_rand.chip(0, 2) == v_perm_max_1 || v_perm_rand.chip(0, 2) == v_perm_min_0).select(v_perm_bcast.chip(0, 2).constant(1), v_perm_bcast.chip(0, 2).constant(0)).cast<LabelsT>().eval();
      auto v_perm_ind_b = (v_perm_ind_a.cumsum(0) > 1).select(v_perm_ind_a.constant(0), v_perm_ind_a);
      auto v_perm_ind = (v_perm_ind_b.cumsum(1) > 1).select(v_perm_ind_b.constant(0), v_perm_ind_b);
      indices_values.slice(offset_1, span_1).device(device) = v_perm_ind.contract(indices_values.slice(offset_1, span_1).eval(), product_dims);
    }

    // Permute vertex labels (out)
    {
      Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> vertex_permutation(indices->getDataPointer().get(), M, 1, 1);
      auto v_perm_bcast = vertex_permutation.broadcast(Eigen::array<Eigen::Index, 3>({ 1, M, 1 })).cast<float>().eval();
      auto v_perm_rand = v_perm_bcast.random().eval();
      auto v_perm_min_0 = v_perm_rand.minimum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).eval();
      auto v_perm_max_1 = v_perm_rand.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).eval();
      auto v_perm_ind_a = (v_perm_rand.chip(0, 2) == v_perm_max_1 || v_perm_rand.chip(0, 2) == v_perm_min_0).select(v_perm_bcast.chip(0, 2).constant(1), v_perm_bcast.chip(0, 2).constant(0)).cast<LabelsT>().eval();
      auto v_perm_ind_b = (v_perm_ind_a.cumsum(0) > 1).select(v_perm_ind_a.constant(0), v_perm_ind_a);
      auto v_perm_ind = (v_perm_ind_b.cumsum(1) > 1).select(v_perm_ind_b.constant(0), v_perm_ind_b);
      indices_values.slice(offset_2, span_2).device(device) = v_perm_ind.contract(indices_values.slice(offset_2, span_2).eval(), product_dims);
    }
    indices_values.device(device) = (indices_values == indices_values.constant(0)).select(indices_values_copy, indices_values);

    // Permute the edge list
    {
      Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> vertex_permutation(indices->getDataPointer().get(), M, 1, 1);
      auto v_perm_bcast = vertex_permutation.broadcast(Eigen::array<Eigen::Index, 3>({ 1, M, 1 })).cast<float>().eval();
      auto v_perm_rand = v_perm_bcast.random().eval();
      auto v_perm_min_0 = v_perm_rand.minimum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).eval();
      auto v_perm_max_1 = v_perm_rand.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, M })).eval();
      auto v_perm_ind_a = (v_perm_rand.chip(0, 2) == v_perm_max_1 || v_perm_rand.chip(0, 2) == v_perm_min_0).select(v_perm_bcast.chip(0, 2).constant(1), v_perm_bcast.chip(0, 2).constant(0)).cast<LabelsT>().eval();
      auto v_perm_ind_b = (v_perm_ind_a.cumsum(0) > 1).select(v_perm_ind_a.constant(0), v_perm_ind_a);
      auto v_perm_ind = (v_perm_ind_b.cumsum(1) > 1).select(v_perm_ind_b.constant(0), v_perm_ind_b);
      indices_values.slice(offset_1, span_1).device(device) = v_perm_ind.contract(indices_values.slice(offset_1, span_1).eval(), product_dims);
      indices_values.slice(offset_2, span_2).device(device) = v_perm_ind.contract(indices_values.slice(offset_2, span_2).eval(), product_dims);
    }
    indices_values.device(device) = (indices_values == indices_values.constant(0)).select(indices_values_copy, indices_values);

    // Adjust to zero-based labels.
    indices_values.device(device) -= indices_values.constant(LabelsT(1));
  }
}
#endif //TENSORBASE_GRAPHGENERATORS_H