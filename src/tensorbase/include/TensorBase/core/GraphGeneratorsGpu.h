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
  class GraphGeneratorGpu : public virtual GraphGenerator<LabelsT, TensorT, Eigen::GpuDevice> {
  protected:
    void initIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, const int& M, Eigen::GpuDevice& device) const override;
    void initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_or_link_ids, const int& N, Eigen::GpuDevice& device) const override;
    void getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_ids, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void GraphGeneratorGpu<LabelsT, TensorT>::initIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, const int& M, Eigen::GpuDevice& device) const
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
  inline void GraphGeneratorGpu<LabelsT, TensorT>::initIDs(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_or_link_ids, const int& N, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ N }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    node_or_link_ids = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 1>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void GraphGeneratorGpu<LabelsT, TensorT>::getUniqueIds(const int& offset, const int& span, const std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_ids, Eigen::GpuDevice& device) const
  {
    // Sort a copy of the data
    TensorDataGpuPrimitiveT<LabelsT, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ 2 * span }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    auto indices_tmp_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 1>>(indices_tmp);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 1>> indices_tmp_values(indices_tmp_ptr->getDataPointer().get(), indices_tmp_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> indices_values(indices->getDataPointer().get(), indices->getDimensions());
    indices_tmp_values.device(device) = indices_values.slice(Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 2 })).reshape(Eigen::array<Eigen::Index, 1>({ 2 * span }));
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
      gpuErrchk(cudaStreamSynchronize(device.stream()));
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

  template<typename LabelsT, typename TensorT>
  class KroneckerGraphGeneratorGpu: public KroneckerGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice>, public GraphGeneratorGpu<LabelsT, TensorT> {
  public:
    using KroneckerGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice>::KroneckerGraphGenerator;
  protected:
    void initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>>& indices_float, const int& M, Eigen::GpuDevice& device) const override;
    void permuteEdgeAndVertexList(std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>>& indices_float, const int& M, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::initKroneckerGraphTmpData(std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>>& indices_float, const int& M, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<float, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices_float = std::make_shared<TensorDataGpuPrimitiveT<float, 2>>(indices_tmp);
  }
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::permuteEdgeAndVertexList(std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>>& indices_float, const int& M, Eigen::GpuDevice& device) const
  {
    // Copy the current indices values
    Eigen::TensorMap<Eigen::Tensor<float, 2>> indices_float_values(indices_float->getDataPointer().get(), indices_float->getDimensions());
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    Eigen::array<Eigen::Index, 2> offset_1 = { 0,0 };
    Eigen::array<Eigen::Index, 2> span_1 = { M, 1 };
    Eigen::array<Eigen::Index, 2> offset_2 = { 0, 1 };
    Eigen::array<Eigen::Index, 2> span_2 = { M, 1 };
    std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> indices_copy = indices_float->copyToHost(device);
    indices_copy->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<float, 2>> indices_values_copy(indices_copy->getDataPointer().get(), indices_copy->getDimensions());

    // allocate temporary memory
    float* tmp_data;
    if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
      size_t bytes = M * M * sizeof(float);
      gpuErrchk(cudaMalloc((void**)(&tmp_data), bytes));
    }

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

    // deallocate temporary memory
    if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
      gpuErrchk(cudaFree(tmp_data));
    }
  }

  template<typename LabelsT, typename TensorT>
  class BinaryTreeGraphGeneratorGpu : public BinaryTreeGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice>, public GraphGeneratorGpu<LabelsT, TensorT> {
  public:
    using BinaryTreeGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice>::BinaryTreeGraphGenerator;
  };
}
#endif
#endif //TENSORBASE_GRAPHGENERATORSGPU_H