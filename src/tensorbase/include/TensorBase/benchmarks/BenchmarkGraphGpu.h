/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKGRAPHGPU_H
#define TENSORBASE_BENCHMARKGRAPHGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkGraph.h>
#include <TensorBase/ml/TensorCollectionGpu.h>
#include <TensorBase/ml/TensorOperationGpu.h>
#include <TensorBase/core/GraphGeneratorsGpu.h>
#include <TensorBase/core/GraphAlgorithmsGpu.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/// Specialized class for selecting and counting nodes with particular properties for the Gpu case
	template<typename LabelsT, typename TensorT>
	class SelectAndCountNodePropertyGpu: public SelectAndCountNodeProperty<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using SelectAndCountNodeProperty<LabelsT, TensorT, Eigen::GpuDevice>::SelectAndCountNodeProperty;
		void setLabelsValuesResult(Eigen::GpuDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectAndCountNodePropertyGpu<LabelsT, TensorT>::setLabelsValuesResult(Eigen::GpuDevice& device)
  {
    static_assert(std::is_same<TensorArrayGpu8<char>, LabelsT>::value, "Type mismatch");
    static_assert(std::is_same<TensorArrayGpu8<char>, TensorT>::value, "Type mismatch");

    // make the labels and sync to the device
    Eigen::Tensor<LabelsT, 2> select_labels_values(1, 1);
    select_labels_values.setConstant(LabelsT("label"));
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<TensorT, 1> select_values_values(1);
    select_values_values.setConstant(TensorT("white"));
    TensorDataGpuClassT<TensorArrayGpu8, char, 1> select_values(select_values_values.dimensions());
    select_values.setData(select_values_values);
    select_values.syncHAndDData(device);
    this->select_values_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(select_values);
  }

	/// Specialized class for selecting and counting nodes with particular properties for the Gpu case
	template<typename LabelsT, typename TensorT>
	class SelectAndCountLinkPropertyGpu : public SelectAndCountLinkProperty<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using SelectAndCountLinkProperty<LabelsT, TensorT, Eigen::GpuDevice>::SelectAndCountLinkProperty;
		void setLabelsValuesResult(Eigen::GpuDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectAndCountLinkPropertyGpu<LabelsT, TensorT>::setLabelsValuesResult(Eigen::GpuDevice& device)
  {
    static_assert(std::is_same<TensorArrayGpu8<char>, LabelsT>::value, "Type mismatch");
    static_assert(std::is_same<TensorArrayGpu8<char>, TensorT>::value, "Type mismatch");

    // make the labels and sync to the device
    Eigen::Tensor<LabelsT, 2> select_labels_values(1, 1);
    select_labels_values.setConstant(LabelsT("label"));
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<TensorT, 1> select_values_values(1);
    select_values_values.setConstant(TensorT("dashed"));
    TensorDataGpuClassT<TensorArrayGpu8, char, 1> select_values(select_values_values.dimensions());
    select_values.setData(select_values_values);
    select_values.syncHAndDData(device);
    this->select_values_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(select_values);
  }

  /// Specialized GraphAlgorithmHelper for the Gpu
  template<typename LabelsT, typename TensorT>
  struct GraphAlgorithmHelperGpu: public GraphAlgorithmHelper<LabelsT, TensorT, Eigen::GpuDevice> {
    void setIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device) override;
    void setNodeAndLinkIds(const std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& link_ids, Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void GraphAlgorithmHelperGpu<LabelsT, TensorT>::setIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device)
  {
    // set the indices
    TensorDataGpuPrimitiveT<LabelsT, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({
      tensor_collection->tables_.at("Graph_sparse_indices")->getDimSizeFromAxisName("1_links"),
      tensor_collection->tables_.at("Graph_sparse_indices")->getDimSizeFromAxisName("2_nodes") }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(indices_tmp);

    // set the weights
    TensorDataGpuPrimitiveT<TensorT, 2> weights_tmp(Eigen::array<Eigen::Index, 2>({
      tensor_collection->tables_.at("Graph_weights")->getDimSizeFromAxisName("1_links"),
      tensor_collection->tables_.at("Graph_weights")->getDimSizeFromAxisName("2_weights") }));
    weights_tmp.setData();
    weights_tmp.syncHAndDData(device);
    weights = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(weights_tmp);

    // get the indices
    std::shared_ptr<LabelsT[]> data_indices;
    tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_indices);
    indices->getDataPointer() = data_indices;

    // get the indices weights
    std::shared_ptr<TensorT[]> data_weights;
    tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_weights);
    weights->getDataPointer() = data_weights;
  }
  template<typename LabelsT, typename TensorT>
  inline void GraphAlgorithmHelperGpu<LabelsT, TensorT>::setNodeAndLinkIds(const std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>>& link_ids, Eigen::GpuDevice& device)
  {
    KroneckerGraphGeneratorGpu<LabelsT, TensorT> graph_generator;
    graph_generator.getNodeAndLinkIds(0, indices->getDimensions().at(0), indices, node_ids, link_ids, device);
  }

  /// Specialized class for make the adjacency matrix for the Gpu case
  template<typename LabelsT, typename TensorT>
  struct SelectAdjacencyGpu : public SelectAdjacency<LabelsT, TensorT, Eigen::GpuDevice> {
    void operator() (std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectAdjacencyGpu<LabelsT, TensorT>::operator()(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device)
  {
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> node_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> link_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> indices;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> weights;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> adjacency;
    GraphAlgorithmHelperGpu<LabelsT, TensorT> graph_algo_helper;
    graph_algo_helper.setIndicesAndWeights(indices, weights, tensor_collection, device);
    graph_algo_helper.setNodeAndLinkIds(indices, node_ids, link_ids, device);
    IndicesAndWeightsToAdjacencyMatrixGpu<LabelsT, TensorT> to_adjacency;
    to_adjacency(node_ids, indices, weights, adjacency, device);
    this->adjacency_ = adjacency;
  }

  /// Specialized class for making the BFS tree for the Gpu case
  template<typename LabelsT, typename TensorT>
  struct SelectBFSGpu : public SelectBFS<LabelsT, TensorT, Eigen::GpuDevice> {
    void operator() (std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectBFSGpu<LabelsT, TensorT>::operator()(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device)
  {
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> node_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> link_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> indices;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> weights;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> adjacency;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> tree;
    GraphAlgorithmHelperGpu<LabelsT, TensorT> graph_algo_helper;
    graph_algo_helper.setIndicesAndWeights(indices, weights, tensor_collection, device);
    graph_algo_helper.setNodeAndLinkIds(indices, node_ids, link_ids, device);
    IndicesAndWeightsToAdjacencyMatrixGpu<LabelsT, TensorT> to_adjacency;
    to_adjacency(node_ids, indices, weights, adjacency, device);
    BreadthFirstSearchGpu<LabelsT, TensorT> breadth_first_search;
    breadth_first_search(0, node_ids, adjacency, tree, device);
    this->tree_ = tree;
  }

  /// Specialized class for making the SSSP path lengths vector for the Gpu case
  template<typename LabelsT, typename TensorT>
  struct SelectSSSPGpu : public SelectSSSP<LabelsT, TensorT, Eigen::GpuDevice> {
    void operator() (std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectSSSPGpu<LabelsT, TensorT>::operator()(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device)
  {
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> node_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 1>> link_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> indices;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> weights;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> adjacency;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> tree;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>> path_lengths;
    GraphAlgorithmHelperGpu<LabelsT, TensorT> graph_algo_helper;
    graph_algo_helper.setIndicesAndWeights(indices, weights, tensor_collection, device);
    graph_algo_helper.setNodeAndLinkIds(indices, node_ids, link_ids, device);
    IndicesAndWeightsToAdjacencyMatrixGpu<LabelsT, TensorT> to_adjacency;
    to_adjacency(node_ids, indices, weights, adjacency, device);
    BreadthFirstSearchGpu<LabelsT, TensorT> breadth_first_search;
    breadth_first_search(0, node_ids, adjacency, tree, device);
    SingleSourceShortestPathGpu<LabelsT, TensorT> sssp;
    sssp(tree, path_lengths, device);
    this->path_lengths_ = path_lengths;
  }

  /// Helper structure to manage the KroneckerGraph data for the Gpu
  template<typename KGLabelsT, typename KGTensorT>
  struct GraphManagerHelperGpu: public GraphManagerHelper<KGLabelsT, KGTensorT, Eigen::GpuDevice> {
    void makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::GpuDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT>
  inline void GraphManagerHelperGpu<KGLabelsT, KGTensorT>::makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::GpuDevice& device) {
    KroneckerGraphGeneratorGpu<KGLabelsT, KGTensorT> graph_generator;
    graph_generator.makeKroneckerGraph(scale, edge_factor, this->kronecker_graph_indices_, this->kronecker_graph_weights_, device);
    graph_generator.getNodeAndLinkIds(0, this->kronecker_graph_indices_->getDimensions().at(0), this->kronecker_graph_indices_, this->kronecker_graph_node_ids_, this->kronecker_graph_link_ids_, device);
  }

	/*
	@class Specialized `GraphManagerSparseIndices` for the Gpu case
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	class GraphManagerSparseIndicesGpu : public GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice>::GraphManagerSparseIndices;
		void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) override;
		void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) override;
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	void GraphManagerSparseIndicesGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) {
		TensorDataGpuPrimitiveT<LabelsT, 2> tmp(dimensions);
		tmp.setData();
		tmp.syncHAndDData(device);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(tmp);
	}
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	void GraphManagerSparseIndicesGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) {
		TensorDataGpuPrimitiveT<TensorT, 2> tmp(dimensions);
		tmp.setData();
		tmp.syncHAndDData(device);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(tmp);
	}

  /*
  @class Specialized `GraphManagerWeights` for the Gpu case
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  class GraphManagerWeightsGpu : public GraphManagerWeights<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using GraphManagerWeights<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice>::GraphManagerWeights;
    void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) override;
    void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerWeightsGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) {
    TensorDataGpuPrimitiveT<LabelsT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerWeightsGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) {
    TensorDataGpuPrimitiveT<TensorT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(tmp);
  }

  /*
  @class Specialized `GraphManagerNodeProperty` for the Gpu case
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  class GraphManagerNodePropertyGpu : public GraphManagerNodeProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using GraphManagerNodeProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice>::GraphManagerNodeProperty;
    void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) override;
    void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) override;
    bool setNodeIds(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::GpuDevice, 2>>& kronecker_graph_indices, Eigen::GpuDevice& device) override;
    void setLabels(Eigen::GpuDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerNodePropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) {
    TensorDataGpuPrimitiveT<LabelsT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerNodePropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) {
    static_assert(std::is_same<TensorArrayGpu8<char>, TensorT>::value, "Type mismatch");
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    values_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  inline bool GraphManagerNodePropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::setNodeIds(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::GpuDevice, 2>>& kronecker_graph_indices, Eigen::GpuDevice& device)
  {
    // get all of the unique nodes up to this point
    KroneckerGraphGeneratorGpu<KGLabelsT, KGTensorT> graph_generator;
    std::shared_ptr<TensorData<KGLabelsT, Eigen::GpuDevice, 1>> link_ids; // temporary
    std::shared_ptr<TensorData<KGLabelsT, Eigen::GpuDevice, 1>> node_ids; // temporary
    graph_generator.getNodeAndLinkIds(0, offset + span, kronecker_graph_indices, node_ids, link_ids, device);

    // allocate and extract out the new unique node ids
    const int node_id_size = node_ids->getTensorSize() - this->nodes_added_cumulative_;
    if (node_id_size > 0) {
      TensorDataGpuPrimitiveT<KGLabelsT, 1> tmp(Eigen::array<Eigen::Index, 1>({ node_id_size }));
      tmp.setData();
      tmp.syncHAndDData(device);
      Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 1>> tmp_values(tmp.getDataPointer().get(), tmp.getDimensions());
      Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 1>> node_ids_values(node_ids->getDataPointer().get(), node_ids->getDimensions());
      tmp_values.device(device) = node_ids_values.slice(Eigen::array<Eigen::Index, 1>({ this->nodes_added_cumulative_ }), Eigen::array<Eigen::Index, 1>({ node_id_size }));

      // assign the values
      this->node_ids_ = std::make_shared<TensorDataGpuPrimitiveT<KGLabelsT, 1>>(tmp);
      this->nodes_added_cumulative_ += node_id_size;
      return true;
    }
    else return false;
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  inline void GraphManagerNodePropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::setLabels(Eigen::GpuDevice& device)
  {

    static_assert(std::is_same<TensorArrayGpu8<char>, TensorT>::value, "Type mismatch");
    TensorDataGpuClassT<TensorArrayGpu8, char, 1> tmp(this->node_ids_->getDimensions());
    tmp.setData();

    // Cycle through each color node;
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tmp_values(tmp.getDataPointer().get(), tmp.getDimensions());
    Eigen::Tensor<TensorT, 2> labels(1, 5);
    labels.setValues({ {TensorT("white"), TensorT("black"), TensorT("red"), TensorT("blue"), TensorT("green")} });
    const int bcast_length = this->node_ids_->getTensorSize() / 5 + 1;
    auto labels_bcast_reshape = labels.shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, bcast_length })).reshape(Eigen::array<Eigen::Index, 1>({ bcast_length * 5 }));
    tmp_values = labels_bcast_reshape.slice(Eigen::array<Eigen::Index, 1>({ 0 }), Eigen::array<Eigen::Index, 1>({ (int)tmp.getTensorSize() }));

    // Sync the data to the device and assign the data
    tmp.syncHAndDData(device);
    this->labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(tmp);
  }

  /*
  @class Specialized `GraphManagerLinkProperty` for the Gpu case
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  class GraphManagerLinkPropertyGpu : public GraphManagerLinkProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using GraphManagerLinkProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::GpuDevice>::GraphManagerLinkProperty;
    void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) override;
    void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) override;
    void setLabels(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::GpuDevice, 1>>& kronecker_graph_link_ids, Eigen::GpuDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerLinkPropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr, Eigen::GpuDevice& device) {
    TensorDataGpuPrimitiveT<LabelsT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerLinkPropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr, Eigen::GpuDevice& device) {

    static_assert(std::is_same<TensorArrayGpu8<char>, TensorT>::value, "Type mismatch");
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    values_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  inline void GraphManagerLinkPropertyGpu<KGLabelsT, KGTensorT, LabelsT, TensorT>::setLabels(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::GpuDevice, 1>>& kronecker_graph_link_ids, Eigen::GpuDevice& device)
  {
    static_assert(std::is_same<TensorArrayGpu8<char>, TensorT>::value, "Type mismatch");
    TensorDataGpuClassT<TensorArrayGpu8, char, 1> tmp(Eigen::array<Eigen::Index, 1>({span}));
    tmp.setData();

    // Dashed link every third index
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tmp_values(tmp.getDataPointer().get(), tmp.getDimensions());
    Eigen::Tensor<int, 1> indices(span);
    indices = indices.constant(1).cumsum(0) + indices.constant(offset - 1);
    auto indices_mod = indices - (indices.constant(3) * (indices / indices.constant(3))).eval(); // a mod n = a - (n * int(a/n))
    tmp_values = (indices_mod == indices.constant(0)).select(tmp_values.constant(TensorT("dashed")), tmp_values.constant(TensorT("solid")));
    
    // Sync the data to the device and assign the output
    tmp.syncHAndDData(device);
    this->labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(tmp);
  }

  /*
  @class A class for running 1 line insertion, deletion, and update benchmarks
  */
  class BenchmarkGraph1LinkGpu : public BenchmarkGraph1Link<int, float, Eigen::GpuDevice> {
  protected:
    void _makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::GpuDevice& device);
    void _insert1Link(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1Link`
    void _update1Link(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1Link`
    void _delete1Link(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1Link`
    int _selectAndCountNodeProperty(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndCountNodeProperty`
    int _selectAndCountLinkProperty(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndCountLinkProperty`
    float _selectAdjacency(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAdjacency`
    float _selectBFS(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectBFS`
    float _selectSSSP(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectSSSP`
  };
  inline void BenchmarkGraph1LinkGpu::_makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::GpuDevice& device)
  {
    GraphManagerHelperGpu<int, float> gmh;
    gmh.makeKroneckerGraph(scale, edge_factor, device);
    this->graph_manager_helper_ = std::make_shared<GraphManagerHelperGpu<int, float>>(gmh);
  }
  void BenchmarkGraph1LinkGpu::_insert1Link(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const {
    GraphManagerSparseIndicesGpu<int, float, int, int> graph_manager_sparse_indices(false);
    GraphManagerWeightsGpu<int, float, int, float> graph_manager_weights(false);
    GraphManagerNodePropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_node_property(false);
    GraphManagerLinkPropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_link_property(false);
    int span = std::pow(2, scale);
    for (int i = 0; i < std::pow(2, scale) * edge_factor; i += span) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_sparse_indices_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_sparse_indices_ptr;
      graph_manager_sparse_indices.getInsertData(i, span, labels_sparse_indices_ptr, values_sparse_indices_ptr, 
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_,
        device);
      TensorAppendToAxis<int, int, Eigen::GpuDevice, 2> appendToAxis_sparse_indices("Graph_sparse_indices", "1_links", labels_sparse_indices_ptr, values_sparse_indices_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_sparse_indices_ptr = std::make_shared<TensorAppendToAxis<int, int, Eigen::GpuDevice, 2>>(appendToAxis_sparse_indices);
      transaction_manager.executeOperation(appendToAxis_sparse_indices_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_weights_ptr;
      std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> values_weights_ptr;
      graph_manager_weights.getInsertData(i, span, labels_weights_ptr, values_weights_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      TensorAppendToAxis<int, float, Eigen::GpuDevice, 2> appendToAxis_weights("Graph_weights", "1_links", labels_weights_ptr, values_weights_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_weights_ptr = std::make_shared<TensorAppendToAxis<int, float, Eigen::GpuDevice, 2>>(appendToAxis_weights);
      transaction_manager.executeOperation(appendToAxis_weights_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_node_property_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_node_property_ptr;
      graph_manager_node_property.getInsertData(i, span, labels_node_property_ptr, values_node_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      if (labels_node_property_ptr != nullptr && labels_node_property_ptr->getTensorSize() > 0) {
        TensorAppendToAxis<int, TensorArrayGpu8<char>, Eigen::GpuDevice, 2> appendToAxis_node_property("Graph_node_property", "1_nodes", labels_node_property_ptr, values_node_property_ptr);
        std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_node_property_ptr = std::make_shared<TensorAppendToAxis<int, TensorArrayGpu8<char>, Eigen::GpuDevice, 2>>(appendToAxis_node_property);
        transaction_manager.executeOperation(appendToAxis_node_property_ptr, device);
      }
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_link_property_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_link_property_ptr;
      graph_manager_link_property.getInsertData(i, span, labels_link_property_ptr, values_link_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      TensorAppendToAxis<int, TensorArrayGpu8<char>, Eigen::GpuDevice, 2> appendToAxis_link_property("Graph_link_property", "1_links", labels_link_property_ptr, values_link_property_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_link_property_ptr = std::make_shared<TensorAppendToAxis<int, TensorArrayGpu8<char>, Eigen::GpuDevice, 2>>(appendToAxis_link_property);
      transaction_manager.executeOperation(appendToAxis_link_property_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
  }
  void BenchmarkGraph1LinkGpu::_update1Link(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const {
    GraphManagerSparseIndicesGpu<int, float, int, int> graph_manager_sparse_indices(true);
    GraphManagerWeightsGpu<int, float, int, float> graph_manager_weights(true);
    GraphManagerNodePropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_node_property(true);
    GraphManagerLinkPropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_link_property(true);
    int span = std::pow(2, scale);
    for (int i = 0; i < std::pow(2, scale) * edge_factor; i += span) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_sparse_indices_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_sparse_indices_ptr;
      graph_manager_sparse_indices.getInsertData(i, span, labels_sparse_indices_ptr, values_sparse_indices_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_sparse_indices(labels_sparse_indices_ptr, "Graph_sparse_indices", "1_links");
      TensorUpdateValues<int, Eigen::GpuDevice, 2> updateValues_sparse_indices("Graph_sparse_indices", selectClause_sparse_indices, values_sparse_indices_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_sparse_indices_ptr = std::make_shared<TensorUpdateValues<int, Eigen::GpuDevice, 2>>(updateValues_sparse_indices);
      transaction_manager.executeOperation(updateValues_sparse_indices_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_weights_ptr;
      std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> values_weights_ptr;
      graph_manager_weights.getInsertData(i, span, labels_weights_ptr, values_weights_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_weights(labels_weights_ptr, "Graph_weights", "1_links");
      TensorUpdateValues<float, Eigen::GpuDevice, 2> updateValues_weights("Graph_weights", selectClause_weights, values_weights_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_weights_ptr = std::make_shared<TensorUpdateValues<float, Eigen::GpuDevice, 2>>(updateValues_weights);
      transaction_manager.executeOperation(updateValues_weights_ptr, device);
      // Update the nodes outside the loop
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_link_property_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_link_property_ptr;
      graph_manager_link_property.getInsertData(i, span, labels_link_property_ptr, values_link_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_link_property(labels_link_property_ptr, "Graph_link_property", "1_links");
      TensorUpdateValues<TensorArrayGpu8<char>, Eigen::GpuDevice, 2> updateValues_link_property("Graph_link_property", selectClause_link_property, values_link_property_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_link_property_ptr = std::make_shared<TensorUpdateValues<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>>(updateValues_link_property);
      transaction_manager.executeOperation(updateValues_link_property_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_node_property_ptr;
    std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_node_property_ptr;
    graph_manager_node_property.getInsertData(0, std::pow(2, scale) * edge_factor, labels_node_property_ptr, values_node_property_ptr,
    this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
    SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_node_property(labels_node_property_ptr, "Graph_node_property", "1_nodes");
    TensorUpdateValues<TensorArrayGpu8<char>, Eigen::GpuDevice, 2> updateValues_node_property("Graph_node_property", selectClause_node_property, values_node_property_ptr);
    std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_node_property_ptr = std::make_shared<TensorUpdateValues<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>>(updateValues_node_property);
    transaction_manager.executeOperation(updateValues_node_property_ptr, device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
  }
  void BenchmarkGraph1LinkGpu::_delete1Link(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const {
    GraphManagerSparseIndicesGpu<int, float, int, int> graph_manager_sparse_indices(true);
    GraphManagerWeightsGpu<int, float, int, float> graph_manager_weights(true);
    GraphManagerNodePropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_node_property(true);
    GraphManagerLinkPropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_link_property(true);
    int span = std::pow(2, scale);
    for (int i = 0; i < std::pow(2, scale) * edge_factor - 1; i += span) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_sparse_indices_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_sparse_indices_ptr;
      graph_manager_sparse_indices.getInsertData(i, span, labels_sparse_indices_ptr, values_sparse_indices_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_sparse_indices(labels_sparse_indices_ptr, "Graph_sparse_indices", "1_links");
      TensorDeleteFromAxisGpuPrimitiveT<int, int, 2> tensorDelete_sparse_indices("Graph_sparse_indices", "1_links", selectClause_sparse_indices);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_sparse_indices_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<int, int, 2>>(tensorDelete_sparse_indices);
      transaction_manager.executeOperation(tensorDelete_sparse_indices_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_weights_ptr;
      std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> values_weights_ptr;
      graph_manager_weights.getInsertData(i, span, labels_weights_ptr, values_weights_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_weights(labels_weights_ptr, "Graph_weights", "1_links");
      TensorDeleteFromAxisGpuPrimitiveT<int, float, 2> tensorDelete_weights("Graph_weights", "1_links", selectClause_weights);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_weights_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<int, float, 2>>(tensorDelete_weights);
      transaction_manager.executeOperation(tensorDelete_weights_ptr, device);
      // Delete the nodes outside the loop
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_link_property_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_link_property_ptr;
      graph_manager_link_property.getInsertData(i, span, labels_link_property_ptr, values_link_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_link_property(labels_link_property_ptr, "Graph_link_property", "1_links");
      TensorDeleteFromAxisGpuClassT<int, TensorArrayGpu8, char, 2> tensorDelete_link_property("Graph_link_property", "1_links", selectClause_link_property);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_link_property_ptr = std::make_shared<TensorDeleteFromAxisGpuClassT<int, TensorArrayGpu8, char, 2>>(tensorDelete_link_property);
      transaction_manager.executeOperation(tensorDelete_link_property_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }      
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_node_property_ptr;
    std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_node_property_ptr;
    graph_manager_node_property.getInsertData(0, std::pow(2, scale) * edge_factor, labels_node_property_ptr, values_node_property_ptr,
    this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
    SelectGraphNodeLinkIDs<int, Eigen::GpuDevice> selectClause_node_property(labels_node_property_ptr, "Graph_node_property", "1_nodes");
    TensorDeleteFromAxisGpuClassT<int, TensorArrayGpu8, char, 2> tensorDelete_node_property("Graph_node_property", "1_nodes", selectClause_node_property);
    std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_node_property_ptr = std::make_shared<TensorDeleteFromAxisGpuClassT<int, TensorArrayGpu8, char, 2>>(tensorDelete_node_property);
    transaction_manager.executeOperation(tensorDelete_node_property_ptr, device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
  }
  inline int BenchmarkGraph1LinkGpu::_selectAndCountNodeProperty(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndCountNodePropertyGpu<TensorArrayGpu8<char>, TensorArrayGpu8<char>> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_and_sum.result_;
  }
  inline int BenchmarkGraph1LinkGpu::_selectAndCountLinkProperty(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndCountNodePropertyGpu<TensorArrayGpu8<char>, TensorArrayGpu8<char>> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_and_sum.result_;
  }
  inline float BenchmarkGraph1LinkGpu::_selectAdjacency(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAdjacencyGpu<int, float> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.adjacency_->syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    return select_and_sum.adjacency_->getData()(0, 0);
  }
  inline float BenchmarkGraph1LinkGpu::_selectBFS(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectBFSGpu<int, float> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.tree_->syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    return select_and_sum.tree_->getData()(0, 0);
  }
  inline float BenchmarkGraph1LinkGpu::_selectSSSP(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectSSSPGpu<int, float> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.path_lengths_->syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    return select_and_sum.path_lengths_->getData()(0);
  }

  class GraphTensorCollectionGeneratorGpu : public GraphTensorCollectionGenerator<Eigen::GpuDevice> {
  public:
    std::shared_ptr<TensorCollection<Eigen::GpuDevice>> makeTensorCollection(const int& scale, const int& edge_factor, const double& shard_span_perc, Eigen::GpuDevice& device) const override;
  };
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> GraphTensorCollectionGeneratorGpu::makeTensorCollection(const int& scale, const int& edge_factor, const double& shard_span_perc, Eigen::GpuDevice& device) const
  {
    const int data_size = std::pow(2, scale) * edge_factor;

    // Setup the axes
    Eigen::Tensor<std::string, 1> dimensions_1a(1), dimensions_1b(1), dimensions_2a(1), dimensions_2b(1), dimensions_2(1), dimensions_3a(1), dimensions_3b(1);
    dimensions_1a.setValues({ "links" });
    dimensions_1b.setValues({ "nodes" });
    dimensions_2a.setValues({ "weights" });
    dimensions_2b.setValues({ "indices" });
    dimensions_3a.setValues({ "link_property" });
    dimensions_3b.setValues({ "node_property" });
    Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_1b(1, 2), labels_1c(1, 1), labels_3(1, 1);;
    labels_1b.setValues({ { TensorArrayGpu8<char>("node_in"), TensorArrayGpu8<char>("node_out")} });
    labels_1c.setValues({ { TensorArrayGpu8<char>("weights")} });
    labels_3.setValues({ { TensorArrayGpu8<char>("label")} });

    // Setup the tables
    std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(TensorTableGpuPrimitiveT<int, 2>("Graph_sparse_indices"));
    std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("2_nodes", dimensions_2b, labels_1b));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_links", 1, 0));
    table_1_axis_2_ptr->setDimensions(dimensions_2);
    table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
    table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
    table_1_ptr->setAxes(device);

    // Setup the table data
    table_1_ptr->setData();
    std::map<std::string, int> shard_span_1;
    shard_span_1.emplace("2_nodes", 1);
    shard_span_1.emplace("1_links", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    table_1_ptr->setShardSpans(shard_span_1);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 2 }));

    // Setup the tables
    std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 2>> table_2_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 2>>(TensorTableGpuPrimitiveT<float, 2>("Graph_weights"));
    std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_2_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("2_weights", dimensions_2a, labels_1c));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_2_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_links", 1, 0));
    table_2_axis_2_ptr->setDimensions(dimensions_2);
    table_2_ptr->addTensorAxis(table_2_axis_1_ptr);
    table_2_ptr->addTensorAxis(table_2_axis_2_ptr);
    table_2_ptr->setAxes(device);

    // Setup the table data
    table_2_ptr->setData();
    std::map<std::string, int> shard_span_2;
    shard_span_2.emplace("2_weights", 1);
    shard_span_2.emplace("1_links", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    table_2_ptr->setShardSpans(shard_span_2);
    table_2_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 1 }));

    // Setup the tables
    std::shared_ptr<TensorTable<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> table_3_ptr = std::make_shared<TensorTableGpuClassT<TensorArrayGpu8, char, 2>>(TensorTableGpuClassT<TensorArrayGpu8, char, 2>("Graph_node_property"));
    std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_3_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("2_property", dimensions_3b, labels_3));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_3_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_nodes", 1, 0));
    table_3_axis_2_ptr->setDimensions(dimensions_2);
    table_3_ptr->addTensorAxis(table_3_axis_1_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_2_ptr);
    table_3_ptr->setAxes(device);

    // Setup the table data
    table_3_ptr->setData();
    std::map<std::string, int> shard_span_3;
    shard_span_3.emplace("2_property", 1);
    shard_span_3.emplace("1_nodes", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    table_3_ptr->setShardSpans(shard_span_3);
    table_3_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 1 }));

    // Setup the tables
    std::shared_ptr<TensorTable<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> table_4_ptr = std::make_shared<TensorTableGpuClassT<TensorArrayGpu8, char, 2>>(TensorTableGpuClassT<TensorArrayGpu8, char, 2>("Graph_link_property"));
    std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_4_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("2_property", dimensions_1a, labels_3));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_4_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_links", 1, 0));
    table_4_axis_2_ptr->setDimensions(dimensions_2);
    table_4_ptr->addTensorAxis(table_4_axis_1_ptr);
    table_4_ptr->addTensorAxis(table_4_axis_2_ptr);
    table_4_ptr->setAxes(device);

    // Setup the table data
    table_4_ptr->setData();
    std::map<std::string, int> shard_span_4;
    shard_span_4.emplace("2_property", 1);
    shard_span_4.emplace("1_links", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    table_4_ptr->setShardSpans(shard_span_4);
    table_4_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 1 }));

    // Setup the collection
    auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
    collection_1_ptr->addTensorTable(table_1_ptr, "Graph");
    collection_1_ptr->addTensorTable(table_2_ptr, "Graph");
    collection_1_ptr->addTensorTable(table_3_ptr, "Graph");
    collection_1_ptr->addTensorTable(table_4_ptr, "Graph");
    // TODO: linking of axes
    return collection_1_ptr;
  }
};
#endif
#endif //TENSORBASE_BENCHMARKGRAPHGPU_H