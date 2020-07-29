/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKGRAPHDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKGRAPHDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkGraph.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>
#include <TensorBase/ml/TensorOperationDefaultDevice.h>
#include <TensorBase/core/GraphGeneratorsDefaultDevice.h>
#include <TensorBase/core/GraphAlgorithmsDefaultDevice.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT>
	class SelectAndCountNodePropertyDefaultDevice: public SelectAndCountNodeProperty<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectAndCountNodeProperty<LabelsT, TensorT, Eigen::DefaultDevice>::SelectAndCountNodeProperty;
		void setLabelsValuesResult(Eigen::DefaultDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectAndCountNodePropertyDefaultDevice<LabelsT, TensorT>::setLabelsValuesResult(Eigen::DefaultDevice& device)
  { // TODO
  }

	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT>
	class SelectAndCountLinkPropertyDefaultDevice : public SelectAndCountLinkProperty<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectAndCountLinkProperty<LabelsT, TensorT, Eigen::DefaultDevice>::SelectAndCountLinkProperty;
		void setLabelsValuesResult(Eigen::DefaultDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectAndCountLinkPropertyDefaultDevice<LabelsT, TensorT>::setLabelsValuesResult(Eigen::DefaultDevice& device)
  { // TODO
  }

  /// Specialized SelectGraphAlgorithm for the DefaultDevice
  template<typename LabelsT, typename TensorT>
  class SelectGraphAlgorithmDefaultDevice: public virtual SelectGraphAlgorithm<LabelsT, TensorT, Eigen::DefaultDevice> {
  public:
    void setIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device) override;
    void setNodeAndLinkIds(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& link_ids, Eigen::DefaultDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectGraphAlgorithmDefaultDevice<LabelsT, TensorT>::setIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device)
  {
    // set the indices
    TensorDataDefaultDevice<LabelsT, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({
      tensor_collection->tables_.at("Graph_sparse_indices")->getDimSizeFromAxisName("1_links"),
      tensor_collection->tables_.at("Graph_sparse_indices")->getDimSizeFromAxisName("2_nodes") }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(indices_tmp);

    // set the weights
    TensorDataDefaultDevice<TensorT, 2> weights_tmp(Eigen::array<Eigen::Index, 2>({
      tensor_collection->tables_.at("Graph_weights")->getDimSizeFromAxisName("1_links"),
      tensor_collection->tables_.at("Graph_weights")->getDimSizeFromAxisName("2_weights") }));
    weights_tmp.setData();
    weights_tmp.syncHAndDData(device);
    weights = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(weights_tmp);

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
  inline void SelectGraphAlgorithmDefaultDevice<LabelsT, TensorT>::setNodeAndLinkIds(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& link_ids, Eigen::DefaultDevice& device)
  {
    KroneckerGraphGeneratorDefaultDevice<LabelsT, TensorT> graph_generator;
    graph_generator.getNodeAndLinkIds(0, indices->getDimensions().at(0), indices, node_ids, link_ids, device);
  }

	/// Specialized class for make the adjacency matrix for the DefaultDevice case
	template<typename LabelsT, typename TensorT>
	class SelectAdjacencyDefaultDeviceV : public virtual SelectAdjacency<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
    void makeAdjacencyMatrix(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& adjacency, Eigen::DefaultDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectAdjacencyDefaultDeviceV<LabelsT, TensorT>::makeAdjacencyMatrix(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& indices, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& weights, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& adjacency, Eigen::DefaultDevice& device)
  {
    IndicesAndWeightsToAdjacencyMatrixDefaultDevice<LabelsT, TensorT> to_adjacency;
    to_adjacency(node_ids, indices, weights, adjacency, device);
  }

  /// Specialized class for make the adjacency matrix for the DefaultDevice case
  template<typename LabelsT, typename TensorT>
  class SelectAdjacencyDefaultDevice : public SelectAdjacencyDefaultDeviceV<LabelsT, TensorT>, public SelectGraphAlgorithmDefaultDevice<LabelsT, TensorT> {
  public:
    void operator() (std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectAdjacencyDefaultDevice<LabelsT, TensorT>::operator()(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device)
  {
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> node_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> link_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> indices;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> weights;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> adjacency;
    this->setIndicesAndWeights(indices, weights, tensor_collection, device);
    this->setNodeAndLinkIds(indices, node_ids, link_ids, device);
    this->makeAdjacencyMatrix(node_ids, indices, weights, adjacency, device);
  }

	/// Specialized class for making the BFS tree for the DefaultDevice case
	template<typename LabelsT, typename TensorT>
	class SelectBFSDefaultDeviceV : public virtual SelectBFS<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		void makeBFSTree(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& adjacency, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& tree, Eigen::DefaultDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectBFSDefaultDeviceV<LabelsT, TensorT>::makeBFSTree(const std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>>& node_ids, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& adjacency, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& tree, Eigen::DefaultDevice& device)
  {
    BreadthFirstSearchDefaultDevice<int, float> breadth_first_search;
    breadth_first_search(0, node_ids, adjacency, tree, device);
  }

  /// Specialized class for making the BFS tree for the DefaultDevice case
  template<typename LabelsT, typename TensorT>
  class SelectBFSDefaultDevice : public SelectBFSDefaultDeviceV<LabelsT, TensorT>, public SelectAdjacencyDefaultDeviceV<LabelsT, TensorT>, public SelectGraphAlgorithmDefaultDevice<LabelsT, TensorT> {
  public:
    void operator() (std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectBFSDefaultDevice<LabelsT, TensorT>::operator()(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device)
  {
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> node_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> link_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> indices;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> weights;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> adjacency;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> tree;
    this->setIndicesAndWeights(indices, weights, tensor_collection, device);
    this->setNodeAndLinkIds(indices, node_ids, link_ids, device);
    this->makeAdjacencyMatrix(node_ids, indices, weights, adjacency, device);
    this->makeBFSTree(node_ids, adjacency, tree, device);
  }

  /// Specialized class for making the SSSP path lengths vector for the DefaultDevice case
  template<typename LabelsT, typename TensorT>
	class SelectSSSPDefaultDeviceV : public virtual SelectSSSP<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		void makeSSSPPathLengths(const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& tree, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& path_lengths, Eigen::DefaultDevice& device) override;
	};
  template<typename LabelsT, typename TensorT>
  inline void SelectSSSPDefaultDeviceV<LabelsT, TensorT>::makeSSSPPathLengths(const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& tree, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& path_lengths, Eigen::DefaultDevice& device)
  {
    SingleSourceShortestPathDefaultDevice<LabelsT, TensorT> sssp;
    sssp(tree, path_lengths, device);
  }

  /// Specialized class for making the SSSP path lengths vector for the DefaultDevice case
  template<typename LabelsT, typename TensorT>
  class SelectSSSPDefaultDevice : public SelectSSSPDefaultDeviceV<LabelsT, TensorT>, public SelectAdjacencyDefaultDeviceV<LabelsT, TensorT>, public SelectBFSDefaultDeviceV<LabelsT, TensorT>, public SelectGraphAlgorithmDefaultDevice<LabelsT, TensorT> {
  public:
    void operator() (std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectSSSPDefaultDevice<LabelsT, TensorT>::operator()(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device)
  {
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> node_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 1>> link_ids;
    std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> indices;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> weights;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> adjacency;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> tree;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>> path_lengths;
    this->setIndicesAndWeights(indices, weights, tensor_collection, device);
    this->setNodeAndLinkIds(indices, node_ids, link_ids, device);
    this->makeAdjacencyMatrix(node_ids, indices, weights, adjacency, device);
    this->makeBFSTree(node_ids, adjacency, tree, device);
    this->makeSSSPPathLengths(tree, path_lengths, device);
  }

  /// Helper structure to manage the KroneckerGraph data for the DefaultDevice
  template<typename KGLabelsT, typename KGTensorT>
  struct GraphManagerHelperDefaultDevice: public GraphManagerHelper<KGLabelsT, KGTensorT, Eigen::DefaultDevice> {
    void makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::DefaultDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT>
  inline void GraphManagerHelperDefaultDevice<KGLabelsT, KGTensorT>::makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::DefaultDevice& device) {
    KroneckerGraphGeneratorDefaultDevice<KGLabelsT, KGTensorT> graph_generator;
    graph_generator.makeKroneckerGraph(scale, edge_factor, this->kronecker_graph_indices_, this->kronecker_graph_weights_, device);
    graph_generator.getNodeAndLinkIds(0, this->kronecker_graph_indices_->getDimensions().at(0), this->kronecker_graph_indices_, this->kronecker_graph_node_ids_, this->kronecker_graph_link_ids_, device);
  }

	/*
	@class Specialized `GraphManagerSparseIndices` for the DefaultDevice case
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	class GraphManagerSparseIndicesDefaultDevice : public GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice>::GraphManagerSparseIndices;
		void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) override;
		void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) override;
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	void GraphManagerSparseIndicesDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) {
		TensorDataDefaultDevice<LabelsT, 2> tmp(dimensions);
		tmp.setData();
		tmp.syncHAndDData(device);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(tmp);
	}
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	void GraphManagerSparseIndicesDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) {
		TensorDataDefaultDevice<TensorT, 2> tmp(dimensions);
		tmp.setData();
		tmp.syncHAndDData(device);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(tmp);
	}

  /*
  @class Specialized `GraphManagerWeights` for the DefaultDevice case
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  class GraphManagerWeightsDefaultDevice : public GraphManagerWeights<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice> {
  public:
    using GraphManagerWeights<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice>::GraphManagerWeights;
    void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) override;
    void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerWeightsDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) {
    TensorDataDefaultDevice<LabelsT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerWeightsDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) {
    TensorDataDefaultDevice<TensorT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(tmp);
  }

  /*
  @class Specialized `GraphManagerNodeProperty` for the DefaultDevice case
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  class GraphManagerNodePropertyDefaultDevice : public GraphManagerNodeProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice> {
  public:
    using GraphManagerNodeProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice>::GraphManagerNodeProperty;
    void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) override;
    void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) override;
    bool setNodeIds(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::DefaultDevice, 2>>& kronecker_graph_indices, Eigen::DefaultDevice& device) override;
    void setLabels(Eigen::DefaultDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerNodePropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) {
    TensorDataDefaultDevice<LabelsT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerNodePropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) {
    TensorDataDefaultDevice<TensorT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  inline bool GraphManagerNodePropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::setNodeIds(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::DefaultDevice, 2>>& kronecker_graph_indices, Eigen::DefaultDevice& device)
  {
    // get all of the unique nodes up to this point
    KroneckerGraphGeneratorDefaultDevice<KGLabelsT, KGTensorT> graph_generator;
    std::shared_ptr<TensorData<KGLabelsT, Eigen::DefaultDevice, 1>> link_ids; // temporary
    std::shared_ptr<TensorData<KGLabelsT, Eigen::DefaultDevice, 1>> node_ids; // temporary
    graph_generator.getNodeAndLinkIds(0, offset + span, kronecker_graph_indices, node_ids, link_ids, device);

    // allocate and extract out the new unique node ids
    const int node_id_size = node_ids->getTensorSize() - this->nodes_added_cumulative_;
    if (node_id_size > 0) {
      TensorDataDefaultDevice<KGLabelsT, 1> tmp(Eigen::array<Eigen::Index, 1>({ node_id_size }));
      tmp.setData();
      tmp.syncHAndDData(device);
      Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 1>> tmp_values(tmp.getDataPointer().get(), tmp.getDimensions());
      Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 1>> node_ids_values(node_ids->getDataPointer().get(), node_ids->getDimensions());
      tmp_values.device(device) = node_ids_values.slice(Eigen::array<Eigen::Index, 1>({ this->nodes_added_cumulative_ }), Eigen::array<Eigen::Index, 1>({ node_id_size }));

      // assign the values
      node_ids = std::make_shared<TensorDataDefaultDevice<KGLabelsT, 1>>(tmp);
      this->nodes_added_cumulative_ += node_id_size;
      return true;
    }
    else return false;
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  inline void GraphManagerNodePropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::setLabels(Eigen::DefaultDevice& device)
  {
    TensorDataDefaultDevice<TensorT, 1> tmp(node_ids->getDimensions());
    tmp.setData();

    // Cycle through each color node;
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tmp_values(tmp.getDataPointer().get(), tmp.getDimensions());
    Eigen::Tensor<TensorT, 2> labels(1, 5);
    labels.setValues({ {TensorT("white"), TensorT("black"), TensorT("red"), TensorT("blue"), TensorT("green")} });
    const int bcast_length = node_ids->getTensorSize() / 5 + 1;
    auto labels_bcast_reshape = labels.shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 })).broadcast(Eigen::array<Eigen::Index, 2>({ 1, bcast_length })).reshape(Eigen::array<Eigen::Index, 1>({ bcast_length * 5 }));
    tmp_values = labels_bcast_reshape.slice(Eigen::array<Eigen::Index, 1>({ 0 }), Eigen::array<Eigen::Index, 1>({ (int)tmp.getTensorSize() }));

    // Sync the data to the device and assign the data
    tmp.syncHAndDData(device);
    this->labels_ = std::make_shared<TensorDataDefaultDevice<TensorT, 1>>(tmp);
  }

  /*
  @class Specialized `GraphManagerLinkProperty` for the DefaultDevice case
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  class GraphManagerLinkPropertyDefaultDevice : public GraphManagerLinkProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice> {
  public:
    using GraphManagerLinkProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice>::GraphManagerLinkProperty;
    void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) override;
    void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) override;
    void setLabels(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::DefaultDevice, 1>>& kronecker_graph_link_ids, Eigen::DefaultDevice& device) override;
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerLinkPropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) {
    TensorDataDefaultDevice<LabelsT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  void GraphManagerLinkPropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) {
    TensorDataDefaultDevice<TensorT, 2> tmp(dimensions);
    tmp.setData();
    tmp.syncHAndDData(device);
    values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(tmp);
  }
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
  inline void GraphManagerLinkPropertyDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::setLabels(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, Eigen::DefaultDevice, 1>>& kronecker_graph_link_ids, Eigen::DefaultDevice& device)
  {
    TensorDataDefaultDevice<TensorT, 1> tmp(Eigen::array<Eigen::Index, 1>({span}));
    tmp.setData();

    // Dashed link every third index
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tmp_values(tmp.getDataPointer().get(), tmp.getDimensions());
    Eigen::Tensor<int, 1> indices(span);
    indices = indices.constant(1).cumsum(0) + indices.constant(offset - 1);
    auto indices_mod = indices - (indices.constant(3) * (indices / indices.constant(3))).eval(); // a mod n = a - (n * int(a/n))
    tmp_values = (indices_mod == indices.constant(0)).select(tmp_values.constant(TensorT("dashed")), tmp_values.constant(TensorT("solid")));
    
    // Sync the data to the device and assign the output
    tmp.syncHAndDData(device);
    this->labels_ = std::make_shared<TensorDataDefaultDevice<TensorT, 1>>(tmp);
  }

  /*
  @class A class for running 1 line insertion, deletion, and update benchmarks
  */
  class BenchmarkGraph1LinkDefaultDevice : public BenchmarkGraph1Link<int, float, Eigen::DefaultDevice> {
  protected:
    void _makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::DefaultDevice& device);
    void _insert1Link(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1Link`
    void _update1Link(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1Link`
    void _delete1Link(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1Link`
    int _selectAndCountNodeProperty(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndCountNodeProperty`
    int _selectAndCountLinkProperty(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndCountLinkProperty`
    float _selectAdjacency(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAdjacency`
    float _selectBFS(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectBFS`
    float _selectSSSP(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectSSSP`
  };
  inline void BenchmarkGraph1LinkDefaultDevice::_makeKroneckerGraph(const int& scale, const int& edge_factor, Eigen::DefaultDevice& device)
  {
    GraphManagerHelperDefaultDevice<int, float> gmh;
    gmh.makeKroneckerGraph(scale, edge_factor, device);
    this->graph_manager_helper_ = std::make_shared<GraphManagerHelperDefaultDevice<int, float>>(gmh);
  }
  void BenchmarkGraph1LinkDefaultDevice::_insert1Link(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const {
    GraphManagerSparseIndicesDefaultDevice<int, float, int, int> graph_manager_sparse_indices(false);
    GraphManagerWeightsDefaultDevice<int, float, int, float> graph_manager_weights(false);
    GraphManagerNodePropertyDefaultDevice<int, float, int, TensorArray8<char>> graph_manager_node_property(false);
    GraphManagerLinkPropertyDefaultDevice<int, float, int, TensorArray8<char>> graph_manager_link_property(false);
    int span = std::pow(2, scale);
    for (int i = 0; i < std::pow(2, scale) * edge_factor; i += span) {
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_sparse_indices_ptr;
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> values_sparse_indices_ptr;
      graph_manager_sparse_indices.getInsertData(i, span, labels_sparse_indices_ptr, values_sparse_indices_ptr, 
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_,
        device);
      TensorAppendToAxis<int, int, Eigen::DefaultDevice, 2> appendToAxis_sparse_indices("Graph_sparse_indices", "1_links", labels_sparse_indices_ptr, values_sparse_indices_ptr);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> appendToAxis_sparse_indices_ptr = std::make_shared<TensorAppendToAxis<int, int, Eigen::DefaultDevice, 2>>(appendToAxis_sparse_indices);
      transaction_manager.executeOperation(appendToAxis_sparse_indices_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_weights_ptr;
      std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> values_weights_ptr;
      graph_manager_weights.getInsertData(i, span, labels_weights_ptr, values_weights_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      TensorAppendToAxis<int, float, Eigen::DefaultDevice, 2> appendToAxis_weights("Graph_weights", "1_links", labels_weights_ptr, values_weights_ptr);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> appendToAxis_weights_ptr = std::make_shared<TensorAppendToAxis<int, float, Eigen::DefaultDevice, 2>>(appendToAxis_weights);
      transaction_manager.executeOperation(appendToAxis_weights_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_node_property_ptr;
      std::shared_ptr<TensorData<TensorArray8<char>, Eigen::DefaultDevice, 2>> values_node_property_ptr;
      graph_manager_node_property.getInsertData(i, span, labels_node_property_ptr, values_node_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      if (labels_node_property_ptr != nullptr && labels_node_property_ptr->getTensorSize() > 0) {
        TensorAppendToAxis<int, TensorArray8<char>, Eigen::DefaultDevice, 2> appendToAxis_node_property("Graph_node_property", "1_nodes", labels_node_property_ptr, values_node_property_ptr);
        std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> appendToAxis_node_property_ptr = std::make_shared<TensorAppendToAxis<int, TensorArray8<char>, Eigen::DefaultDevice, 2>>(appendToAxis_node_property);
        transaction_manager.executeOperation(appendToAxis_node_property_ptr, device);
      }
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_link_property_ptr;
      std::shared_ptr<TensorData<TensorArray8<char>, Eigen::DefaultDevice, 2>> values_link_property_ptr;
      graph_manager_link_property.getInsertData(i, span, labels_link_property_ptr, values_link_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      TensorAppendToAxis<int, TensorArray8<char>, Eigen::DefaultDevice, 2> appendToAxis_link_property("Graph_link_property", "1_links", labels_link_property_ptr, values_link_property_ptr);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> appendToAxis_link_property_ptr = std::make_shared<TensorAppendToAxis<int, TensorArray8<char>, Eigen::DefaultDevice, 2>>(appendToAxis_link_property);
      transaction_manager.executeOperation(appendToAxis_link_property_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
  }
  void BenchmarkGraph1LinkDefaultDevice::_update1Link(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const {
    GraphManagerSparseIndicesDefaultDevice<int, float, int, int> graph_manager_sparse_indices(true);
    GraphManagerWeightsDefaultDevice<int, float, int, float> graph_manager_weights(true);
    GraphManagerNodePropertyDefaultDevice<int, float, int, TensorArray8<char>> graph_manager_node_property(true);
    GraphManagerLinkPropertyDefaultDevice<int, float, int, TensorArray8<char>> graph_manager_link_property(true);
    int span = std::pow(2, scale);
    for (int i = 0; i < std::pow(2, scale) * edge_factor; i += span) {
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_sparse_indices_ptr;
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> values_sparse_indices_ptr;
      graph_manager_sparse_indices.getInsertData(i, span, labels_sparse_indices_ptr, values_sparse_indices_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_sparse_indices(labels_sparse_indices_ptr, "Graph_sparse_indices", "1_links");
      TensorUpdateValues<int, Eigen::DefaultDevice, 2> updateValues_sparse_indices("Graph_sparse_indices", selectClause_sparse_indices, values_sparse_indices_ptr);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> updateValues_sparse_indices_ptr = std::make_shared<TensorUpdateValues<int, Eigen::DefaultDevice, 2>>(updateValues_sparse_indices);
      transaction_manager.executeOperation(updateValues_sparse_indices_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_weights_ptr;
      std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> values_weights_ptr;
      graph_manager_weights.getInsertData(i, span, labels_weights_ptr, values_weights_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_weights(labels_weights_ptr, "Graph_weights", "1_links");
      TensorUpdateValues<float, Eigen::DefaultDevice, 2> updateValues_weights("Graph_weights", selectClause_weights, values_weights_ptr);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> updateValues_weights_ptr = std::make_shared<TensorUpdateValues<float, Eigen::DefaultDevice, 2>>(updateValues_weights);
      transaction_manager.executeOperation(updateValues_weights_ptr, device);
      // Update the nodes outside the loop
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_link_property_ptr;
      std::shared_ptr<TensorData<TensorArray8<char>, Eigen::DefaultDevice, 2>> values_link_property_ptr;
      graph_manager_link_property.getInsertData(i, span, labels_link_property_ptr, values_link_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_link_property(labels_link_property_ptr, "Graph_link_property", "1_links");
      TensorUpdateValues<TensorArray8<char>, Eigen::DefaultDevice, 2> updateValues_link_property("Graph_link_property", selectClause_link_property, values_link_property_ptr);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> updateValues_link_property_ptr = std::make_shared<TensorUpdateValues<TensorArray8<char>, Eigen::DefaultDevice, 2>>(updateValues_link_property);
      transaction_manager.executeOperation(updateValues_link_property_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_node_property_ptr;
    std::shared_ptr<TensorData<TensorArray8<char>, Eigen::DefaultDevice, 2>> values_node_property_ptr;
    graph_manager_node_property.getInsertData(0, std::pow(2, scale) * edge_factor, labels_node_property_ptr, values_node_property_ptr,
    this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
    SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_node_property(labels_node_property_ptr, "Graph_node_property", "1_nodes");
    TensorUpdateValues<TensorArray8<char>, Eigen::DefaultDevice, 2> updateValues_node_property("Graph_node_property", selectClause_node_property, values_node_property_ptr);
    std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> updateValues_node_property_ptr = std::make_shared<TensorUpdateValues<TensorArray8<char>, Eigen::DefaultDevice, 2>>(updateValues_node_property);
    transaction_manager.executeOperation(updateValues_node_property_ptr, device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
  }
  void BenchmarkGraph1LinkDefaultDevice::_delete1Link(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const {
    GraphManagerSparseIndicesDefaultDevice<int, float, int, int> graph_manager_sparse_indices(true);
    GraphManagerWeightsDefaultDevice<int, float, int, float> graph_manager_weights(true);
    GraphManagerNodePropertyDefaultDevice<int, float, int, TensorArray8<char>> graph_manager_node_property(true);
    GraphManagerLinkPropertyDefaultDevice<int, float, int, TensorArray8<char>> graph_manager_link_property(true);
    int span = std::pow(2, scale);
    for (int i = 0; i < std::pow(2, scale) * edge_factor - 1; i += span) {
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_sparse_indices_ptr;
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> values_sparse_indices_ptr;
      graph_manager_sparse_indices.getInsertData(i, span, labels_sparse_indices_ptr, values_sparse_indices_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_sparse_indices(labels_sparse_indices_ptr, "Graph_sparse_indices", "1_links");
      TensorDeleteFromAxisDefaultDevice<int, int, 2> tensorDelete_sparse_indices("Graph_sparse_indices", "1_links", selectClause_sparse_indices);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_sparse_indices_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<int, int, 2>>(tensorDelete_sparse_indices);
      transaction_manager.executeOperation(tensorDelete_sparse_indices_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_weights_ptr;
      std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> values_weights_ptr;
      graph_manager_weights.getInsertData(i, span, labels_weights_ptr, values_weights_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_weights(labels_weights_ptr, "Graph_weights", "1_links");
      TensorDeleteFromAxisDefaultDevice<int, float, 2> tensorDelete_weights("Graph_weights", "1_links", selectClause_weights);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_weights_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<int, float, 2>>(tensorDelete_weights);
      transaction_manager.executeOperation(tensorDelete_weights_ptr, device);
      // Delete the nodes outside the loop
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_link_property_ptr;
      std::shared_ptr<TensorData<TensorArray8<char>, Eigen::DefaultDevice, 2>> values_link_property_ptr;
      graph_manager_link_property.getInsertData(i, span, labels_link_property_ptr, values_link_property_ptr,
        this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
      SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_link_property(labels_link_property_ptr, "Graph_link_property", "1_links");
      TensorDeleteFromAxisDefaultDevice<int, TensorArray8<char>, 2> tensorDelete_link_property("Graph_link_property", "1_links", selectClause_link_property);
      std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_link_property_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<int, TensorArray8<char>, 2>>(tensorDelete_link_property);
      transaction_manager.executeOperation(tensorDelete_link_property_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }      
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_node_property_ptr;
    std::shared_ptr<TensorData<TensorArray8<char>, Eigen::DefaultDevice, 2>> values_node_property_ptr;
    graph_manager_node_property.getInsertData(0, std::pow(2, scale) * edge_factor, labels_node_property_ptr, values_node_property_ptr,
    this->graph_manager_helper_->kronecker_graph_indices_, this->graph_manager_helper_->kronecker_graph_weights_, this->graph_manager_helper_->kronecker_graph_node_ids_, this->graph_manager_helper_->kronecker_graph_link_ids_, device);
    SelectGraphNodeLinkIDs<int, Eigen::DefaultDevice> selectClause_node_property(labels_node_property_ptr, "Graph_node_property", "1_nodes");
    TensorDeleteFromAxisDefaultDevice<int, TensorArray8<char>, 2> tensorDelete_node_property("Graph_node_property", "1_nodes", selectClause_node_property);
    std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_node_property_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<int, TensorArray8<char>, 2>>(tensorDelete_node_property);
    transaction_manager.executeOperation(tensorDelete_node_property_ptr, device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
  }
  inline int BenchmarkGraph1LinkDefaultDevice::_selectAndCountNodeProperty(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndCountNodePropertyDefaultDevice<int, TensorArray8<char>> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_and_sum.result_;
  }
  inline int BenchmarkGraph1LinkDefaultDevice::_selectAndCountLinkProperty(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndCountNodePropertyDefaultDevice<int, TensorArray8<char>> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_and_sum.result_;
  }
  inline float BenchmarkGraph1LinkDefaultDevice::_selectAdjacency(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAdjacencyDefaultDevice<int, float> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.adjacency_->syncHAndDData(device);
    return select_and_sum.adjacency_->getData()(0, 0);
  }
  inline float BenchmarkGraph1LinkDefaultDevice::_selectBFS(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectBFSDefaultDevice<int, float> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.tree_->syncHAndDData(device);
    return select_and_sum.tree_->getData()(0, 0);
  }
  inline float BenchmarkGraph1LinkDefaultDevice::_selectSSSP(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectSSSPDefaultDevice<int, float> select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.path_lengths_->syncHAndDData(device);
    return select_and_sum.path_lengths_->getData()(0);
  }

  class GraphTensorCollectionGeneratorDefaultDevice : public GraphTensorCollectionGenerator<Eigen::DefaultDevice> {
  public:
    std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> makeTensorCollection(const int& scale, const int& edge_factor, const double& shard_span_perc, Eigen::DefaultDevice& device) const override;
  };
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> GraphTensorCollectionGeneratorDefaultDevice::makeTensorCollection(const int& scale, const int& edge_factor, const double& shard_span_perc, Eigen::DefaultDevice& device) const
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
    Eigen::Tensor<TensorArray8<char>, 2> labels_1b(1, 2), labels_1c(1, 1), labels_3(1, 1);;
    labels_1b.setValues({ { TensorArray8<char>("node_in"), TensorArray8<char>("node_out")} });
    labels_1c.setValues({ { TensorArray8<char>("weights")} });
    labels_3.setValues({ { TensorArray8<char>("label")} });

    // Setup the tables
    std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(TensorTableDefaultDevice<int, 2>("Graph_sparse_indices"));
    std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("2_nodes", dimensions_2b, labels_1b));
    std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1_links", 1, 0));
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
    std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 2>> table_2_ptr = std::make_shared<TensorTableDefaultDevice<float, 2>>(TensorTableDefaultDevice<float, 2>("Graph_weights"));
    std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_2_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("2_weights", dimensions_2a, labels_1c));
    std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>> table_2_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1_links", 1, 0));
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
    std::shared_ptr<TensorTable<TensorArray8<char>, Eigen::DefaultDevice, 2>> table_3_ptr = std::make_shared<TensorTableDefaultDevice<TensorArray8<char>, 2>>(TensorTableDefaultDevice<TensorArray8<char>, 2>("Graph_node_property"));
    std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_3_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("2_property", dimensions_3b, labels_3));
    std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>> table_3_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1_nodes", 1, 0));
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
    std::shared_ptr<TensorTable<TensorArray8<char>, Eigen::DefaultDevice, 2>> table_4_ptr = std::make_shared<TensorTableDefaultDevice<TensorArray8<char>, 2>>(TensorTableDefaultDevice<TensorArray8<char>, 2>("Graph_link_property"));
    std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_4_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("2_property", dimensions_1a, labels_3));
    std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>> table_4_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1_links", 1, 0));
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
    auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
    collection_1_ptr->addTensorTable(table_1_ptr, "Graph");
    collection_1_ptr->addTensorTable(table_2_ptr, "Graph");
    collection_1_ptr->addTensorTable(table_3_ptr, "Graph");
    collection_1_ptr->addTensorTable(table_4_ptr, "Graph");
    // TODO: linking of axes
    return collection_1_ptr;
  }
};
#endif //TENSORBASE_BENCHMARKGRAPHDEFAULTDEVICE_H