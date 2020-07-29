/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKGRAPH_H
#define TENSORBASE_BENCHMARKGRAPH_H

#include <ctime> // time format
#include <chrono> // current time
#include <math.h> // std::pow
#include <random> // random number generator

#include <unsupported/Eigen/CXX11/Tensor>

#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/ml/TensorSelect.h>
#include <TensorBase/core/GraphGenerators.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/// The base select Functor for the Graph node or link IDs
	template<typename LabelsT, typename DeviceT>
	class SelectGraphNodeLinkIDs {
	public:
		SelectGraphNodeLinkIDs(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, const std::string& table_name, const std::string& axis_name) : select_labels_(select_labels), table_name_(table_name), axis_name_(axis_name) {};
		~SelectGraphNodeLinkIDs() = default;
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels_;
		std::string table_name_;
		std::string axis_name_;
	};
	template<typename LabelsT, typename DeviceT>
	inline void SelectGraphNodeLinkIDs<LabelsT, DeviceT>::operator()(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
	{
		SelectClause<LabelsT, DeviceT> select_clause1(this->table_name_, this->axis_name_, this->select_labels_);
		TensorSelect tensorSelect;
		tensorSelect.selectClause(tensor_collection, select_clause1, device);
	}

	/*
	@brief Class for selecting and counting nodes with particular properties
		Count the number of "white" in nodes that are connected to a "black" out node	

	NOTES:
		- Need to implement the MapClause in order to run the full query without a lot of hacks
		- Current implementation only counts the number of "white" nodes
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAndCountNodeProperty {
	public:
		int result_ = 0; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	inline void SelectAndCountNodeProperty<LabelsT, TensorT, DeviceT>::operator()(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
	{
		setLabelsValuesResult(device);

		// Select all white nodes
		WhereClause<LabelsT, TensorT, DeviceT> where_clause1("Graph_node_property", "2_property", select_labels_, select_values_, logicalComparitors::EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);
		TensorSelect tensorSelect;
		tensorSelect.whereClause(tensor_collection, where_clause1, device);
		tensorSelect.applySelect(tensor_collection, { "Graph_node_property" }, { "Graph_node_property_tmp" }, device);
		tensor_collection->tables_.at("Graph_node_property")->resetIndicesView(device);

		// Get their IDs
		result_ = tensor_collection->tables_.at("Graph_node_property_tmp")->getDimSizeFromAxisName("1_nodes");
		tensor_collection->removeTensorTable("Graph_node_property_tmp");
	}

	/*
	@brief Class for selecting and counting links with particular properties
		Count the number of "dashed" links that connect two "blue" nodes

	NOTES:
		- Need to implement the JoinClause in order to run the full query without a lot of hacks
		- Current implementation only counts the number of "dashed" links
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAndCountLinkProperty {
	public:
		int result_ = 0; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	inline void SelectAndCountLinkProperty<LabelsT, TensorT, DeviceT>::operator()(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
	{
		setLabelsValuesResult(device);

		// Select all white nodes
		WhereClause<LabelsT, TensorT, DeviceT> where_clause1("Graph_link_property", "2_property", select_labels_, select_values_, logicalComparitors::EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);
		TensorSelect tensorSelect;
		tensorSelect.whereClause(tensor_collection, where_clause1, device);
		tensorSelect.applySelect(tensor_collection, { "Graph_link_property" }, { "Graph_link_property_tmp" }, device);
		tensor_collection->tables_.at("Graph_link_property")->resetIndicesView(device);

		// Get their IDs
		result_ = tensor_collection->tables_.at("Graph_link_property_tmp")->getDimSizeFromAxisName("1_links");
		tensor_collection->removeTensorTable("Graph_link_property_tmp");
	}

	/*
	@brief Base class for graph algorithms
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	struct GraphAlgorithmHelper {
		virtual ~GraphAlgorithmHelper() = default;
		virtual void setIndicesAndWeights(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& weights, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
		virtual void setNodeAndLinkIds(const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& node_ids, std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& link_ids, DeviceT& device) = 0;
	};

	/*
	@brief Class for making the adjacency matrix from a sparse representation
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	struct SelectAdjacency {
		virtual void operator() (std::shared_ptr<TensorCollection<DeviceT>> & tensor_collection, DeviceT & device) = 0;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> adjacency_; ///< The results of the query
	};

	/*
	@brief Class for running the breadth-first search algorithm
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	struct SelectBFS {
		virtual void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> tree_;
	};

	/*
	@brief Class for running the single source shortest path search algorithm
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	struct SelectSSSP {
		virtual void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> path_lengths_;
	};

	/// Helper structure to manage the KroneckerGraph data
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	struct GraphManagerHelper {
		virtual void makeKroneckerGraph(const int& scale, const int& edge_factor, DeviceT& device) = 0;
		std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>> kronecker_graph_indices_;
		std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>> kronecker_graph_weights_;
		std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>> kronecker_graph_node_ids_;
		std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>> kronecker_graph_link_ids_;
	};

	/*
	@brief Class for managing the generation of data for the Graph
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT, int NDim>
	class GraphManager {
	public:
    GraphManager(const bool& use_random_values = false) : use_random_values_(use_random_values){};
		~GraphManager() = default;
		virtual void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr, 
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
			const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids,
			DeviceT& device) = 0;
		virtual void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, DeviceT& device) = 0;
		virtual void makeValuesPtr(const Eigen::array<Eigen::Index, NDim>& dimensions, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr, DeviceT& device) = 0;
    void setUseRandomValues(const bool& use_random_values) { use_random_values_ = use_random_values; }
	protected:
		bool use_random_values_;
	};

	/*
	@class Specialized `GraphManager` for generating sparse graph representation
		that includes input and output `node_id`s matched to `link_id`s

		NOTES: original idea
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerSparse : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
		using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
			const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device);
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparse<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
		const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device) {
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 3, span }; // node_in, node_out, link_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by weights
		this->makeLabelsPtr(labels_dims, labels_ptr);
		this->makeValuesPtr(values_dims, values_ptr);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> indices_values(kronecker_graph_indices->getDataPointer().get(), kronecker_graph_indices->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(kronecker_graph_link_ids->getDataPointer().get(), 1, (int)kronecker_graph_link_ids->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 2, span })).device(device) = indices_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 2 })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 }));
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 2, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span }));

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> weights_values(kronecker_graph_weights->getDataPointer().get(), kronecker_graph_weights->getDimensions());
		values_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).device(device) = weights_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 }));
	}

	/*
	@class Specialized `GraphManager` for generating sparse graph representation
    that includes input and output `node_id`s matched to `link_id`s
	*/
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerSparseIndices : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
    using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
			const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device);
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
		const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device) {
		// Make the labels and values
    Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // link_id
    Eigen::array<Eigen::Index, 2> values_dims = { span, 2 }; // indices by [node_in, node_out]
    this->makeLabelsPtr(labels_dims, labels_ptr, device);
    this->makeValuesPtr(values_dims, values_ptr, device);

    // Assign the labels data
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(kronecker_graph_link_ids->getDataPointer().get(), 1, (int)kronecker_graph_link_ids->getTensorSize());
    labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
      Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

    // Assign the values data
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> indices_values(kronecker_graph_indices->getDataPointer().get(), kronecker_graph_indices->getDimensions());
		values_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ span, 2 })).device(device) = indices_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 2 })).cast<TensorT>();
	}

	/*
	@class Specialized `GraphManager` for generating sparse graph representation
		that includes `weights` matched to `link_id`s
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerWeights : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
		using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
			const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device);
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerWeights<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
		const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device) {
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // link_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by weights
		this->makeLabelsPtr(labels_dims, labels_ptr, device);
		this->makeValuesPtr(values_dims, values_ptr, device);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(kronecker_graph_link_ids->getDataPointer().get(), 1, (int)kronecker_graph_link_ids->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> weights_values(kronecker_graph_weights->getDataPointer().get(), kronecker_graph_weights->getDimensions());
		values_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).device(device) = weights_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).cast<TensorT>();
	}

  /*
  @brief Specialized `GraphManager` for generating the node properties
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
  class GraphManagerNodeProperty : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
  public:
    using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
			const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device);
		virtual void setLabels(DeviceT& device) = 0;
		virtual bool setNodeIds(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices, DeviceT& device) = 0;
  protected:
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_ = nullptr;
		std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>> node_ids_ = nullptr;
		int nodes_added_cumulative_ = 0;
  };
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	inline void GraphManagerNodeProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
		const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device)
	{		
		// Set the values on the device for transfer
		const bool add_node_ids = setNodeIds(offset, span, kronecker_graph_indices, device);
		if (add_node_ids) {
			setLabels(device);

			// Make the labels and values
			Eigen::array<Eigen::Index, 2> labels_dims = { 1, (int)this->node_ids_->getTensorSize() }; // node_id
			Eigen::array<Eigen::Index, 2> values_dims = { (int)this->node_ids_->getTensorSize(), 1 }; // indices by "label"
			this->makeLabelsPtr(labels_dims, labels_ptr, device);
			this->makeValuesPtr(values_dims, values_ptr, device);

			// Assign the labels data
			Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
			Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> node_ids_values(this->node_ids_->getDataPointer().get(), 1, (int)this->node_ids_->getTensorSize());
			labels_values.device(device) = node_ids_values.cast<LabelsT>();

			// Assign the values data
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> property_values(this->labels_->getDataPointer().get(), (int)this->labels_->getTensorSize(), 1);
			values_values.device(device) = property_values;
		}
		else {
			labels_ptr = nullptr;
			values_ptr = nullptr;
		}
	}

  /*
  @brief Specialized `GraphManager` for generating the link properties
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerLinkProperty : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
		using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
			const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
			const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device);
		virtual void setLabels(const int& offset, const int& span, const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_ = nullptr;
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	inline void GraphManagerLinkProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>>& kronecker_graph_indices,
		const std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>>& kronecker_graph_weights,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_node_ids,
		const std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>>& kronecker_graph_link_ids, DeviceT& device)
	{
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // link_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by "label"
		this->makeLabelsPtr(labels_dims, labels_ptr, device);
		this->makeValuesPtr(values_dims, values_ptr, device);

		// Set the values on the device for transfer
		setLabels(offset, span, kronecker_graph_link_ids, device);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(kronecker_graph_link_ids->getDataPointer().get(), 1, (int)kronecker_graph_link_ids->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> property_values(this->labels_->getDataPointer().get(), this->labels_->getTensorSize(), 1);
		values_values.device(device) = property_values;
	}
	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	class BenchmarkGraph1Link {
	public:
		BenchmarkGraph1Link() = default;
		~BenchmarkGraph1Link() = default;
		/*
		@brief insert 1 time-link at a time

		@param[in, out] transaction_manager
		@param[in] data_size
		@param[in] device

		@returns A string with the total time of the benchmark in milliseconds
		*/
		std::string insert1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
		std::string update1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
		std::string delete1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
    std::pair<std::string, int> selectAndCountNodeProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
    std::pair<std::string, int> selectAndCountLinkProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
    std::pair<std::string, float> selectAdjacency(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
		std::pair<std::string, float> selectBFS(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
		std::pair<std::string, float> selectSSSP(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const;
		std::string makeKroneckerGraph(const int& scale, const int& edge_factor, DeviceT& device);
		std::shared_ptr<GraphManagerHelper<KGLabelsT, KGTensorT, DeviceT>> graph_manager_helper_; ///< Kronecker graph generator
	protected:
		virtual void _makeKroneckerGraph(const int& scale, const int& edge_factor, DeviceT& device) = 0;
		virtual void _insert1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1Link`
		virtual void _update1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1Link`
		virtual void _delete1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1Link`
    virtual int _selectAndCountNodeProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndCountNodeProperty`
    virtual int _selectAndCountLinkProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndCountLinkProperty`
    virtual float _selectAdjacency(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAdjacency`
		virtual float _selectBFS(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectBFS`
		virtual float _selectSSSP(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectSSSP`
	};
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	inline std::string BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::makeKroneckerGraph(const int& scale, const int& edge_factor, DeviceT& device)
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_makeKroneckerGraph(scale, edge_factor, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	std::string BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::insert1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_insert1Link(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	std::string BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::update1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_update1Link(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	std::string BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::delete1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_delete1Link(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
  template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
  inline std::pair<std::string, int> BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::selectAndCountNodeProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const int result = _selectAndCountNodeProperty(transaction_manager, scale, edge_factor, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
  template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
  inline std::pair<std::string, int> BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::selectAndCountLinkProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const int result = _selectAndCountLinkProperty(transaction_manager, scale, edge_factor, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
  template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
  inline std::pair<std::string, float> BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::selectAdjacency(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const float result = _selectAdjacency(transaction_manager, scale, edge_factor, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	inline std::pair<std::string, float> BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::selectBFS(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		const float result = _selectBFS(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return std::pair(milli_time, result);
	}
	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	inline std::pair<std::string, float> BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>::selectSSSP(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		const float result = _selectSSSP(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return std::pair(milli_time, result);
	}

	/*
	@brief Simulate a typical graph with mixed column types and mixed entry dimensions
	*/
	template<typename DeviceT>
	class GraphTensorCollectionGenerator {
	public:
		GraphTensorCollectionGenerator() = default;
		~GraphTensorCollectionGenerator() = default;
		virtual std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& scale, const int& edge_factor, const double& shard_span_perc, DeviceT& device) const = 0;
	};

	template<typename KGLabelsT, typename KGTensorT, typename DeviceT>
	static void runBenchmarkGraph(const std::string& data_dir, const int& scale, const int& edge_factor, const bool& in_memory, const double& shard_span_perc,
		const BenchmarkGraph1Link<KGLabelsT, KGTensorT, DeviceT>& benchmark_1_link,
		const GraphTensorCollectionGenerator<DeviceT>& tensor_collection_generator, DeviceT& device) {
		std::cout << "Starting insert/delete/update Graph benchmarks for scale=" << scale << ", edge_factor=" << edge_factor << ", in_memory=" << in_memory << ", and shard_span_perc=" << shard_span_perc << std::endl;

		// Make the nD TensorTables
		std::shared_ptr<TensorCollection<DeviceT>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, device);

		// Setup the transaction manager
		TransactionManager<DeviceT> transaction_manager;
		transaction_manager.setMaxOperations(data_size + 1);

		// Run the table through the benchmarks
		transaction_manager.setTensorCollection(n_dim_tensor_collection);
		std::cout << "Graph Kronecker graph generation took " << benchmark_1_link.makeKroneckerGraph(scale, edge_factor, device) << " milliseconds." << std::endl;
		std::cout << "Graph link insertion took " << benchmark_1_link.insert1Link(transaction_manager, scale, edge_factor, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Graph select and count [...] nodes took " << (benchmark_1_link.selectAndCountNodeProperty(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Graph select and count [...] links took " << (benchmark_1_link.selectAndCountLinkProperty(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Graph select and make the adjacency matrix " << (benchmark_1_link.selectAdjacency(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Graph select and perform a breadth-first search " << (benchmark_1_link.selectBFS(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Graph select and perform a single-source shortest path search " << (benchmark_1_link.selectSSSP(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Graph link update took " << benchmark_1_link.update1Link(transaction_manager, scale, edge_factor, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Graph link deletion took " << benchmark_1_link.delete1Link(transaction_manager, scale, edge_factor, in_memory, device) << " milliseconds." << std::endl;
	}

	///Parse the command line arguments
	static void parseCmdArgsGraph(const int& argc, char** argv, std::string& data_dir, int& scale, int& edge_factor, bool& in_memory, double& shard_span_perc, int& n_engines) {
		if (argc >= 2) {
			data_dir = argv[1];
		}
		if (argc >= 3) {
			edge_factor = 16;
			if (argv[2] == std::string("XS")) {
				scale = 8;
			}
      else if (argv[2] == std::string("S")) {
				scale = 14;
      }
			else if (argv[2] == std::string("M")) {
				scale = 16;
			}
			else if (argv[2] == std::string("L")) {
				scale = 20;
			}
			else if (argv[2] == std::string("XL")) {
				scale = 24;
			}
      else if (argv[2] == std::string("XXL")) {
				scale = 26;
      }
		}
		if (argc >= 4) {
			in_memory = (argv[3] == std::string("true")) ? true : false;
		}
		if (argc >= 5) {
			try {
				if (std::stoi(argv[4]) == 5) shard_span_perc = 0.05;
				else if (std::stoi(argv[4]) == 20) shard_span_perc = 0.2;
				else if (std::stoi(argv[4]) == 100) shard_span_perc = 1;
			}
			catch (std::exception & e) {
				std::cout << e.what() << std::endl;
			}
		}
    if (argc >= 6) {
      try {
        n_engines = std::stoi(argv[5]);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
    }
	}
};
#endif //TENSORBASE_BENCHMARKGRAPH_H