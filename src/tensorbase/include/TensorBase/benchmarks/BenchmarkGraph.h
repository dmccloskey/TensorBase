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
		SelectGraphNodeLinkIDs(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, const std::string& table_name) : select_labels_(select_labels), table_name_(table_name) {};
		~SelectGraphNodeLinkIDs() = default;
		virtual void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels_;
		std::string table_name_;
	};

	/*
	@brief Class for selecting and counting nodes with particular properties
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAndCountNodeProperty {
	public:
		int result_; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};

	/*
	@brief Class for selecting and counting links with particular properties
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAndCountLinkProperty {
	public:
		int result_; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};

	/*
	@brief Class for making the adjacency matrix from a sparse representation
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAdjacency {
	public:
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> result_; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};

	/*
	@brief Class for running the breadth-first search algorithm
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectBFS {
	public:
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> result_; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};

	/*
	@brief Class for running the single source shortest path search algorithm
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectSSSP {
	public:
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> result_; ///< The results of the query
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
		virtual void setLabelsValuesResult(DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
	};

	/*
	@brief Class for managing the generation of data for the Graph
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT, int NDim>
	class GraphManager {
	public:
    GraphManager(const bool& use_random_values = false) : use_random_values_(use_random_values){};
		~GraphManager() = default;
		virtual void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr, DeviceT& device) = 0;
		virtual void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, DeviceT& device) = 0;
		virtual void makeValuesPtr(const Eigen::array<Eigen::Index, NDim>& dimensions, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr, DeviceT& device) = 0;
    void makeKroneckerGraph(const int& scale, const int& edge_factor, DeviceT& device);
    void setUseRandomValues(const bool& use_random_values) { use_random_values_ = use_random_values; }
    std::shared_ptr<TensorData<KGLabelsT, DeviceT, 2>> kronecker_graph_indices_;
    std::shared_ptr<TensorData<KGTensorT, DeviceT, 2>> kronecker_graph_weights_;
    std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>> kronecker_graph_node_ids_;
    std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>> kronecker_graph_link_ids_;
	protected:
		bool use_random_values_;
	};
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT, int NDim>
  inline void GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, NDim>::makeKroneckerGraph(const int& scale, const int& edge_factor, DeviceT& device) {
    KroneckerGraphGenerator<KGLabelsT, KGTensorT, DeviceT> graph_generator;
		graph_generator.makeKroneckerGraph(scale, edge_factor, kronecker_graph_indices_, kronecker_graph_weights_, device);
    graph_generator.getNodeAndLinkIds(0, kronecker_graph_indices_->getDimensions().at(0), kronecker_graph_indices_, kronecker_graph_node_ids_, kronecker_graph_link_ids_, device);
  }

	/*
	@class Specialized `GraphManager` for generating sparse graph representation
		that includes input and output `node_id`s matched to `link_id`s

		NOTES: original idea
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerSparse : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
		using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparse<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device) {
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 3, span }; // node_in, node_out, link_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by weights
		this->makeLabelsPtr(labels_dims, labels_ptr);
		this->makeValuesPtr(values_dims, values_ptr);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> indices_values(this->kronecker_graph_indices_->getDataPointer().get(), this->kronecker_graph_indices_->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(this->kronecker_graph_link_ids_->getDataPointer().get(), 1, (int)this->kronecker_graph_link_ids_->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 2, span })).device(device) = indices_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 2 })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 }));
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 2, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span }));

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> weights_values(this->kronecker_graph_weights_->getDataPointer().get(), this->kronecker_graph_weights_->getDimensions());
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
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device) {
		// Make the labels and values
    Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // link_id
    Eigen::array<Eigen::Index, 2> values_dims = { span, 2 }; // indices by [node_in, node_out]
    this->makeLabelsPtr(labels_dims, labels_ptr);
    this->makeValuesPtr(values_dims, values_ptr);

    // Assign the labels data
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(this->kronecker_graph_link_ids_->getDataPointer().get(), 1, (int)this->kronecker_graph_link_ids_->getTensorSize());
    labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
      Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

    // Assign the values data
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> indices_values(this->kronecker_graph_indices_->getDataPointer().get(), this->kronecker_graph_indices_->getDimensions());
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
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerWeights<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device) {
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // link_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by weights
		this->makeLabelsPtr(labels_dims, labels_ptr);
		this->makeValuesPtr(values_dims, values_ptr);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(this->kronecker_graph_link_ids_->getDataPointer().get(), 1, (int)this->kronecker_graph_link_ids_->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> weights_values(this->kronecker_graph_weights_->getDataPointer().get(), this->kronecker_graph_weights_->getDimensions());
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
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
		virtual void setLabels(DeviceT& device) = 0;
		virtual void setNodeIds(const int& offset, const int& span, DeviceT& device) = 0;
  private:
		std::vector<std::string> node_colors_ = {"white", "black", "red", "blue", "green"};
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_;
		std::shared_ptr<TensorData<KGLabelsT, DeviceT, 1>> node_ids_;
  };
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	inline void GraphManagerNodeProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device)
	{
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // node_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by "label"
		this->makeLabelsPtr(labels_dims, labels_ptr);
		this->makeValuesPtr(values_dims, values_ptr);

		// Set the values on the device for transfer
		setLabels(device);
		setNodeIds(device);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> node_ids_values(this->node_ids_->getDataPointer().get(), 1, (int)this->node_ids_->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = node_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> labels_values(this->labels_->getDataPointer().get(), this->labels_->getTensorSize(), 1);
		values_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).device(device) = labels_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).cast<TensorT>();
	}

  /*
  @brief Specialized `GraphManager` for generating the link properties
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerLinkProperty : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
		using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
		virtual void setLabels(DeviceT& device) = 0;
	private:
		std::vector<std::string> link_types_ = { "solid", "dashed" };
		std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_;
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	inline void GraphManagerLinkProperty<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device)
	{
		// Make the labels and values
		Eigen::array<Eigen::Index, 2> labels_dims = { 1, span }; // link_id
		Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by "label"
		this->makeLabelsPtr(labels_dims, labels_ptr);
		this->makeValuesPtr(values_dims, values_ptr);

		// Set the values on the device for transfer
		setLabels(device);

		// Assign the labels data
		Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(this->kronecker_graph_link_ids_->getDataPointer().get(), 1, (int)this->kronecker_graph_link_ids_->getTensorSize());
		labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
			Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span })).cast<LabelsT>();

		// Assign the values data
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
		Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> labels_values(this->labels_->getDataPointer().get(), this->labels_->getTensorSize(), 1);
		values_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).device(device) = labels_values.slice(
			Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).cast<TensorT>();
	}
	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename DeviceT>
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

	protected:
		virtual void _insert1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1Link`
		virtual void _update1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1Link`
		virtual void _delete1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1Link`
    virtual int _selectAndCountNodeProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndCountNodeProperty`
    virtual int _selectAndCountLinkProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndCountLinkProperty`
    virtual float _selectAdjacency(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAdjacency`
		virtual float _selectBFS(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectBFS`
		virtual float _selectSSSP(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectSSSP`
	};
	template<typename DeviceT>
	std::string BenchmarkGraph1Link<DeviceT>::insert1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_insert1Link(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DeviceT>
	std::string BenchmarkGraph1Link<DeviceT>::update1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_update1Link(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DeviceT>
	std::string BenchmarkGraph1Link<DeviceT>::delete1Link(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_delete1Link(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
  template<typename DeviceT>
  inline std::pair<std::string, int> BenchmarkGraph1Link<DeviceT>::selectAndCountNodeProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const int result = _selectAndCountNodeProperty(transaction_manager, scale, edge_factor, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
  template<typename DeviceT>
  inline std::pair<std::string, int> BenchmarkGraph1Link<DeviceT>::selectAndCountLinkProperty(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const int result = _selectAndCountLinkProperty(transaction_manager, scale, edge_factor, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
  template<typename DeviceT>
  inline std::pair<std::string, float> BenchmarkGraph1Link<DeviceT>::selectAdjacency(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const float result = _selectAdjacency(transaction_manager, scale, edge_factor, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
	template<typename DeviceT>
	inline std::pair<std::string, float> BenchmarkGraph1Link<DeviceT>::selectBFS(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		const float result = _selectBFS(transaction_manager, scale, edge_factor, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return std::pair(milli_time, result);
	}
	template<typename DeviceT>
	inline std::pair<std::string, float> BenchmarkGraph1Link<DeviceT>::selectSSSP(TransactionManager<DeviceT>& transaction_manager, const int& scale, const int& edge_factor, const bool& in_memory, DeviceT& device) const
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

	template<typename DeviceT>
	static void runBenchmarkGraph(const std::string& data_dir, const int& scale, const int& edge_factor, const bool& in_memory, const double& shard_span_perc,
		const BenchmarkGraph1Link<DeviceT>& benchmark_1_link,
		const GraphTensorCollectionGenerator<DeviceT>& tensor_collection_generator, DeviceT& device) {
		std::cout << "Starting insert/delete/update Graph benchmarks for scale=" << scale << ", edge_factor=" << edge_factor << ", in_memory=" << in_memory << ", and shard_span_perc=" << shard_span_perc << std::endl;

		// Make the nD TensorTables
		std::shared_ptr<TensorCollection<DeviceT>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, device);

		// Setup the transaction manager
		TransactionManager<DeviceT> transaction_manager;
		transaction_manager.setMaxOperations(data_size + 1);

		// Run the table through the benchmarks
		transaction_manager.setTensorCollection(n_dim_tensor_collection);
		std::cout << "Tensor Table time-point insertion took " << benchmark_1_link.insert1Link(transaction_manager, scale, edge_factor, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Tensor Table time-point select and count black nodes took " << (benchmark_1_link.selectAndCountNodeProperty(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point select and count dashed links took " << (benchmark_1_link.selectAndCountLinkProperty(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point select and make the adjacency matrix " << (benchmark_1_link.selectAdjacency(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point select and perform a breadth-first search " << (benchmark_1_link.selectBFS(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point select and perform a single-source shortest path search " << (benchmark_1_link.selectSSSP(transaction_manager, scale, edge_factor, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point update took " << benchmark_1_link.update1Link(transaction_manager, scale, edge_factor, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Tensor Table time-point deletion took " << benchmark_1_link.delete1Link(transaction_manager, scale, edge_factor, in_memory, device) << " milliseconds." << std::endl;
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