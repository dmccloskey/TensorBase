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
    ~SelectTableDataIndices() = default;
    void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
  protected:
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels_;
    std::string table_name_;
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
    KroneckerGraphGeneratorDefaultDevice<KGLabelsT, KGTensorT> graph_generator;
    graph_generator.makeKroneckerGraph(scale, edge_factor, kronecker_graph_indices_, kronecker_graph_weights_, device);
    graph_generator.getNodeAndLinkIds(kronecker_graph_indices_, kronecker_graph_node_ids_, kronecker_graph_link_ids_, device);
  }

	/*
	@class Specialized `GraphManager` for generating sparse graph representation
    that includes input and output `node_id`s, `link_id`s, and `weights`
	*/
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	class GraphManagerSparse : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
	public:
    using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
		void getInsertData(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
  };
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparse<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT>::getInsertData(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device) {
		// Make the labels and values
    Eigen::array<Eigen::Index, 2> labels_dims = { 3, span }; // node_in, node_out, link_id
    Eigen::array<Eigen::Index, 2> values_dims = { span, 1 }; // indices by weights
    this->makeLabelsPtr(labels_dims, labels_ptr);
    this->makeValuesPtr(values_dims, values_ptr);

    // Assign the labels data
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_ptr->getDataPointer().get(), labels_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> indices_values(kronecker_graph_indices_->getDataPointer().get(), kronecker_graph_indices_->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<KGLabelsT, 2>> link_ids_values(kronecker_graph_link_ids_->getDataPointer().get(), 1, (int)kronecker_graph_link_ids_->getTensorSize());
    labels_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ 2, span })).device(device) = indices_values.slice(
      Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 2 })).shuffle(Eigen::array<Eigen::Index, 2>({ 1, 0 }));
    labels_values.slice(Eigen::array<Eigen::Index, 2>({ 2, 0 }), Eigen::array<Eigen::Index, 2>({ 1, span })).device(device) = link_ids_values.slice(
      Eigen::array<Eigen::Index, 2>({ 0, offset }), Eigen::array<Eigen::Index, 2>({ 1, span }));

    // Assign the values data
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> values_values(values_ptr->getDataPointer().get(), values_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<KGTensorT, 2>> weights_values(kronecker_graph_weights_->getDataPointer().get(), kronecker_graph_weights_->getDimensions());
    values_values.slice(Eigen::array<Eigen::Index, 2>({ 0, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 })).device(device) = weights_values.slice(
      Eigen::array<Eigen::Index, 2>({ offset, 0 }), Eigen::array<Eigen::Index, 2>({ span, 1 }));
	}

  /*
  @brief Specialized `GraphManager` for generating the node properties
  */
  template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
  class GraphManagerNodeProperty : public GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2> {
  public:
    using GraphManager<KGLabelsT, KGTensorT, LabelsT, TensorT, DeviceT, 2>::GraphManager;
    void getInsertData(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr, DeviceT& device);
  private:
    std::vector<std::string> labels_ = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
  };

  /*
  @brief Specialized `GraphManager` for generating the link properties
  

	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename DeviceT>
	class BenchmarkGraph1TimePoint {
	public:
		BenchmarkGraph1TimePoint() = default;
		~BenchmarkGraph1TimePoint() = default;
		/*
		@brief insert 1 time-point at a time

		@param[in, out] transaction_manager
		@param[in] data_size
		@param[in] device

		@returns A string with the total time of the benchmark in milliseconds
		*/
		std::string insert1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
		std::string update1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
		std::string delete1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
    std::pair<std::string, int> selectAndSumIsValid(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
    std::pair<std::string, int> selectAndCountLabels(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
    std::pair<std::string, float> selectAndMeanImage2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;

	protected:
		virtual void _insert1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint0D`
		virtual void _update1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint0D`
		virtual void _delete1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint0D`
    virtual int _selectAndSumIsValid(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndSumIsValid0D`
    virtual int _selectAndCountLabels(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndCountLabels0D`
    virtual float _selectAndMeanImage2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `selectAndMeanImage2D0D`
	};
	template<typename DeviceT>
	std::string BenchmarkGraph1TimePoint<DeviceT>::insert1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_insert1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DeviceT>
	std::string BenchmarkGraph1TimePoint<DeviceT>::update1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_update1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DeviceT>
	std::string BenchmarkGraph1TimePoint<DeviceT>::delete1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_delete1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
  template<typename DeviceT>
  inline std::pair<std::string, int> BenchmarkGraph1TimePoint<DeviceT>::selectAndSumIsValid(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const int result = _selectAndSumIsValid(transaction_manager, data_size, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
  template<typename DeviceT>
  inline std::pair<std::string, int> BenchmarkGraph1TimePoint<DeviceT>::selectAndCountLabels(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const int result = _selectAndCountLabels(transaction_manager, data_size, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }
  template<typename DeviceT>
  inline std::pair<std::string, float> BenchmarkGraph1TimePoint<DeviceT>::selectAndMeanImage2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
  {
    // Start the timer
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const float result = _selectAndMeanImage2D(transaction_manager, data_size, in_memory, device);

    // Stop the timer
    auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::string milli_time = std::to_string(stop - start);
    return std::pair(milli_time, result);
  }

	/*
	@brief Simulate a typical dataframe with mixed column types and mixed entry dimensions
	*/
	template<typename DeviceT>
	class GraphTensorCollectionGenerator {
	public:
		GraphTensorCollectionGenerator() = default;
		~GraphTensorCollectionGenerator() = default;
		virtual std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, DeviceT& device) const = 0;
	};

	template<typename DeviceT>
	static void runBenchmarkGraph(const std::string& data_dir, const int& data_size, const bool& in_memory, const bool& is_columnar, const double& shard_span_perc,
		const BenchmarkGraph1TimePoint<DeviceT>& benchmark_1_tp,
		const GraphTensorCollectionGenerator<DeviceT>& tensor_collection_generator, DeviceT& device) {
		std::cout << "Starting insert/delete/update Graph benchmarks for data_size=" << data_size << ", in_memory=" << in_memory << ", is_columnar=" << is_columnar << ", and shard_span_perc=" << shard_span_perc << std::endl;

		// Make the nD TensorTables
		std::shared_ptr<TensorCollection<DeviceT>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, is_columnar, device);

		// Setup the transaction manager
		TransactionManager<DeviceT> transaction_manager;
		transaction_manager.setMaxOperations(data_size + 1);

		// Run the table through the benchmarks
		transaction_manager.setTensorCollection(n_dim_tensor_collection);
		std::cout << "Tensor Table time-point insertion took " << benchmark_1_tp.insert1TimePoint(transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Tensor Table time-point select and sum `is_valid` took " << (benchmark_1_tp.selectAndSumIsValid(transaction_manager, data_size, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point select and count `labels` == 'one' took " << (benchmark_1_tp.selectAndCountLabels(transaction_manager, data_size, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point select and mean `image_2d` in the first 14 days of Jan took " << (benchmark_1_tp.selectAndMeanImage2D(transaction_manager, data_size, in_memory, device)).first << " milliseconds." << std::endl;
    std::cout << "Tensor Table time-point update took " << benchmark_1_tp.update1TimePoint(transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Tensor Table time-point deletion took " << benchmark_1_tp.delete1TimePoint(transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
	}

	///Parse the command line arguments
	static void parseCmdArgsGraph(const int& argc, char** argv, std::string& data_dir, int& data_size, bool& in_memory, bool& is_columnar, double& shard_span_perc, int& n_engines) {
		if (argc >= 2) {
			data_dir = argv[1];
		}
		if (argc >= 3) {
			if (argv[2] == std::string("XS")) {
				data_size = 1296;
			}
      else if (argv[2] == std::string("S")) {
        data_size = 104976;
      }
			else if (argv[2] == std::string("M")) {
				data_size = 1048576;
			}
			else if (argv[2] == std::string("L")) {
				data_size = 10556001;
			}
			else if (argv[2] == std::string("XL")) {
				data_size = 1003875856;
			}
      else if (argv[2] == std::string("XXL")) {
        data_size = 1e12;
      }
		}
		if (argc >= 4) {
			in_memory = (argv[3] == std::string("true")) ? true : false;
		}
    if (argc >= 5) {
      is_columnar = (argv[4] == std::string("true")) ? true : false;
    }
		if (argc >= 6) {
			try {
				if (std::stoi(argv[5]) == 5) shard_span_perc = 0.05;
				else if (std::stoi(argv[5]) == 20) shard_span_perc = 0.2;
				else if (std::stoi(argv[5]) == 100) shard_span_perc = 1;
			}
			catch (std::exception & e) {
				std::cout << e.what() << std::endl;
			}
		}
    if (argc >= 7) {
      try {
        n_engines = std::stoi(argv[6]);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
    }
	}
};
#endif //TENSORBASE_BENCHMARKGRAPH_H