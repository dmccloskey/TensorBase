/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKDATAFRAME_H
#define TENSORBASE_BENCHMARKDATAFRAME_H

#include <ctime> // time format
#include <chrono> // current time
#include <math.h> // std::pow
#include <random> // random number generator

#include <unsupported/Eigen/CXX11/Tensor>

#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/ml/TensorSelect.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/// Base class for all select functors
	template<typename LabelsT, typename DeviceT>
	class DataFrameSelectTable {
	public:
		DataFrameSelectTable(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels) : select_labels_(select_labels){};
		~DataFrameSelectTable() = default;
		virtual void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels_;
		bool apply_select_ = false;
	};

	/// The select Functor for the DataFrame columns
	template<typename LabelsT, typename DeviceT>
	class SelectTableDataColumns: public DataFrameSelectTable<LabelsT, DeviceT> {
	public:
		using DataFrameSelectTable::DataFrameSelectTable;
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::string& table_name, DeviceT& device) override {
			SelectClause<LabelsT, DeviceT> select_clause1(table_name, "1_columns", this->select_labels_);
			TensorSelect tensorSelect;
			tensorSelect.selectClause(tensor_collection, select_clause1, device);
			if (this->apply_select_) tensorSelect.applySelect(tensor_collection, { table_name }, device);
		}
	};

	/*
	@brief Class for managing the generation of data for the DataFrame

	*/
	template<typename LabelsT, typename TensorT, typename DeviceT, int NDim>
	class DataFrameManager {
	public:
    DataFrameManager(const int& data_size, const bool& use_random_values = false) : data_size_(data_size), use_random_values_(use_random_values){};
		~DataFrameManager() = default;
		virtual void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr) = 0;
		virtual void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr) = 0;
		virtual void makeValuesPtr(const Eigen::Tensor<TensorT, NDim>& values, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr) = 0;

		/*
		@brief Generate a random value
		*/
		TensorT getRandomValue();
	protected:
		int data_size_;
		bool use_random_values_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT, int NDim>
	TensorT DataFrameManager<LabelsT, TensorT, DeviceT, NDim>::getRandomValue() {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> d{ 0.0f, 10.0f };
		return TensorT(d(gen));
	}

	/*
	@brief Specialized `DataFrameManager` for generating time_stamps
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class DataFrameManagerTime : public DataFrameManager<LabelsT, TensorT, DeviceT, 3> {
	public:
		using DataFrameManager::DataFrameManager;
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 3>>& values_ptr);
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
	void DataFrameManagerTime<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 3>>& values_ptr) {
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(1, span);
		Eigen::Tensor<TensorT, 3> values(1, span, 6);

    // Make the fixed starting time
    std::tm time_start;
    std::istringstream iss("01/01/2008 00:00:00");
    iss.imbue(std::locale(""));
    iss >> std::get_time(&time_start, "%d/%m/%Y %H:%M:%S");

		for (int i = offset; i < offset + span; ++i) {      
			labels(0, i - offset) = LabelsT(i);
      time_start.tm_sec += i * 10;
      std::mktime(&time_start);
			values(0, i - offset, 0) = TensorT(time_start.tm_sec);
			values(0, i - offset, 1) = TensorT(time_start.tm_min);
			values(0, i - offset, 2) = TensorT(time_start.tm_hour);
			values(0, i - offset, 3) = TensorT(time_start.tm_mday);
      values(0, i - offset, 4) = TensorT(time_start.tm_mon);
      values(0, i - offset, 5) = TensorT(time_start.tm_year);
		}
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}

  /*
  @brief Specialized `DataFrameManager` for generating labels
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class DataFrameManagerLabels : public DataFrameManager<LabelsT, TensorT, DeviceT, 2> {
  public:
    using DataFrameManager::DataFrameManager;
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
  private:
    std::vector<std::string> labels_ = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  void DataFrameManagerLabels<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr) {
    // Make the labels and values
    Eigen::Tensor<LabelsT, 2> labels(1, span);
    Eigen::Tensor<TensorT, 2> values(1, span);

    for (int i = offset; i < offset + span; ++i) {
      labels(0, i - offset) = LabelsT(i);
      values(0, i - offset, 0) = TensorT(i);
    }
    this->makeLabelsPtr(labels, labels_ptr);
    this->makeValuesPtr(values, values_ptr);
  }

  /*
  @brief Specialized `DataFrameManager` for generating image_2d
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class DataFrameManagerImage2D : public DataFrameManager<LabelsT, TensorT, DeviceT, 4> {
  public:
    using DataFrameManager::DataFrameManager;
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 4>>& values_ptr);
  };

  /*
  @brief Specialized `DataFrameManager` for generating is_valid
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class DataFrameManagerIsValid : public DataFrameManager<LabelsT, TensorT, DeviceT, 2> {
  public:
    using DataFrameManager::DataFrameManager;
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
  };

	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelsT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	class BenchmarkDataFrame1TimePoint {
	public:
		BenchmarkDataFrame1TimePoint() = default;
		~BenchmarkDataFrame1TimePoint() = default;
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
	protected:
		virtual void _insert1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint0D`
		virtual void _update1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint0D`
		virtual void _delete1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint0D`
	};
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelsT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	std::string BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelsT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>::insert1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		insert1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelsT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	std::string BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelsT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>::update1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		update1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelsT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	std::string BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelsT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>::delete1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		delete1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}

	/*
	@brief Simulate a typical dataframe with mixed column types and mixed entry dimensions
	*/
	template<typename DeviceT>
	class DataFrameTensorCollectionGenerator {
	public:
		DataFrameTensorCollectionGenerator() = default;
		~DataFrameTensorCollectionGenerator() = default;
		virtual std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, DeviceT& device) const = 0;
	};

	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelsT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	static void runBenchmarkDataFrame(const std::string& data_dir, const int& data_size, const bool& in_memory, const bool& is_columnar, const double& shard_span_perc,
		const BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelsT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>& benchmark_1_tp,
		const DataFrameTensorCollectionGenerator<DeviceT>& tensor_collection_generator, DeviceT& device) {
		std::cout << "Starting insert/delete/update DataFrame benchmarks for data_size=" << data_size << ", in_memory=" << in_memory << ", is_columnar=" << is_columnar << ", and shard_span_perc=" << shard_span_perc << std::endl;

		// Make the nD TensorTables
		std::shared_ptr<TensorCollection<DeviceT>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, is_columnar, device);

		// Setup the transaction manager
		TransactionManager<DeviceT> transaction_manager;
		transaction_manager.setMaxOperations(data_size + 1);

		// Run the table through the benchmarks
		transaction_manager.setTensorCollection(n_dim_tensor_collection);
		std::cout << "Tensor Table time-point insertion took " << benchmark_1_tp.insert1TimePoint(transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Tensor Table time-point update took " << benchmark_1_tp.update1TimePoint(transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
		std::cout << "Tensor Table time-point deletion took " << benchmark_1_tp.delete1TimePoint(transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
	}

	///Parse the command line arguments
	static void parseCmdArgsDataFrame(const int& argc, char** argv, std::string& data_dir, int& data_size, bool& in_memory, bool& is_columnar, double& shard_span_perc, int& n_engines) {
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
#endif //TENSORBASE_BENCHMARKDATAFRAME_H