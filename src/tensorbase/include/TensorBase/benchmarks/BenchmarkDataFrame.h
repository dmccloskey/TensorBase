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
	/// The select Functor for the DataFrame `indices`
	template<typename LabelsT, typename DeviceT>
	class SelectTableDataIndices {
	public:
    SelectTableDataIndices(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, const std::string& table_name) : select_labels_(select_labels), table_name_(table_name) {};
    ~SelectTableDataIndices() = default;
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
			SelectClause<LabelsT, DeviceT> select_clause1(this->table_name_, "1_indices", this->select_labels_);
			TensorSelect tensorSelect;
			tensorSelect.selectClause(tensor_collection, select_clause1, device);
			if (this->apply_select_) tensorSelect.applySelect(tensor_collection, { this->table_name_ }, { this->table_name_ }, device);
		}
  protected:
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels_;
    std::string table_name_;
    bool apply_select_ = false;
	};

  /* 
  @class The select Functor for the DataFrame `is_valid` column

  The query selectes all data where `is_valid = 1` and performs a reduction sum on the
  `is_valid` column to count the number of valid entries.  The results are copied over
  to `results_` where they can be synced to the cpu and viewed.
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class SelectTableDataIsValid {
  public:
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> result_; ///< The results of the query
    void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
      setLabelsValuesResult(device);
      // Make and apply the where clause
      WhereClause<LabelsT, TensorT, DeviceT> where_clause1("DataFrame_is_valid", "2_columns", select_labels_, select_values_, logicalComparitors::EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);
      TensorSelect tensorSelect;
      tensorSelect.whereClause(tensor_collection, where_clause1, device);
      tensorSelect.applySelect(tensor_collection, { "DataFrame_is_valid" }, { "DataFrame_is_valid_true" }, device);

      // Reset the indices
      tensor_collection->tables_.at("DataFrame_is_valid")->resetIndicesView(device);

      // Make and apply the reduction clause
      ReductionClause<DeviceT> reduction_clause1("DataFrame_is_valid_true", reductionFunctions::SUM);
      tensorSelect.applyReduction(tensor_collection, reduction_clause1, device);

      // Copy out the results
      std::shared_ptr<TensorT[]> data_is_valid;
      tensor_collection->tables_.at("DataFrame_is_valid_true")->getDataPointer(data_is_valid);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_is_valid_values(data_is_valid.get(), 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> result_values(result_->getDataPointer().get(), 1);
      result_values.device(device) = data_is_valid_values;

      // Remove the intermediate tables
      tensor_collection->removeTensorTable("DataFrame_is_valid_true");
    }
    virtual void setLabelsValuesResult(DeviceT& device) = 0;
  protected:
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
  };

  /*
  @class The select Functor for the DataFrame `label` column

  The query selects and aggregates all data where `label = "one"`, and counts the number of resulting indices.  
  The results are copied over to `results_` where they can be viewed.
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class SelectTableDataLabel {
  public:
    int result_; ///< The results of the query
    void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
      setLabelsValuesResult(device);
      // Make and apply the where clause
      WhereClause<LabelsT, TensorT, DeviceT> where_clause1("DataFrame_label", "2_columns", select_labels_, select_values_, logicalComparitors::EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);
      TensorSelect tensorSelect;
      tensorSelect.whereClause(tensor_collection, where_clause1, device);
      tensorSelect.applySelect(tensor_collection, { "DataFrame_label" }, { "DataFrame_label_one" }, device);

      // Reset the indices
      tensor_collection->tables_.at("DataFrame_label")->resetIndicesView(device);

      // Make and apply the reduction clause
      ReductionClause<DeviceT> reduction_clause1("DataFrame_label_one", reductionFunctions::COUNT);
      tensorSelect.applyReduction(tensor_collection, reduction_clause1, device);

      // Copy out the results
      result_ = tensor_collection->tables_.at("DataFrame_label_one")->getDimSizeFromAxisName("1_indices");

      // Remove the intermediate tables
      tensor_collection->removeTensorTable("DataFrame_label_one");
    }
    virtual void setLabelsValuesResult(DeviceT& device) = 0;
  protected:
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> select_values_; ///< The values to select
  };

  /*
  @class The select Functor for the DataFrame `image_2D` column

  The query selectes all data in the first two weeks of January and performs a reduction MEAN on the
  `image_2D` column to determine the average pixel intensity.  The results are copied over
  to `results_` where they can be synced to the cpu and viewed.
  */
  template<typename LabelsT, typename TensorT1, typename TensorT2, typename DeviceT>
  class SelectTableDataImage2D {
  public:
    std::shared_ptr<TensorData<TensorT2, DeviceT, 1>> result_; ///< The results of the query
    void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
      setLabelsValuesResult(device);
      TensorSelect tensorSelect;
      // Make the where clause on the time table
      WhereClause<LabelsT, TensorT1, DeviceT> where_clause1("DataFrame_time", "3_time", select_labels_, select_values_lt_, logicalComparitors::LESS_THAN_OR_EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);
      tensorSelect.whereClause(tensor_collection, where_clause1, device);
      WhereClause<LabelsT, TensorT1, DeviceT> where_clause2("DataFrame_time", "3_time", select_labels_, select_values_gt_, logicalComparitors::GREATER_THAN_OR_EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);
      tensorSelect.whereClause(tensor_collection, where_clause2, device);

      // Copy the indice view from time to image_2D
      tensor_collection->tables_.at("DataFrame_image_2D")->replaceIndicesView("1_indices", tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices"), device);

      // Apply the where clause on the image_2d table
      tensorSelect.applySelect(tensor_collection, { "DataFrame_image_2D" }, { "DataFrame_image_2D_jan" }, device);

      // Reset the indices
      tensor_collection->tables_.at("DataFrame_time")->resetIndicesView(device);
      tensor_collection->tables_.at("DataFrame_image_2D")->resetIndicesView(device);

      // Make and apply the reduction clause
      ReductionClause<DeviceT> reduction_clause1("DataFrame_image_2D_jan", reductionFunctions::MEAN);
      tensorSelect.applyReduction(tensor_collection, reduction_clause1, device);

      // Copy out the results
      std::shared_ptr<TensorT2[]> data_image_2D;
      tensor_collection->tables_.at("DataFrame_image_2D_jan")->getDataPointer(data_image_2D);
      Eigen::TensorMap<Eigen::Tensor<TensorT2, 1>> data_image_2D_values(data_image_2D.get(), 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT2, 1>> result_values(result_->getDataPointer().get(), 1);
      result_values.device(device) = data_image_2D_values;

      // Remove the intermediate tables
      tensor_collection->removeTensorTable("DataFrame_image_2D_jan");
    }
    virtual void setLabelsValuesResult(DeviceT& device) = 0;
  protected:
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> select_labels_; ///< The labels to select
    std::shared_ptr<TensorData<TensorT1, DeviceT, 1>> select_values_lt_; ///< The values to select
    std::shared_ptr<TensorData<TensorT1, DeviceT, 1>> select_values_gt_; ///< The values to select
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
    void setUseRandomValues(const bool& use_random_values) { use_random_values_ = use_random_values; }

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
    std::tm time_ = { 0 };
	public:
    DataFrameManagerTime(const int& data_size, const bool& use_random_values = false) : DataFrameManager(data_size, use_random_values) { initTime(); };
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 3>>& values_ptr);
    void initTime();
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
	void DataFrameManagerTime<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 3>>& values_ptr) {
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(1, span);
		Eigen::Tensor<TensorT, 3> values(span, 1, 6);

    // Make the fixed starting time
    std::tm time_start = time_;
    for (int i = offset; i < offset + span; ++i) {
      labels(0, i - offset) = LabelsT(i);
      if (this->use_random_values_) {
        time_start.tm_hour += 2;
        std::mktime(&time_start);
      }
      else {
        time_start.tm_hour += 1;
        std::mktime(&time_start);
      }
      values(i - offset, 0, 0) = TensorT(time_start.tm_sec);
      values(i - offset, 0, 1) = TensorT(time_start.tm_min);
      values(i - offset, 0, 2) = TensorT(time_start.tm_hour);
      values(i - offset, 0, 3) = TensorT(time_start.tm_mday);
      values(i - offset, 0, 4) = TensorT(time_start.tm_mon);
      values(i - offset, 0, 5) = TensorT(time_start.tm_year);
    }
    time_ = time_start;
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void DataFrameManagerTime<LabelsT, TensorT, DeviceT>::initTime()
  {
    time_ = { 0 };
    //// The below is not defined in CUDA c++11...
    //std::istringstream iss("01/01/2008 00:00:00");
    //iss.imbue(std::locale(""));
    //iss >> std::get_time(&time_, "%d/%m/%Y %H:%M:%S");
    time_.tm_isdst = false;
  }

  /*
  @brief Specialized `DataFrameManager` for generating labels
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class DataFrameManagerLabel : public DataFrameManager<LabelsT, TensorT, DeviceT, 2> {
  public:
    using DataFrameManager::DataFrameManager;
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
  private:
    std::vector<std::string> labels_ = { "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  void DataFrameManagerLabel<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr) {
    // Make the labels and values
    Eigen::Tensor<LabelsT, 2> labels(1, span);
    Eigen::Tensor<TensorT, 2> values(span, 1);

    for (int i = offset; i < offset + span; ++i) {
      labels(0, i - offset) = LabelsT(i);
      if (this->use_random_values_) values(i - offset, 0) = TensorT("null");
      else values(i - offset, 0) = TensorT(labels_.at(i % labels_.size()));
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
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void DataFrameManagerImage2D<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 4>>& values_ptr)
  {
    // Make the labels and values
    Eigen::Tensor<LabelsT, 2> labels(1, span);
    Eigen::Tensor<TensorT, 4> values(span, 1, 28, 28);

    for (int i = offset; i < offset + span; ++i) {
      labels(0, i - offset) = LabelsT(i);
      Eigen::Tensor<TensorT, 4> image(1, 1, 28, 28);
      if (this->use_random_values_) image.setConstant(1);// image = image.random().abs(); // should be from 0 to 1
      else image.setZero();
      values.slice(Eigen::array<Eigen::Index, 4>({ i - offset, 0, 0, 0}), Eigen::array<Eigen::Index, 4>({ 1, 1, 28, 28 })) = image;
    }
    this->makeLabelsPtr(labels, labels_ptr);
    this->makeValuesPtr(values, values_ptr);
  }

  /*
  @brief Specialized `DataFrameManager` for generating is_valid
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class DataFrameManagerIsValid : public DataFrameManager<LabelsT, TensorT, DeviceT, 2> {
  public:
    using DataFrameManager::DataFrameManager;
    void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
  };
  template<typename LabelsT, typename TensorT, typename DeviceT>
  inline void DataFrameManagerIsValid<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr)
  {
    // Make the labels and values
    Eigen::Tensor<LabelsT, 2> labels(1, span);
    Eigen::Tensor<TensorT, 2> values(span, 1);

    for (int i = offset; i < offset + span; ++i) {
      labels(0, i - offset) = LabelsT(i);
      if (this->use_random_values_) values(i - offset, 0) = TensorT(0); // all images are not valid
      else values(i - offset, 0) = TensorT(i % 2); // every other image is not valid
    }
    this->makeLabelsPtr(labels, labels_ptr);
    this->makeValuesPtr(values, values_ptr);
  }

	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
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
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	std::string BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>::insert1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_insert1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	std::string BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>::update1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_update1TimePoint(transaction_manager, data_size, in_memory, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	std::string BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>::delete1TimePoint(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		_delete1TimePoint(transaction_manager, data_size, in_memory, device);

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

	template<typename DataFrameManagerTimeT, typename DataFrameManagerLabelT, typename DataFrameManagerImage2DT, typename DataFrameManagerIsValidT, typename DeviceT>
	static void runBenchmarkDataFrame(const std::string& data_dir, const int& data_size, const bool& in_memory, const bool& is_columnar, const double& shard_span_perc,
		const BenchmarkDataFrame1TimePoint<DataFrameManagerTimeT, DataFrameManagerLabelT, DataFrameManagerImage2DT, DataFrameManagerIsValidT, DeviceT>& benchmark_1_tp,
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