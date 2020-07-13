/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKDATAFRAMEDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKDATAFRAMEDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkDataFrame.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>
#include <TensorBase/ml/TensorOperationDefaultDevice.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/*
	@brief Specialized `DataFrameManager` for the DefaultDevice case
	*/
	class DataFrameManagerTimeDefaultDevice : public DataFrameManagerTime<int, int, Eigen::DefaultDevice> {
	public:
		using DataFrameManagerTime::DataFrameManagerTime;
		void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<int, 3>& values, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>>& values_ptr);
	};
	void DataFrameManagerTimeDefaultDevice::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_data);
	}
	void DataFrameManagerTimeDefaultDevice::makeValuesPtr(const Eigen::Tensor<int, 3>& values, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>>& values_ptr) {
		TensorDataDefaultDevice<int, 3> values_data(Eigen::array<Eigen::Index, 3>({ values.dimension(0), values.dimension(1), values.dimension(2) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<int, 3>>(values_data);
	}

  /*
  @brief Specialized `DataFrameManager` for the DefaultDevice case
  */
  class DataFrameManagerLabelsDefaultDevice : public DataFrameManagerLabels<int, TensorArray32<char>, Eigen::DefaultDevice> {
  public:
    using DataFrameManagerLabels::DataFrameManagerLabels;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<TensorArray32<char>, 2>& values, std::shared_ptr<TensorData<TensorArray32<char>, Eigen::DefaultDevice, 2>>& values_ptr);
  };
  void DataFrameManagerLabelsDefaultDevice::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr) {
    TensorDataDefaultDevice<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_data);
  }
  void DataFrameManagerLabelsDefaultDevice::makeValuesPtr(const Eigen::Tensor<TensorArray32<char>, 2>& values, std::shared_ptr<TensorData<TensorArray32<char>, Eigen::DefaultDevice, 2>>& values_ptr) {
    TensorDataDefaultDevice<TensorArray32<char>, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataDefaultDevice<TensorArray32<char>, 2>>(values_data);
  }

  /*
  @brief Specialized `DataFrameManager` for the DefaultDevice case
  */
  class DataFrameManagerImage2DDefaultDevice : public DataFrameManagerImage2D<int, float, Eigen::DefaultDevice> {
  public:
    using DataFrameManagerImage2D::DataFrameManagerImage2D;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<float, 4>& values, std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 4>>& values_ptr);
  };

  /*
  @brief Specialized `DataFrameManager` for the DefaultDevice case
  */
  class DataFrameManagerIsValidDefaultDevice : public DataFrameManagerIsValid<int, int, Eigen::DefaultDevice> {
  public:
    using DataFrameManagerIsValid::DataFrameManagerIsValid;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<int, 2>& values, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& values_ptr);
  };

	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	class BenchmarkDataFrame1TimePointDefaultDevice : public BenchmarkDataFrame1TimePoint<DataFrameManagerTimeDefaultDevice, DataFrameManagerLabelsDefaultDevice, DataFrameManagerImage2DDefaultDevice, DataFrameManagerIsValidDefaultDevice, Eigen::DefaultDevice> {
	protected:
		void insert1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
		void update1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
		void delete1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint0D`
	};
	void BenchmarkDataFrame1TimePointDefaultDevice::insert1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
    DataFrameManagerTimeDefaultDevice dataframe_manager_time(data_size, false);
    DataFrameManagerLabelsDefaultDevice dataframe_manager_labels(data_size, false);
    DataFrameManagerImage2DDefaultDevice dataframe_manager_image_2d(data_size, false);
    DataFrameManagerIsValidDefaultDevice dataframe_manager_is_valid(data_size, false);
		this->insert1TimePoint0D_(dataframe_manager_time, dataframe_manager_labels, dataframe_manager_image_2d, dataframe_manager_is_valid, transaction_manager, data_size, in_memory, device);
	}
	void BenchmarkDataFrame1TimePointDefaultDevice::update1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
    DataFrameManagerTimeDefaultDevice dataframe_manager_time(data_size, true);
    DataFrameManagerLabelsDefaultDevice dataframe_manager_labels(data_size, true);
    DataFrameManagerImage2DDefaultDevice dataframe_manager_image_2d(data_size, true);
    DataFrameManagerIsValidDefaultDevice dataframe_manager_is_valid(data_size, true);
		this->update1TimePoint0D_(dataframe_manager_time, dataframe_manager_labels, dataframe_manager_image_2d, dataframe_manager_is_valid, transaction_manager, data_size, in_memory, device);
	}
	void BenchmarkDataFrame1TimePointDefaultDevice::delete1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
    DataFrameManagerTimeDefaultDevice dataframe_manager_time(data_size, true);
    DataFrameManagerLabelsDefaultDevice dataframe_manager_labels(data_size, true);
    DataFrameManagerImage2DDefaultDevice dataframe_manager_image_2d(data_size, true);
    DataFrameManagerIsValidDefaultDevice dataframe_manager_is_valid(data_size, true);
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) {
   //   std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
   //   std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> values_ptr;
			//dataframe_manager.getInsertData(i, span, labels_ptr, values_ptr);
			//SelectTable0D<LabelsT, Eigen::DefaultDevice> selectClause(labels_ptr);
			//TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2> tensorDelete("TTable", "indices", selectClause);
			//std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2>>(tensorDelete);
			//transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}

	class DataFrameTensorCollectionGeneratorDefaultDevice : public DataFrameTensorCollectionGenerator<Eigen::DefaultDevice> {
	public:
    std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::DefaultDevice& device) const override;
  };
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> DataFrameTensorCollectionGeneratorDefaultDevice::makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(1), dimensions_3_t(1), dimensions_3_x(1), dimensions_3_y(1);
		dimensions_1.setValues({ "columns" });
		dimensions_2.setValues({ "indices" });
    dimensions_3_t.setValues({ "time" });
    dimensions_3_x.setValues({ "x" });
    dimensions_3_y.setValues({ "y" });
    Eigen::Tensor<TensorArray32<char>, 2> labels_1_1(1, 1), labels_1_2(1, 1), labels_1_3(1, 1), labels_1_4(1, 1);
    Eigen::Tensor<TensorArray8<char>, 2> labels_3_t(1, 6);
    Eigen::Tensor<int, 2> labels_3_x(1, 28), labels_3_y(1, 28);
    labels_1_1.setValues({ { TensorArray32<char>("time")} });
    labels_1_2.setValues({ { TensorArray32<char>("label")} });
    labels_1_3.setValues({ { TensorArray32<char>("image_2D")} });
    labels_1_4.setValues({ { TensorArray32<char>("is_valid")} });
    labels_3_t.setValues({ { TensorArray8<char>("sec"), TensorArray8<char>("min"), TensorArray8<char>("hour"), TensorArray8<char>("day"), TensorArray8<char>("month"), TensorArray8<char>("year")} });
    labels_3_x.setZero();
    labels_3_x = labels_3_x.constant(1).cumsum(1);
    labels_3_y.setZero();
    labels_3_y = labels_3_y.constant(1).cumsum(1);

		// Setup the tables
		std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 3>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<int, 3>>(TensorTableDefaultDevice<int, 3>("DataFrame_time"));
		auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray32<char>>>(TensorAxisDefaultDevice<TensorArray32<char>>("1_columns", dimensions_1, labels_1_1));
		auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2_indices", 1, 0));
    auto table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("3_time", dimensions_3_t, labels_3_t));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
    table_1_ptr->addTensorAxis(table_1_axis_3_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData(); 
    std::map<std::string, int> shard_span_1;
    shard_span_1.emplace("1_columns", 1);
    shard_span_1.emplace("2_indices", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    shard_span_1.emplace("3_time", 6);
		table_1_ptr->setShardSpans(shard_span_1);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 3>({ 1, data_size, 6}));

    // Setup the tables
		std::shared_ptr<TensorTable<TensorArray32<char>, Eigen::DefaultDevice, 2>> table_2_ptr = std::make_shared<TensorTableDefaultDevice<TensorArray32<char>, 2>>(TensorTableDefaultDevice<TensorArray32<char>, 2>("DataFrame_label"));
		auto table_2_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray32<char>>>(TensorAxisDefaultDevice<TensorArray32<char>>("1_columns", dimensions_1, labels_1_1));
		auto table_2_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2_indices", 1, 0));
		table_2_axis_2_ptr->setDimensions(dimensions_2);
    table_2_ptr->addTensorAxis(table_2_axis_1_ptr);
		table_2_ptr->addTensorAxis(table_2_axis_2_ptr);
		table_2_ptr->setAxes(device);

		// Setup the table data
    table_2_ptr->setData();
    std::map<std::string, int> shard_span_2;
    shard_span_2.emplace("1_columns", 1);
    shard_span_2.emplace("2_indices", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    table_2_ptr->setShardSpans(shard_span_2);
    table_2_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ 1, data_size}));

    // Setup the tables
    std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 4>> table_3_ptr = std::make_shared<TensorTableDefaultDevice<float, 4>>(TensorTableDefaultDevice<float, 4>("DataFrame_image_2D"));
    auto table_3_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray32<char>>>(TensorAxisDefaultDevice<TensorArray32<char>>("1_columns", dimensions_1, labels_1_3));
    auto table_3_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2_indices", 1, 0));
    auto table_3_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3_x", dimensions_3_x, labels_3_x));
    auto table_3_axis_4_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3_y", dimensions_3_y, labels_3_y));
    table_3_axis_2_ptr->setDimensions(dimensions_2);
    table_3_ptr->addTensorAxis(table_3_axis_1_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_2_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_3_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_4_ptr);
    table_3_ptr->setAxes(device);

    // Setup the table data
    table_3_ptr->setData();
    std::map<std::string, int> shard_span_3;
    shard_span_3.emplace("1_columns", 1);
    shard_span_3.emplace("2_indices", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    shard_span_3.emplace("3_x", 28);
    shard_span_3.emplace("3_y", 28);
    table_3_ptr->setShardSpans(shard_span_3);
    table_3_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 4>({ 1, data_size, 28, 28 }));

    // Setup the tables
    std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> table_4_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(TensorTableDefaultDevice<int, 2>("DataFrame_is_valid"));
    auto table_4_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray32<char>>>(TensorAxisDefaultDevice<TensorArray32<char>>("1_columns", dimensions_1, labels_1_1));
    auto table_4_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2_indices", 1, 0));
    table_4_axis_2_ptr->setDimensions(dimensions_2);
    table_4_ptr->addTensorAxis(table_4_axis_1_ptr);
    table_4_ptr->addTensorAxis(table_4_axis_2_ptr);
    table_4_ptr->setAxes(device);

    // Setup the table data
    table_4_ptr->setData();
    table_4_ptr->setShardSpans(shard_span_2);
    table_4_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ 1, data_size }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
		collection_1_ptr->addTensorTable(table_1_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_2_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_3_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_4_ptr, "DataFrame");
    // TODO: linking of axes
		return collection_1_ptr;
	}
};
#endif //TENSORBASE_BENCHMARKDATAFRAMEDEFAULTDEVICE_H