/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKDATAFRAMECPU_H
#define TENSORBASE_BENCHMARKDATAFRAMECPU_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkDataFrame.h>
#include <TensorBase/ml/TensorCollectionCpu.h>
#include <TensorBase/ml/TensorOperationCpu.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
  /*
  @class Specialized `SelectAndSumIsValid` for the Cpu case
  */
  class SelectAndSumIsValidCpu: public SelectAndSumIsValid<TensorArray32<char>, int, Eigen::ThreadPoolDevice> {
  public:
    void setLabelsValuesResult(Eigen::ThreadPoolDevice& device) override;
  };
  inline void SelectAndSumIsValidCpu::setLabelsValuesResult(Eigen::ThreadPoolDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArray32<char>, 2> select_labels_values(1, 1);
    select_labels_values.setConstant(TensorArray32<char>("is_valid"));
    TensorDataCpu<TensorArray32<char>, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataCpu<TensorArray32<char>, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<int, 1> select_values_values(1);
    select_values_values.setConstant(1);
    TensorDataCpu<int, 1> select_values(select_values_values.dimensions());
    select_values.setData(select_values_values);
    select_values.syncHAndDData(device);
    this->select_values_ = std::make_shared<TensorDataCpu<int, 1>>(select_values);

    // allocate memory for the results
    TensorDataCpu<int, 1> results(Eigen::array<Eigen::Index, 1>({1}));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataCpu<int, 1>>(results);
  }

  /*
  @class Specialized `SelectAndCountLabels` for the Cpu case
  */
  class SelectAndCountLabelsCpu : public SelectAndCountLabels<TensorArray32<char>, TensorArray32<char>, Eigen::ThreadPoolDevice> {
  public:
    void setLabelsValuesResult(Eigen::ThreadPoolDevice& device) override;
  };
  inline void SelectAndCountLabelsCpu::setLabelsValuesResult(Eigen::ThreadPoolDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArray32<char>, 2> select_labels_values(1, 1);
    select_labels_values.setConstant(TensorArray32<char>("label"));
    TensorDataCpu<TensorArray32<char>, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataCpu<TensorArray32<char>, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<TensorArray32<char>, 1> select_values_values(1);
    select_values_values.setConstant(TensorArray32<char>("one"));
    TensorDataCpu<TensorArray32<char>, 1> select_values(select_values_values.dimensions());
    select_values.setData(select_values_values);
    select_values.syncHAndDData(device);
    this->select_values_ = std::make_shared<TensorDataCpu<TensorArray32<char>, 1>>(select_values);
  }
  
  /*
  @class Specialized `SelectTableDataImage2D` for the Cpu case
  */
  class SelectTableDataImage2DCpu : public SelectTableDataImage2D<TensorArray8<char>, int, float, Eigen::ThreadPoolDevice> {
  public:
    void setLabelsValuesResult(Eigen::ThreadPoolDevice& device) override;
  };
  inline void SelectTableDataImage2DCpu::setLabelsValuesResult(Eigen::ThreadPoolDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArray8<char>, 2> select_labels_values(1, 2);
    select_labels_values.setValues({ { TensorArray8<char>("day"),TensorArray8<char>("month") } });
    TensorDataCpu<TensorArray8<char>, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataCpu<TensorArray8<char>, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<int, 1> select_values_values(2);
    select_values_values.setValues({14, 1});
    TensorDataCpu<int, 1> select_values_lt(select_values_values.dimensions());
    select_values_lt.setData(select_values_values);
    select_values_lt.syncHAndDData(device);
    this->select_values_lt_ = std::make_shared<TensorDataCpu<int, 1>>(select_values_lt);
    select_values_values.setValues({ 0, 1 });
    TensorDataCpu<int, 1> select_values_gt(select_values_values.dimensions());
    select_values_gt.setData(select_values_values);
    select_values_gt.syncHAndDData(device);
    this->select_values_gt_ = std::make_shared<TensorDataCpu<int, 1>>(select_values_gt);

    // allocate memory for the results
    TensorDataCpu<float, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataCpu<float, 1>>(results);
  }

	/*
	@class Specialized `DataFrameManager` for the Cpu case
	*/
	class DataFrameManagerTimeCpu : public DataFrameManagerTime<int, int, Eigen::ThreadPoolDevice> {
	public:
		using DataFrameManagerTime<int, int, Eigen::ThreadPoolDevice>::DataFrameManagerTime;
		void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<int, 3>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>>& values_ptr);
	};
	void DataFrameManagerTimeCpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr) {
		TensorDataCpu<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataCpu<int, 2>>(labels_data);
	}
	void DataFrameManagerTimeCpu::makeValuesPtr(const Eigen::Tensor<int, 3>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>>& values_ptr) {
		TensorDataCpu<int, 3> values_data(Eigen::array<Eigen::Index, 3>({ values.dimension(0), values.dimension(1), values.dimension(2) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataCpu<int, 3>>(values_data);
	}

  /*
  @class Specialized `DataFrameManager` for the Cpu case
  */
  class DataFrameManagerLabelCpu : public DataFrameManagerLabel<int, TensorArray32<char>, Eigen::ThreadPoolDevice> {
  public:
    using DataFrameManagerLabel<int, TensorArray32<char>, Eigen::ThreadPoolDevice>::DataFrameManagerLabel;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<TensorArray32<char>, 2>& values, std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>>& values_ptr);
  };
  void DataFrameManagerLabelCpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr) {
    TensorDataCpu<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataCpu<int, 2>>(labels_data);
  }
  void DataFrameManagerLabelCpu::makeValuesPtr(const Eigen::Tensor<TensorArray32<char>, 2>& values, std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>>& values_ptr) {
    TensorDataCpu<TensorArray32<char>, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataCpu<TensorArray32<char>, 2>>(values_data);
  }

  /*
  @class Specialized `DataFrameManager` for the Cpu case
  */
  class DataFrameManagerImage2DCpu : public DataFrameManagerImage2D<int, float, Eigen::ThreadPoolDevice> {
  public:
    using DataFrameManagerImage2D<int, float, Eigen::ThreadPoolDevice>::DataFrameManagerImage2D;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<float, 4>& values, std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>>& values_ptr);
  };
  void DataFrameManagerImage2DCpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr) {
    TensorDataCpu<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataCpu<int, 2>>(labels_data);
  }
  void DataFrameManagerImage2DCpu::makeValuesPtr(const Eigen::Tensor<float, 4>& values, std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>>& values_ptr) {
    TensorDataCpu<float, 4> values_data(Eigen::array<Eigen::Index, 4>({ values.dimension(0), values.dimension(1), values.dimension(2), values.dimension(3) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataCpu<float, 4>>(values_data);
  }

  /*
  @class Specialized `DataFrameManager` for the Cpu case
  */
  class DataFrameManagerIsValidCpu : public DataFrameManagerIsValid<int, int, Eigen::ThreadPoolDevice> {
  public:
    using DataFrameManagerIsValid<int, int, Eigen::ThreadPoolDevice>::DataFrameManagerIsValid;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<int, 2>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& values_ptr);
  };
  void DataFrameManagerIsValidCpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& labels_ptr) {
    TensorDataCpu<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataCpu<int, 2>>(labels_data);
  }
  void DataFrameManagerIsValidCpu::makeValuesPtr(const Eigen::Tensor<int, 2>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& values_ptr) {
    TensorDataCpu<int, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataCpu<int, 2>>(values_data);
  }

	/*
	@class A class for running 1 line insertion, deletion, and update benchmarks
	*/
	class BenchmarkDataFrame1TimePointCpu : public BenchmarkDataFrame1TimePoint<Eigen::ThreadPoolDevice> {
	protected:
		void _insert1TimePoint(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
		void _update1TimePoint(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
		void _delete1TimePoint(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const override; ///< Device specific interface to call `delete1TimePoint0D`
    int _selectAndSumIsValid(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const override; ///< Device specific interface to call `selectAndSumIsValid`
    int _selectAndCountLabels(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const override; ///< Device specific interface to call `selectAndCountLabels`
    float _selectAndMeanImage2D(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const override; ///< Device specific interface to call `selectAndMeanImage2D`
  };
	void BenchmarkDataFrame1TimePointCpu::_insert1TimePoint(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const {
    DataFrameManagerTimeCpu dataframe_manager_time(data_size, false);
    DataFrameManagerLabelCpu dataframe_manager_labels(data_size, false);
    DataFrameManagerImage2DCpu dataframe_manager_image_2d(data_size, false);
    DataFrameManagerIsValidCpu dataframe_manager_is_valid(data_size, false);
    int span = data_size / std::pow(data_size, 0.25);
    for (int i = 0; i < data_size; i += span) {
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_time_ptr;
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_time_ptr;
      dataframe_manager_time.getInsertData(i, span, labels_time_ptr, values_time_ptr);
      TensorAppendToAxis<int, int, Eigen::ThreadPoolDevice, 3> appendToAxis_time("DataFrame_time", "1_indices", labels_time_ptr, values_time_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> appendToAxis_time_ptr = std::make_shared<TensorAppendToAxis<int, int, Eigen::ThreadPoolDevice, 3>>(appendToAxis_time);
      transaction_manager.executeOperation(appendToAxis_time_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
      std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
      dataframe_manager_labels.getInsertData(i, span, labels_labels_ptr, values_labels_ptr);
      TensorAppendToAxis<int, TensorArray32<char>, Eigen::ThreadPoolDevice, 2> appendToAxis_labels("DataFrame_label", "1_indices", labels_labels_ptr, values_labels_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> appendToAxis_labels_ptr = std::make_shared<TensorAppendToAxis<int, TensorArray32<char>, Eigen::ThreadPoolDevice, 2>>(appendToAxis_labels);
      transaction_manager.executeOperation(appendToAxis_labels_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_image_2d_ptr;
      std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_image_2d_ptr;
      dataframe_manager_image_2d.getInsertData(i, span, labels_image_2d_ptr, values_image_2d_ptr);
      TensorAppendToAxis<int, float, Eigen::ThreadPoolDevice, 4> appendToAxis_image_2d("DataFrame_image_2D", "1_indices", labels_image_2d_ptr, values_image_2d_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> appendToAxis_image_2d_ptr = std::make_shared<TensorAppendToAxis<int, float, Eigen::ThreadPoolDevice, 4>>(appendToAxis_image_2d);
      transaction_manager.executeOperation(appendToAxis_image_2d_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_is_valid_ptr;
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_is_valid_ptr;
      dataframe_manager_is_valid.getInsertData(i, span, labels_is_valid_ptr, values_is_valid_ptr);
      TensorAppendToAxis<int, int, Eigen::ThreadPoolDevice, 2> appendToAxis_is_valid("DataFrame_is_valid", "1_indices", labels_is_valid_ptr, values_is_valid_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> appendToAxis_is_valid_ptr = std::make_shared<TensorAppendToAxis<int, int, Eigen::ThreadPoolDevice, 2>>(appendToAxis_is_valid);
      transaction_manager.executeOperation(appendToAxis_is_valid_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
	}
	void BenchmarkDataFrame1TimePointCpu::_update1TimePoint(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const {
    DataFrameManagerTimeCpu dataframe_manager_time(data_size, true);
    DataFrameManagerLabelCpu dataframe_manager_labels(data_size, true);
    DataFrameManagerImage2DCpu dataframe_manager_image_2d(data_size, true);
    DataFrameManagerIsValidCpu dataframe_manager_is_valid(data_size, true);
    int span = data_size / std::pow(data_size, 0.25);
    for (int i = 0; i < data_size; i += span) {
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_time_ptr;
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_time_ptr;
      dataframe_manager_time.getInsertData(i, span, labels_time_ptr, values_time_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_time(labels_time_ptr, "DataFrame_time");
      TensorUpdateValues<int, Eigen::ThreadPoolDevice, 3> updateValues_time("DataFrame_time", selectClause_time, values_time_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> updateValues_time_ptr = std::make_shared<TensorUpdateValues<int, Eigen::ThreadPoolDevice, 3>>(updateValues_time);
      transaction_manager.executeOperation(updateValues_time_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
      std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
      dataframe_manager_labels.getInsertData(i, span, labels_labels_ptr, values_labels_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_labels(labels_labels_ptr, "DataFrame_label");
      TensorUpdateValues<TensorArray32<char>, Eigen::ThreadPoolDevice, 2> updateValues_labels("DataFrame_label", selectClause_labels, values_labels_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> updateValues_labels_ptr = std::make_shared<TensorUpdateValues<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>>(updateValues_labels);
      transaction_manager.executeOperation(updateValues_labels_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_image_2d_ptr;
      std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_image_2d_ptr;
      dataframe_manager_image_2d.getInsertData(i, span, labels_image_2d_ptr, values_image_2d_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_image_2d(labels_image_2d_ptr, "DataFrame_image_2D");
      TensorUpdateValues<float, Eigen::ThreadPoolDevice, 4> updateValues_image_2d("DataFrame_image_2D", selectClause_image_2d, values_image_2d_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> updateValues_image_2d_ptr = std::make_shared<TensorUpdateValues<float, Eigen::ThreadPoolDevice, 4>>(updateValues_image_2d);
      transaction_manager.executeOperation(updateValues_image_2d_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_is_valid_ptr;
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_is_valid_ptr;
      dataframe_manager_is_valid.getInsertData(i, span, labels_is_valid_ptr, values_is_valid_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_is_valid(labels_is_valid_ptr, "DataFrame_is_valid");
      TensorUpdateValues<int, Eigen::ThreadPoolDevice, 2> updateValues_is_valid("DataFrame_is_valid", selectClause_is_valid, values_is_valid_ptr);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> updateValues_is_valid_ptr = std::make_shared<TensorUpdateValues<int, Eigen::ThreadPoolDevice, 2>>(updateValues_is_valid);
      transaction_manager.executeOperation(updateValues_is_valid_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
	}
	void BenchmarkDataFrame1TimePointCpu::_delete1TimePoint(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const {
    DataFrameManagerTimeCpu dataframe_manager_time(data_size, true);
    DataFrameManagerLabelCpu dataframe_manager_labels(data_size, true);
    DataFrameManagerImage2DCpu dataframe_manager_image_2d(data_size, true);
    DataFrameManagerIsValidCpu dataframe_manager_is_valid(data_size, true);
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) {
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_time_ptr;
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_time_ptr;
      dataframe_manager_time.getInsertData(i, span, labels_time_ptr, values_time_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_time(labels_time_ptr, "DataFrame_time");
      TensorDeleteFromAxisCpu<int, int, 3> tensorDelete_time("DataFrame_time", "1_indices", selectClause_time);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> tensorDelete_time_ptr = std::make_shared<TensorDeleteFromAxisCpu<int, int, 3>>(tensorDelete_time);
      transaction_manager.executeOperation(tensorDelete_time_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
      std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
      dataframe_manager_labels.getInsertData(i, span, labels_labels_ptr, values_labels_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_labels(labels_labels_ptr, "DataFrame_label");
      TensorDeleteFromAxisCpu<int, TensorArray32<char>, 2> tensorDelete_labels("DataFrame_label", "1_indices", selectClause_labels);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> tensorDelete_labels_ptr = std::make_shared<TensorDeleteFromAxisCpu<int, TensorArray32<char>, 2>>(tensorDelete_labels);
      transaction_manager.executeOperation(tensorDelete_labels_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_image_2d_ptr;
      std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_image_2d_ptr;
      dataframe_manager_image_2d.getInsertData(i, span, labels_image_2d_ptr, values_image_2d_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_image_2d(labels_image_2d_ptr, "DataFrame_image_2D");
      TensorDeleteFromAxisCpu<int, float, 4> tensorDelete_image_2d("DataFrame_image_2D", "1_indices", selectClause_image_2d);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> tensorDelete_image_2d_ptr = std::make_shared<TensorDeleteFromAxisCpu<int, float, 4>>(tensorDelete_image_2d);
      transaction_manager.executeOperation(tensorDelete_image_2d_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_is_valid_ptr;
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_is_valid_ptr;
      dataframe_manager_is_valid.getInsertData(i, span, labels_is_valid_ptr, values_is_valid_ptr);
      SelectTableDataIndices<int, Eigen::ThreadPoolDevice> selectClause_is_valid(labels_is_valid_ptr, "DataFrame_is_valid");
      TensorDeleteFromAxisCpu<int, int, 2> tensorDelete_is_valid("DataFrame_is_valid", "1_indices", selectClause_is_valid);
      std::shared_ptr<TensorOperation<Eigen::ThreadPoolDevice>> tensorDelete_is_valid_ptr = std::make_shared<TensorDeleteFromAxisCpu<int, int, 2>>(tensorDelete_is_valid);
      transaction_manager.executeOperation(tensorDelete_is_valid_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
  inline int BenchmarkDataFrame1TimePointCpu::_selectAndSumIsValid(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const
  {
    SelectAndSumIsValidCpu select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.result_->syncHAndDData(device);
    return select_and_sum.result_->getData()(0);
  }
  inline int BenchmarkDataFrame1TimePointCpu::_selectAndCountLabels(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const
  {
    SelectAndCountLabelsCpu select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_and_sum.result_;
  }
  inline float BenchmarkDataFrame1TimePointCpu::_selectAndMeanImage2D(TransactionManager<Eigen::ThreadPoolDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::ThreadPoolDevice& device) const
  {
    SelectTableDataImage2DCpu select_and_sum;
    select_and_sum(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.commit(device);
      transaction_manager.initTensorCollectionTensorData(device);
    }
    select_and_sum.result_->syncHAndDData(device);
    return select_and_sum.result_->getData()(0);
  }

	class DataFrameTensorCollectionGeneratorCpu : public DataFrameTensorCollectionGenerator<Eigen::ThreadPoolDevice> {
	public:
    std::shared_ptr<TensorCollection<Eigen::ThreadPoolDevice>> makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::ThreadPoolDevice& device) const override;
  };
	std::shared_ptr<TensorCollection<Eigen::ThreadPoolDevice>> DataFrameTensorCollectionGeneratorCpu::makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::ThreadPoolDevice& device) const
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
		std::shared_ptr<TensorTable<int, Eigen::ThreadPoolDevice, 3>> table_1_ptr = std::make_shared<TensorTableCpu<int, 3>>(TensorTableCpu<int, 3>("DataFrame_time"));
		std::shared_ptr<TensorAxis<TensorArray32<char>, Eigen::ThreadPoolDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisCpu<TensorArray32<char>>>(TensorAxisCpu<TensorArray32<char>>("2_columns", dimensions_1, labels_1_1));
    std::shared_ptr<TensorAxis<int, Eigen::ThreadPoolDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisCpu<int>>(TensorAxisCpu<int>("1_indices", 1, 0));
    std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::ThreadPoolDevice>> table_1_axis_3_ptr = std::make_shared<TensorAxisCpu<TensorArray8<char>>>(TensorAxisCpu<TensorArray8<char>>("3_time", dimensions_3_t, labels_3_t));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
    table_1_ptr->addTensorAxis(table_1_axis_3_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData(); 
    std::map<std::string, int> shard_span_1;
    shard_span_1.emplace("2_columns", 1);
    shard_span_1.emplace("1_indices", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    shard_span_1.emplace("3_time", 6);
		table_1_ptr->setShardSpans(shard_span_1);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 3>({ data_size, 1, 6}));

    // Setup the tables
		std::shared_ptr<TensorTable<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> table_2_ptr = std::make_shared<TensorTableCpu<TensorArray32<char>, 2>>(TensorTableCpu<TensorArray32<char>, 2>("DataFrame_label"));
    std::shared_ptr<TensorAxis<TensorArray32<char>, Eigen::ThreadPoolDevice>> table_2_axis_1_ptr = std::make_shared<TensorAxisCpu<TensorArray32<char>>>(TensorAxisCpu<TensorArray32<char>>("2_columns", dimensions_1, labels_1_1));
    std::shared_ptr<TensorAxis<int, Eigen::ThreadPoolDevice>> table_2_axis_2_ptr = std::make_shared<TensorAxisCpu<int>>(TensorAxisCpu<int>("1_indices", 1, 0));
		table_2_axis_2_ptr->setDimensions(dimensions_2);
    table_2_ptr->addTensorAxis(table_2_axis_1_ptr);
		table_2_ptr->addTensorAxis(table_2_axis_2_ptr);
		table_2_ptr->setAxes(device);

		// Setup the table data
    table_2_ptr->setData();
    std::map<std::string, int> shard_span_2;
    shard_span_2.emplace("2_columns", 1);
    shard_span_2.emplace("1_indices", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    table_2_ptr->setShardSpans(shard_span_2);
    table_2_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 1}));

    // Setup the tables
    std::shared_ptr<TensorTable<float, Eigen::ThreadPoolDevice, 4>> table_3_ptr = std::make_shared<TensorTableCpu<float, 4>>(TensorTableCpu<float, 4>("DataFrame_image_2D"));
    std::shared_ptr<TensorAxis<TensorArray32<char>, Eigen::ThreadPoolDevice>> table_3_axis_1_ptr = std::make_shared<TensorAxisCpu<TensorArray32<char>>>(TensorAxisCpu<TensorArray32<char>>("2_columns", dimensions_1, labels_1_3));
    std::shared_ptr<TensorAxis<int, Eigen::ThreadPoolDevice>> table_3_axis_2_ptr = std::make_shared<TensorAxisCpu<int>>(TensorAxisCpu<int>("1_indices", 1, 0));
    std::shared_ptr<TensorAxis<int, Eigen::ThreadPoolDevice>> table_3_axis_3_ptr = std::make_shared<TensorAxisCpu<int>>(TensorAxisCpu<int>("3_x", dimensions_3_x, labels_3_x));
    std::shared_ptr<TensorAxis<int, Eigen::ThreadPoolDevice>> table_3_axis_4_ptr = std::make_shared<TensorAxisCpu<int>>(TensorAxisCpu<int>("3_y", dimensions_3_y, labels_3_y));
    table_3_axis_2_ptr->setDimensions(dimensions_2);
    table_3_ptr->addTensorAxis(table_3_axis_1_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_2_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_3_ptr);
    table_3_ptr->addTensorAxis(table_3_axis_4_ptr);
    table_3_ptr->setAxes(device);

    // Setup the table data
    table_3_ptr->setData();
    std::map<std::string, int> shard_span_3;
    shard_span_3.emplace("2_columns", 1);
    shard_span_3.emplace("1_indices", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
    shard_span_3.emplace("3_x", 28);
    shard_span_3.emplace("3_y", 28);
    table_3_ptr->setShardSpans(shard_span_3);
    table_3_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 4>({ data_size, 1, 28, 28 }));

    // Setup the tables
    std::shared_ptr<TensorTable<int, Eigen::ThreadPoolDevice, 2>> table_4_ptr = std::make_shared<TensorTableCpu<int, 2>>(TensorTableCpu<int, 2>("DataFrame_is_valid"));
    std::shared_ptr<TensorAxis<TensorArray32<char>, Eigen::ThreadPoolDevice>> table_4_axis_1_ptr = std::make_shared<TensorAxisCpu<TensorArray32<char>>>(TensorAxisCpu<TensorArray32<char>>("2_columns", dimensions_1, labels_1_1));
    std::shared_ptr<TensorAxis<int, Eigen::ThreadPoolDevice>> table_4_axis_2_ptr = std::make_shared<TensorAxisCpu<int>>(TensorAxisCpu<int>("1_indices", 1, 0));
    table_4_axis_2_ptr->setDimensions(dimensions_2);
    table_4_ptr->addTensorAxis(table_4_axis_1_ptr);
    table_4_ptr->addTensorAxis(table_4_axis_2_ptr);
    table_4_ptr->setAxes(device);

    // Setup the table data
    table_4_ptr->setData();
    table_4_ptr->setShardSpans(shard_span_2);
    table_4_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 1 }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionCpu>(TensorCollectionCpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_2_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_3_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_4_ptr, "DataFrame");
    // TODO: linking of axes
		return collection_1_ptr;
	}
};
#endif //TENSORBASE_BENCHMARKDATAFRAMECPU_H