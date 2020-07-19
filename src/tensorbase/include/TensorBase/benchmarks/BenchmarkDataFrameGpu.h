/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKDATAFRAMEGPU_H
#define TENSORBASE_BENCHMARKDATAFRAMEGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkDataFrame.h>
#include <TensorBase/ml/TensorCollectionGpu.h>
#include <TensorBase/ml/TensorOperationGpu.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
  /*
  @class Specialized `SelectTableDataIsValid` for the Gpu case
  */
  template<template<class> class ArrayT, class TensorT>
  class SelectTableDataIsValidGpuT: public SelectTableDataIsValid<ArrayT<TensorT>, int, Eigen::GpuDevice> {};
  class SelectTableDataIsValidGpu : public SelectTableDataIsValidGpuT<TensorArrayGpu32, char> {
  public:
    void setLabelsValuesResult(Eigen::GpuDevice& device) override;
  };
  inline void SelectTableDataIsValidGpu::setLabelsValuesResult(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArrayGpu32<char>, 2> select_labels_values(1, 1);
    select_labels_values.setConstant(TensorArrayGpu32<char>("is_valid"));
    TensorDataGpuClassT<TensorArrayGpu32, char, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu32, char, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<int, 1> select_values_values(1);
    select_values_values.setConstant(1);
    TensorDataGpuPrimitiveT<int, 1> select_values(select_values_values.dimensions());
    select_values.setData(select_values_values);
    select_values.syncHAndDData(device);
    this->select_values_ = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_values);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<int, 1> results(Eigen::array<Eigen::Index, 1>({1}));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(results);
  }

  /*
  @class Specialized `SelectTableDataLabel` for the Gpu case
  */
  template<template<class> class ArrayT1, class TensorT1, template<class> class ArrayT2, class TensorT2>
  class SelectTableDataLabelGpuT : public SelectTableDataLabel<ArrayT1<TensorT1>, ArrayT1<TensorT1>, Eigen::GpuDevice> {};
  class SelectTableDataLabelGpu : public SelectTableDataLabelGpuT<TensorArrayGpu32, char, TensorArrayGpu32, char> {
  public:
    void setLabelsValuesResult(Eigen::GpuDevice& device) override;
  };
  inline void SelectTableDataLabelGpu::setLabelsValuesResult(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArrayGpu32<char>, 2> select_labels_values(1, 1);
    select_labels_values.setConstant(TensorArrayGpu32<char>("label"));
    TensorDataGpuClassT<TensorArrayGpu32, char, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu32, char, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<TensorArrayGpu32<char>, 1> select_values_values(1);
    select_values_values.setConstant(TensorArrayGpu32<char>("one"));
    TensorDataGpuClassT<TensorArrayGpu32, char, 1> select_values(select_values_values.dimensions());
    select_values.setData(select_values_values);
    select_values.syncHAndDData(device);
    this->select_values_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu32, char, 1>>(select_values);
  }
  
  /*
  @class Specialized `SelectTableDataImage2D` for the Gpu case
  */
  template<template<class> class ArrayT, class TensorT>
  class SelectTableDataImage2DGpuT : public SelectTableDataImage2D<ArrayT<TensorT>, int, float, Eigen::GpuDevice> {};
  class SelectTableDataImage2DGpu : public SelectTableDataImage2DGpuT<TensorArrayGpu8, char> {
  public:
    void setLabelsValuesResult(Eigen::GpuDevice& device) override;
  };
  inline void SelectTableDataImage2DGpu::setLabelsValuesResult(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArrayGpu8<char>, 2> select_labels_values(1, 2);
    select_labels_values.setValues({ { TensorArrayGpu8<char>("day"),TensorArrayGpu8<char>("month") } });
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> select_labels(select_labels_values.dimensions());
    select_labels.setData(select_labels_values);
    select_labels.syncHAndDData(device);
    this->select_labels_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(select_labels);

    // make the corresponding values and sync to the device
    Eigen::Tensor<int, 1> select_values_values(2);
    select_values_values.setValues({14, 1});
    TensorDataGpuPrimitiveT<int, 1> select_values_lt(select_values_values.dimensions());
    select_values_lt.setData(select_values_values);
    select_values_lt.syncHAndDData(device);
    this->select_values_lt_ = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_values_lt);
    select_values_values.setValues({ 0, 1 });
    TensorDataGpuPrimitiveT<int, 1> select_values_gt(select_values_values.dimensions());
    select_values_gt.setData(select_values_values);
    select_values_gt.syncHAndDData(device);
    this->select_values_gt_ = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_values_gt);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<float, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(results);
  }

	/*
	@class Specialized `DataFrameManager` for the Gpu case
	*/
	class DataFrameManagerTimeGpu : public DataFrameManagerTime<int, int, Eigen::GpuDevice> {
	public:
		using DataFrameManagerTime::DataFrameManagerTime;
		void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<int, 3>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>>& values_ptr);
	};
	void DataFrameManagerTimeGpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr) {
		TensorDataGpuPrimitiveT<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_data);
	}
	void DataFrameManagerTimeGpu::makeValuesPtr(const Eigen::Tensor<int, 3>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>>& values_ptr) {
		TensorDataGpuPrimitiveT<int, 3> values_data(Eigen::array<Eigen::Index, 3>({ values.dimension(0), values.dimension(1), values.dimension(2) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(values_data);
	}

  /*
  @class Specialized `DataFrameManager` for the Gpu case
  */
  class DataFrameManagerLabelGpu : public DataFrameManagerLabel<int, TensorArrayGpu32<char>, Eigen::GpuDevice> {
  public:
    using DataFrameManagerLabel::DataFrameManagerLabel;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<TensorArrayGpu32<char>, 2>& values, std::shared_ptr<TensorData<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>>& values_ptr);
  };
  void DataFrameManagerLabelGpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr) {
    TensorDataGpuPrimitiveT<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_data);
  }
  void DataFrameManagerLabelGpu::makeValuesPtr(const Eigen::Tensor<TensorArrayGpu32<char>, 2>& values, std::shared_ptr<TensorData<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>>& values_ptr) {
    TensorDataGpuClassT<TensorArrayGpu32, char, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu32, char, 2>>(values_data);
  }

  /*
  @class Specialized `DataFrameManager` for the Gpu case
  */
  class DataFrameManagerImage2DGpu : public DataFrameManagerImage2D<int, float, Eigen::GpuDevice> {
  public:
    using DataFrameManagerImage2D::DataFrameManagerImage2D;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<float, 4>& values, std::shared_ptr<TensorData<float, Eigen::GpuDevice, 4>>& values_ptr);
  };
  void DataFrameManagerImage2DGpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr) {
    TensorDataGpuPrimitiveT<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_data);
  }
  void DataFrameManagerImage2DGpu::makeValuesPtr(const Eigen::Tensor<float, 4>& values, std::shared_ptr<TensorData<float, Eigen::GpuDevice, 4>>& values_ptr) {
    TensorDataGpuPrimitiveT<float, 4> values_data(Eigen::array<Eigen::Index, 4>({ values.dimension(0), values.dimension(1), values.dimension(2), values.dimension(3) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 4>>(values_data);
  }

  /*
  @class Specialized `DataFrameManager` for the Gpu case
  */
  class DataFrameManagerIsValidGpu : public DataFrameManagerIsValid<int, int, Eigen::GpuDevice> {
  public:
    using DataFrameManagerIsValid::DataFrameManagerIsValid;
    void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr);
    void makeValuesPtr(const Eigen::Tensor<int, 2>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& values_ptr);
  };
  void DataFrameManagerIsValidGpu::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& labels_ptr) {
    TensorDataGpuPrimitiveT<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
    labels_data.setData(labels);
    labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_data);
  }
  void DataFrameManagerIsValidGpu::makeValuesPtr(const Eigen::Tensor<int, 2>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& values_ptr) {
    TensorDataGpuPrimitiveT<int, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(values_data);
  }

	/*
	@class A class for running 1 line insertion, deletion, and update benchmarks
	*/
	class BenchmarkDataFrame1TimePointGpu : public BenchmarkDataFrame1TimePoint<DataFrameManagerTimeGpu, DataFrameManagerLabelGpu, DataFrameManagerImage2DGpu, DataFrameManagerIsValidGpu, Eigen::GpuDevice> {
	protected:
		void _insert1TimePoint(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
		void _update1TimePoint(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
		void _delete1TimePoint(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1TimePoint0D`
	};
	void BenchmarkDataFrame1TimePointGpu::_insert1TimePoint(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
    DataFrameManagerTimeGpu dataframe_manager_time(data_size, false);
    DataFrameManagerLabelGpu dataframe_manager_labels(data_size, false);
    DataFrameManagerImage2DGpu dataframe_manager_image_2d(data_size, false);
    DataFrameManagerIsValidGpu dataframe_manager_is_valid(data_size, false);
    int span = data_size / std::pow(data_size, 0.25);
    for (int i = 0; i < data_size; i += span) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_time_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> values_time_ptr;
      dataframe_manager_time.getInsertData(i, span, labels_time_ptr, values_time_ptr);
      TensorAppendToAxis<int, int, Eigen::GpuDevice, 3> appendToAxis_time("DataFrame_time", "1_indices", labels_time_ptr, values_time_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_time_ptr = std::make_shared<TensorAppendToAxis<int, int, Eigen::GpuDevice, 3>>(appendToAxis_time);
      transaction_manager.executeOperation(appendToAxis_time_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_labels_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>> values_labels_ptr;
      dataframe_manager_labels.getInsertData(i, span, labels_labels_ptr, values_labels_ptr);
      TensorAppendToAxis<int, TensorArrayGpu32<char>, Eigen::GpuDevice, 2> appendToAxis_labels("DataFrame_label", "1_indices", labels_labels_ptr, values_labels_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_labels_ptr = std::make_shared<TensorAppendToAxis<int, TensorArrayGpu32<char>, Eigen::GpuDevice, 2>>(appendToAxis_labels);
      transaction_manager.executeOperation(appendToAxis_labels_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_image_2d_ptr;
      std::shared_ptr<TensorData<float, Eigen::GpuDevice, 4>> values_image_2d_ptr;
      dataframe_manager_image_2d.getInsertData(i, span, labels_image_2d_ptr, values_image_2d_ptr);
      TensorAppendToAxis<int, float, Eigen::GpuDevice, 4> appendToAxis_image_2d("DataFrame_image_2D", "1_indices", labels_image_2d_ptr, values_image_2d_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_image_2d_ptr = std::make_shared<TensorAppendToAxis<int, float, Eigen::GpuDevice, 4>>(appendToAxis_image_2d);
      transaction_manager.executeOperation(appendToAxis_image_2d_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_is_valid_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_is_valid_ptr;
      dataframe_manager_is_valid.getInsertData(i, span, labels_is_valid_ptr, values_is_valid_ptr);
      TensorAppendToAxis<int, int, Eigen::GpuDevice, 2> appendToAxis_is_valid("DataFrame_is_valid", "1_indices", labels_is_valid_ptr, values_is_valid_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> appendToAxis_is_valid_ptr = std::make_shared<TensorAppendToAxis<int, int, Eigen::GpuDevice, 2>>(appendToAxis_is_valid);
      transaction_manager.executeOperation(appendToAxis_is_valid_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
	}
	void BenchmarkDataFrame1TimePointGpu::_update1TimePoint(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
    DataFrameManagerTimeGpu dataframe_manager_time(data_size, true);
    DataFrameManagerLabelGpu dataframe_manager_labels(data_size, true);
    DataFrameManagerImage2DGpu dataframe_manager_image_2d(data_size, true);
    DataFrameManagerIsValidGpu dataframe_manager_is_valid(data_size, true);
    int span = data_size / std::pow(data_size, 0.25);
    for (int i = 0; i < data_size; i += span) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_time_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> values_time_ptr;
      dataframe_manager_time.getInsertData(i, span, labels_time_ptr, values_time_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_time(labels_time_ptr, "DataFrame_time");
      TensorUpdateValues<int, Eigen::GpuDevice, 3> updateValues_time("DataFrame_time", selectClause_time, values_time_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_time_ptr = std::make_shared<TensorUpdateValues<int, Eigen::GpuDevice, 3>>(updateValues_time);
      transaction_manager.executeOperation(updateValues_time_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_labels_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>> values_labels_ptr;
      dataframe_manager_labels.getInsertData(i, span, labels_labels_ptr, values_labels_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_labels(labels_labels_ptr, "DataFrame_label");
      TensorUpdateValues<TensorArrayGpu32<char>, Eigen::GpuDevice, 2> updateValues_labels("DataFrame_label", selectClause_labels, values_labels_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_labels_ptr = std::make_shared<TensorUpdateValues<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>>(updateValues_labels);
      transaction_manager.executeOperation(updateValues_labels_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_image_2d_ptr;
      std::shared_ptr<TensorData<float, Eigen::GpuDevice, 4>> values_image_2d_ptr;
      dataframe_manager_image_2d.getInsertData(i, span, labels_image_2d_ptr, values_image_2d_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_image_2d(labels_image_2d_ptr, "DataFrame_image_2D");
      TensorUpdateValues<float, Eigen::GpuDevice, 4> updateValues_image_2d("DataFrame_image_2D", selectClause_image_2d, values_image_2d_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_image_2d_ptr = std::make_shared<TensorUpdateValues<float, Eigen::GpuDevice, 4>>(updateValues_image_2d);
      transaction_manager.executeOperation(updateValues_image_2d_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_is_valid_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_is_valid_ptr;
      dataframe_manager_is_valid.getInsertData(i, span, labels_is_valid_ptr, values_is_valid_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_is_valid(labels_is_valid_ptr, "DataFrame_is_valid");
      TensorUpdateValues<int, Eigen::GpuDevice, 2> updateValues_is_valid("DataFrame_is_valid", selectClause_is_valid, values_is_valid_ptr);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> updateValues_is_valid_ptr = std::make_shared<TensorUpdateValues<int, Eigen::GpuDevice, 2>>(updateValues_is_valid);
      transaction_manager.executeOperation(updateValues_is_valid_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
    }
	}
	void BenchmarkDataFrame1TimePointGpu::_delete1TimePoint(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
    DataFrameManagerTimeGpu dataframe_manager_time(data_size, true);
    DataFrameManagerLabelGpu dataframe_manager_labels(data_size, true);
    DataFrameManagerImage2DGpu dataframe_manager_image_2d(data_size, true);
    DataFrameManagerIsValidGpu dataframe_manager_is_valid(data_size, true);
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_time_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> values_time_ptr;
      dataframe_manager_time.getInsertData(i, span, labels_time_ptr, values_time_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_time(labels_time_ptr, "DataFrame_time");
      TensorDeleteFromAxisGpuPrimitiveT<int, int, 3> tensorDelete_time("DataFrame_time", "1_indices", selectClause_time);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_time_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<int, int, 3>>(tensorDelete_time);
      transaction_manager.executeOperation(tensorDelete_time_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_labels_ptr;
      std::shared_ptr<TensorData<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>> values_labels_ptr;
      dataframe_manager_labels.getInsertData(i, span, labels_labels_ptr, values_labels_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_labels(labels_labels_ptr, "DataFrame_label");
      TensorDeleteFromAxisGpuClassT<int, TensorArrayGpu32, char, 2> tensorDelete_labels("DataFrame_label", "1_indices", selectClause_labels);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_labels_ptr = std::make_shared<TensorDeleteFromAxisGpuClassT<int, TensorArrayGpu32, char, 2>>(tensorDelete_labels);
      transaction_manager.executeOperation(tensorDelete_labels_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_image_2d_ptr;
      std::shared_ptr<TensorData<float, Eigen::GpuDevice, 4>> values_image_2d_ptr;
      dataframe_manager_image_2d.getInsertData(i, span, labels_image_2d_ptr, values_image_2d_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_image_2d(labels_image_2d_ptr, "DataFrame_image_2D");
      TensorDeleteFromAxisGpuPrimitiveT<int, float, 4> tensorDelete_image_2d("DataFrame_image_2D", "1_indices", selectClause_image_2d);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_image_2d_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<int, float, 4>>(tensorDelete_image_2d);
      transaction_manager.executeOperation(tensorDelete_image_2d_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_is_valid_ptr;
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_is_valid_ptr;
      dataframe_manager_is_valid.getInsertData(i, span, labels_is_valid_ptr, values_is_valid_ptr);
      SelectTableDataIndices<int, Eigen::GpuDevice> selectClause_is_valid(labels_is_valid_ptr, "DataFrame_is_valid");
      TensorDeleteFromAxisGpuPrimitiveT<int, int, 2> tensorDelete_is_valid("DataFrame_is_valid", "1_indices", selectClause_is_valid);
      std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_is_valid_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<int, int, 2>>(tensorDelete_is_valid);
      transaction_manager.executeOperation(tensorDelete_is_valid_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}

	class DataFrameTensorCollectionGeneratorGpu : public DataFrameTensorCollectionGenerator<Eigen::GpuDevice> {
	public:
    std::shared_ptr<TensorCollection<Eigen::GpuDevice>> makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::GpuDevice& device) const override;
  };
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> DataFrameTensorCollectionGeneratorGpu::makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::GpuDevice& device) const
	{
		// Setup the axes
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(1), dimensions_3_t(1), dimensions_3_x(1), dimensions_3_y(1);
		dimensions_1.setValues({ "columns" });
		dimensions_2.setValues({ "indices" });
    dimensions_3_t.setValues({ "time" });
    dimensions_3_x.setValues({ "x" });
    dimensions_3_y.setValues({ "y" });
    Eigen::Tensor<TensorArrayGpu32<char>, 2> labels_1_1(1, 1), labels_1_2(1, 1), labels_1_3(1, 1), labels_1_4(1, 1);
    Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_3_t(1, 6);
    Eigen::Tensor<int, 2> labels_3_x(1, 28), labels_3_y(1, 28);
    labels_1_1.setValues({ { TensorArrayGpu32<char>("time")} });
    labels_1_2.setValues({ { TensorArrayGpu32<char>("label")} });
    labels_1_3.setValues({ { TensorArrayGpu32<char>("image_2D")} });
    labels_1_4.setValues({ { TensorArrayGpu32<char>("is_valid")} });
    labels_3_t.setValues({ { TensorArrayGpu8<char>("sec"), TensorArrayGpu8<char>("min"), TensorArrayGpu8<char>("hour"), TensorArrayGpu8<char>("day"), TensorArrayGpu8<char>("month"), TensorArrayGpu8<char>("year")} });
    labels_3_x.setZero();
    labels_3_x = labels_3_x.constant(1).cumsum(1);
    labels_3_y.setZero();
    labels_3_y = labels_3_y.constant(1).cumsum(1);

		// Setup the tables
		std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 3>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 3>>(TensorTableGpuPrimitiveT<int, 3>("DataFrame_time"));
		std::shared_ptr<TensorAxis<TensorArrayGpu32<char>, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu32, char>>(TensorAxisGpuClassT<TensorArrayGpu32, char>("2_columns", dimensions_1, labels_1_1));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_indices", 1, 0));
    std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_1_axis_3_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("3_time", dimensions_3_t, labels_3_t));
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
		std::shared_ptr<TensorTable<TensorArrayGpu32<char>, Eigen::GpuDevice, 2>> table_2_ptr = std::make_shared<TensorTableGpuClassT<TensorArrayGpu32, char, 2>>(TensorTableGpuClassT<TensorArrayGpu32, char, 2>("DataFrame_label"));
    std::shared_ptr<TensorAxis<TensorArrayGpu32<char>, Eigen::GpuDevice>> table_2_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu32, char>>(TensorAxisGpuClassT<TensorArrayGpu32, char>("2_columns", dimensions_1, labels_1_1));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_2_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_indices", 1, 0));
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
    std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 4>> table_3_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 4>>(TensorTableGpuPrimitiveT<float, 4>("DataFrame_image_2D"));
    std::shared_ptr<TensorAxis<TensorArrayGpu32<char>, Eigen::GpuDevice>> table_3_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu32, char>>(TensorAxisGpuClassT<TensorArrayGpu32, char>("2_columns", dimensions_1, labels_1_3));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_3_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_indices", 1, 0));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_3_axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3_x", dimensions_3_x, labels_3_x));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_3_axis_4_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3_y", dimensions_3_y, labels_3_y));
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
    std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> table_4_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(TensorTableGpuPrimitiveT<int, 2>("DataFrame_is_valid"));
    std::shared_ptr<TensorAxis<TensorArrayGpu32<char>, Eigen::GpuDevice>> table_4_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu32, char>>(TensorAxisGpuClassT<TensorArrayGpu32, char>("2_columns", dimensions_1, labels_1_1));
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> table_4_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1_indices", 1, 0));
    table_4_axis_2_ptr->setDimensions(dimensions_2);
    table_4_ptr->addTensorAxis(table_4_axis_1_ptr);
    table_4_ptr->addTensorAxis(table_4_axis_2_ptr);
    table_4_ptr->setAxes(device);

    // Setup the table data
    table_4_ptr->setData();
    table_4_ptr->setShardSpans(shard_span_2);
    table_4_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 1 }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_2_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_3_ptr, "DataFrame");
    collection_1_ptr->addTensorTable(table_4_ptr, "DataFrame");
    // TODO: linking of axes
		return collection_1_ptr;
	}
};
#endif
#endif //TENSORBASE_BENCHMARKDATAFRAMEGPU_H