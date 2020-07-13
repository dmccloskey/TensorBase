/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BenchmarkDataFrameDefaultDefice test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/benchmarks/BenchmarkDataFrameDefaultDevice.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;
using namespace std;

BOOST_AUTO_TEST_SUITE(benchmarkPixelsDefaultDevice)

BOOST_AUTO_TEST_CASE(InsertUpdateDeleteDefaultDevice)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 0;
  const int data_size = 1296;
  const bool in_memory = true;
  const bool is_columnar = true;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  BenchmarkDataFrame1TimePointDefaultDevice benchmark_1_tp;

  // Setup the DataFrameTensorCollectionGenerator
  DataFrameTensorCollectionGeneratorDefaultDevice tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, is_columnar, device);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("columns")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardSpans().at("xyztv"), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardSpans().at("indices"), data_size);

  // Make the expected tensor axes labels and tensor data
  Eigen::Tensor<TensorArray8<char>, 2> labels_xyztv(1, 5);
  labels_xyztv.setValues({ { TensorArray8<char>("x"), TensorArray8<char>("y"), TensorArray8<char>("z"), TensorArray8<char>("t"), TensorArray8<char>("v")} });
  Eigen::Tensor<int, 2> values(data_size, 5);
  Eigen::Tensor<int, 2> labels_indices(1, data_size);
  for (int i = 0; i < data_size; ++i) {
    values(i, 0) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    values(i, 1) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
    values(i, 2) = int(floor(float(i) / float(std::pow(dim_span, 2)))) % dim_span + 1;
    values(i, 3) = int(floor(float(i) / float(std::pow(dim_span, 3)))) % dim_span + 1;
    values(i, 4) = int(i);
    labels_indices(0, i) = int(i);
  }

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 1296);
  std::shared_ptr<TensorArray8<char>[]> labels_xyztv_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getLabelsDataPointer(labels_xyztv_insert_data);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> labels_xyztv_insert_values(labels_xyztv_insert_data.get(), 5, 1);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 1; ++j) {
      BOOST_CHECK_EQUAL(labels_xyztv_insert_values(i, j), labels_xyztv(i, j));
    }
  }
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_insert_values(i, j), labels_indices(i, j));
    }
  }

  // Test the expected axis indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getTensorSize(), 5);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getData()(i), i+1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getData()(i), i + 1);
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 6480);
  std::shared_ptr<int[]> data_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values(data_insert_data.get(), data_size, 5);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 5; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values(i, j), values(i, j));
    }
  }

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 1296);
  std::shared_ptr<TensorArray8<char>[]> labels_xyztv_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getLabelsDataPointer(labels_xyztv_update_data);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> labels_xyztv_update_values(labels_xyztv_update_data.get(), 5, 1);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 1; ++j) {
      BOOST_CHECK_EQUAL(labels_xyztv_update_values(i, j), labels_xyztv(i, j));
    }
  }
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_update_values(i, j), labels_indices(i, j));
    }
  }

  // Test the expected axis indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getTensorSize(), 5);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getData()(i), i + 1);
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 6480);
  std::shared_ptr<int[]> data_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data.get(), data_size, 5);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 5; ++j) {
      if (j==4) BOOST_CHECK_EQUAL(data_update_values(i, j), -1);
      else BOOST_CHECK_EQUAL(data_update_values(i, j), values(i, j));
    }
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
}

// repeat with sharding
BOOST_AUTO_TEST_CASE(InsertUpdateDeleteShardingDefaultDevice)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 0;
  const int data_size = 1296;
  const bool in_memory = false;
  const bool is_columnar = true;
  const double shard_span_perc = 0.05;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  BenchmarkDataFrame1TimePointDefaultDevice benchmark_1_tp;

  // Setup the DataFrameTensorCollectionGenerator
  DataFrameTensorCollectionGeneratorDefaultDevice tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, is_columnar, device);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardSpans().at("xyztv"), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardSpans().at("indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getMaxDimSizeFromAxisName("xyztv"), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getMaxDimSizeFromAxisName("indices"), data_size);

  // Make the expected tensor axes labels and tensor data
  Eigen::Tensor<TensorArray8<char>, 2> labels_xyztv(1, 5);
  labels_xyztv.setValues({ { TensorArray8<char>("x"), TensorArray8<char>("y"), TensorArray8<char>("z"), TensorArray8<char>("t"), TensorArray8<char>("v")} });
  Eigen::Tensor<int, 2> values(data_size, 5);
  Eigen::Tensor<int, 2> labels_indices(1, data_size);
  for (int i = 0; i < data_size; ++i) {
    values(i, 0) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    values(i, 1) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
    values(i, 2) = int(floor(float(i) / float(std::pow(dim_span, 2)))) % dim_span + 1;
    values(i, 3) = int(floor(float(i) / float(std::pow(dim_span, 3)))) % dim_span + 1;
    values(i, 4) = int(i);
    labels_indices(0, i) = int(i);
  }

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 1296);
  std::shared_ptr<TensorArray8<char>[]> labels_xyztv_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getLabelsDataPointer(labels_xyztv_insert_data);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> labels_xyztv_insert_values(labels_xyztv_insert_data.get(), 5, 1);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 1; ++j) {
      BOOST_CHECK_EQUAL(labels_xyztv_insert_values(i, j), labels_xyztv(i, j));
    }
  }
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_insert_values(i, j), labels_indices(i, j));
    }
  }

  // Test the expected axis indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getTensorSize(), 5);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
  n_dim_tensor_collection->tables_.at("TTable")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("TTable")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 6480);
  std::shared_ptr<int[]> data_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values(data_insert_data.get(), data_size, 5);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 5; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values(i, j), values(i, j));
    }
  }
  n_dim_tensor_collection->tables_.at("TTable")->initData(device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 1296);
  std::shared_ptr<TensorArray8<char>[]> labels_xyztv_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getLabelsDataPointer(labels_xyztv_update_data);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> labels_xyztv_update_values(labels_xyztv_update_data.get(), 5, 1);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 1; ++j) {
      BOOST_CHECK_EQUAL(labels_xyztv_update_values(i, j), labels_xyztv(i, j));
    }
  }
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_update_values(i, j), labels_indices(i, j));
    }
  }

  // Test the expected axis indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getTensorSize(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getTensorSize(), 5);
  for (int i = 0; i < 5; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyztv")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyztv")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyztv")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyztv")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
  n_dim_tensor_collection->tables_.at("TTable")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("TTable")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 6480);
  std::shared_ptr<int[]> data_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data.get(), data_size, 5);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 5; ++j) {
      if (j == 4) BOOST_CHECK_EQUAL(data_update_values(i, j), -1);
      else BOOST_CHECK_EQUAL(data_update_values(i, j), values(i, j));
    }
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyztv")->getNLabels(), 5);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
}

BOOST_AUTO_TEST_SUITE_END()