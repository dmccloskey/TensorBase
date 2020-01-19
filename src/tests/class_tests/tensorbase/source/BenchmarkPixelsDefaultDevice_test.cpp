/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BenchmarkPixelsDefaultDevice test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/benchmarks/BenchmarkPixelsDefaultDevice.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;
using namespace std;

BOOST_AUTO_TEST_SUITE(benchmarkPixelsDefaultDevice)

BOOST_AUTO_TEST_CASE(InsertUpdateDelete0DDefaultDevice)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 0;
  const int data_size = 1296;
  const bool in_memory = true;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp;

  // Setup the TensorCollectionGenerator
  TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

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

BOOST_AUTO_TEST_CASE(InsertUpdateDelete1DDefaultDevice) 
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 1;
  const int data_size = 1296;
  const bool in_memory = true;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp;

  // Setup the TensorCollectionGenerator
  TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNDimensions(), 4);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);

  // Make the expected tensor axes labels and tensor data
  Eigen::Tensor<int, 2> labels(4, data_size);
  Eigen::Tensor<int, 2> values(1, data_size);
  for (int i = 0; i < data_size; ++i) {
    labels(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    labels(1, i) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
    labels(2, i) = int(floor(float(i) / float(std::pow(dim_span, 2)))) % dim_span + 1;
    labels(3, i) = int(floor(float(i) / float(std::pow(dim_span, 3)))) % dim_span + 1;
    values(0, i) = int(i);
  }

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNDimensions(), 4);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNLabels(), 1296);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNLabels(), 1);
  std::shared_ptr<int[]> labels_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getLabelsDataPointer(labels_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_insert_values(labels_insert_data.get(), 4, data_size);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK_EQUAL(labels_insert_values(i, j), labels(i, j));
    }
  }

  // Test the expected axis indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("values")->getTensorSize(), 1);
  for (int i = 0; i < 1; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("values")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("values")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyzt")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyzt")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyzt")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyzt")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyzt")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyzt")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyzt")->getData()(i), i + 1);
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values(data_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values(i, j), values(i, j));
    }
  }

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNDimensions(), 4);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNLabels(), 1296);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNLabels(), 1);
  std::shared_ptr<int[]> labels_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getLabelsDataPointer(labels_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_update_values(labels_update_data.get(), 4, data_size);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 4; ++j) {
      BOOST_CHECK_EQUAL(labels_update_values(i, j), labels(i, j));
    }
  }

  // Test the expected axis indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("values")->getTensorSize(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("values")->getTensorSize(), 1);
  for (int i = 0; i < 1; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("values")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("values")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyzt")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyzt")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyzt")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyzt")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyzt")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyzt")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyzt")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyzt")->getData()(i), i + 1);
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(data_update_values(i, j), -1);
    }
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNDimensions(), 4);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyzt")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("values")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);
}

BOOST_AUTO_TEST_CASE(InsertUpdateDelete2DDefaultDevice)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 2;
  const int data_size = 1296;
  const bool in_memory = false;// true;
  const double shard_span_perc = 0.2; // 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);
  const int xyz_dim_size = std::pow(dim_span, 3);
  const int t_dim_size = dim_span;

  // Setup the Benchmarking suite
  Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp;

  // Setup the TensorCollectionGenerator
  TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNDimensions(), 3);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNLabels(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);

  // Make the expected tensor axes labels and tensor data
  Eigen::Tensor<int, 2> labels_t(1, t_dim_size);
  Eigen::Tensor<int, 2> values(t_dim_size, xyz_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    labels_t(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    Eigen::Tensor<int, 1> new_values(xyz_dim_size);
    new_values.setConstant(i * xyz_dim_size + 1);
    new_values = new_values.cumsum(0);
    values.slice(Eigen::array<Eigen::Index, 2>({ i, 0 }), Eigen::array<Eigen::Index, 2>({ 1, xyz_dim_size })) = new_values.reshape(Eigen::array<Eigen::Index, 2>({ 1, xyz_dim_size }));
  }
  Eigen::Tensor<int, 2> labels_xyz(3, xyz_dim_size);
  for (int i = 0; i < xyz_dim_size; ++i) {
    labels_xyz(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    labels_xyz(1, i) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
    labels_xyz(2, i) = int(floor(float(i) / float(std::pow(dim_span, 2)))) % dim_span + 1;
  }

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNDimensions(), 3);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNLabels(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), t_dim_size);
  std::shared_ptr<int[]> labels_xyz_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getLabelsDataPointer(labels_xyz_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_xyz_insert_values(labels_xyz_insert_data.get(), 3, xyz_dim_size);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < xyz_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_xyz_insert_values(i, j), labels_xyz(i, j));
    }
  }
  std::shared_ptr<int[]> labels_t_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getLabelsDataPointer(labels_t_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_t_insert_values(labels_t_insert_data.get(), 1, t_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_t_insert_values(i, j), labels_t(i, j));
    }
  }

  // Test the expected axis indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyz")->getTensorSize(), xyz_dim_size);
  for (int i = 0; i < xyz_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyz")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyz")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyz")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyz")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyz")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyz")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getTensorSize(), t_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getData()(i), i + 1);
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values(data_insert_data.get(), t_dim_size, xyz_dim_size);
  //std::cout << "values\n" << values << std::endl;
  //std::cout << "data_insert_values\n" << data_insert_values << std::endl;
  for (int i = 0; i < t_dim_size; ++i) {
    for (int j = 0; j < xyz_dim_size; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values(i, j), values(i, j));
    }
  }

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNDimensions(), 3);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNLabels(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), t_dim_size);
  std::shared_ptr<int[]> labels_xyz_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getLabelsDataPointer(labels_xyz_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_xyz_update_values(labels_xyz_update_data.get(), 3, xyz_dim_size);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < xyz_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_xyz_update_values(i, j), labels_xyz(i, j));
    }
  }
  std::shared_ptr<int[]> labels_t_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getLabelsDataPointer(labels_t_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_t_update_values(labels_t_update_data.get(), 1, t_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_t_update_values(i, j), labels_t(i, j));
    }
  }

  // Test the expected axis indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyz")->getTensorSize(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyz")->getTensorSize(), xyz_dim_size);
  for (int i = 0; i < xyz_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xyz")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xyz")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xyz")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xyz")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xyz")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xyz")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getTensorSize(), t_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getData()(i), i + 1);
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data.get(), t_dim_size, xyz_dim_size);
  for (int i = 0; i < xyz_dim_size; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(data_update_values(i, j), -1);
    }
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNDimensions(), 3);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xyz")->getNLabels(), xyz_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 216);
}

BOOST_AUTO_TEST_CASE(InsertUpdateDelete3DDefaultDevice)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 3;
  const int data_size = 1296;
  const bool in_memory = true;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);
  const int xy_dim_size = std::pow(dim_span, 2);
  const int z_dim_size = dim_span;
  const int t_dim_size = dim_span;

  // Setup the Benchmarking suite
  Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp;

  // Setup the TensorCollectionGenerator
  TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNDimensions(), 2);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNLabels(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);

  // Make the expected tensor axes labels and tensor data
  Eigen::Tensor<int, 2> labels_t(1, t_dim_size);
  Eigen::Tensor<int, 3> values(t_dim_size, xy_dim_size, z_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    labels_t(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    Eigen::Tensor<int, 1> new_values(xy_dim_size * z_dim_size);
    new_values.setConstant(i * xy_dim_size * z_dim_size + 1);
    new_values = new_values.cumsum(0);
    values.slice(Eigen::array<Eigen::Index, 3>({ i, 0, 0 }), Eigen::array<Eigen::Index, 3>({ 1, xy_dim_size, z_dim_size })) = new_values.reshape(Eigen::array<Eigen::Index, 3>({ 1, xy_dim_size, z_dim_size }));
  }
  Eigen::Tensor<int, 2> labels_xy(2, xy_dim_size);
  for (int i = 0; i < xy_dim_size; ++i) {
    labels_xy(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    labels_xy(1, i) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
  }
  Eigen::Tensor<int, 2> labels_z(1, z_dim_size);
  for (int i = 0; i < z_dim_size; ++i) {
    labels_z(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
  }

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNDimensions(), 2);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNLabels(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), t_dim_size);
  std::shared_ptr<int[]> labels_xy_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getLabelsDataPointer(labels_xy_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_xy_insert_values(labels_xy_insert_data.get(), 2, xy_dim_size);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < xy_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_xy_insert_values(i, j), labels_xy(i, j));
    }
  }
  std::shared_ptr<int[]> labels_z_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getLabelsDataPointer(labels_z_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_z_insert_values(labels_z_insert_data.get(), 1, z_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < z_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_z_insert_values(i, j), labels_z(i, j));
    }
  }
  std::shared_ptr<int[]> labels_t_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getLabelsDataPointer(labels_t_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_t_insert_values(labels_t_insert_data.get(), 1, t_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_t_insert_values(i, j), labels_t(i, j));
    }
  }

  // Test the expected axis indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xy")->getTensorSize(), xy_dim_size);
  for (int i = 0; i < xy_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xy")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xy")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xy")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xy")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xy")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xy")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getTensorSize(), z_dim_size);
  for (int i = 0; i < z_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getTensorSize(), t_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getData()(i), i + 1);
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> data_insert_values(data_insert_data.get(), t_dim_size, xy_dim_size, z_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    for (int j = 0; j < xy_dim_size; ++j) {
      for (int k = 0; k < z_dim_size; ++k) {
        BOOST_CHECK_EQUAL(data_insert_values(i, j, k), values(i, j, k));
      }
    }
  }

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNDimensions(), 2);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNLabels(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), t_dim_size);
  std::shared_ptr<int[]> labels_xy_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getLabelsDataPointer(labels_xy_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_xy_update_values(labels_xy_update_data.get(), 2, xy_dim_size);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < xy_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_xy_update_values(i, j), labels_xy(i, j));
    }
  }
  std::shared_ptr<int[]> labels_z_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getLabelsDataPointer(labels_z_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_z_update_values(labels_z_update_data.get(), 1, z_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < z_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_z_update_values(i, j), labels_z(i, j));
    }
  }
  std::shared_ptr<int[]> labels_t_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getLabelsDataPointer(labels_t_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_t_update_values(labels_t_update_data.get(), 1, t_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_t_update_values(i, j), labels_t(i, j));
    }
  }

  // Test the expected axis indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xy")->getTensorSize(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xy")->getTensorSize(), xy_dim_size);
  for (int i = 0; i < xy_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("xy")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("xy")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("xy")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("xy")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("xy")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("xy")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getTensorSize(), z_dim_size);
  for (int i = 0; i < z_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getTensorSize(), t_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getData()(i), i + 1);
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> data_update_values(data_update_data.get(), t_dim_size, xy_dim_size, z_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    for (int j = 0; j < xy_dim_size; ++j) {
      for (int k = 0; k < z_dim_size; ++k) {
        BOOST_CHECK_EQUAL(data_update_values(i, j, k), -1);
      }
    }
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNDimensions(), 2);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("xy")->getNLabels(), xy_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 216);
}

BOOST_AUTO_TEST_CASE(InsertUpdateDelete4DDefaultDevice)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 4;
  const int data_size = 1296;
  const bool in_memory = true;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);
  const int x_dim_size = dim_span;
  const int y_dim_size = dim_span;
  const int z_dim_size = dim_span;
  const int t_dim_size = dim_span;

  // Setup the Benchmarking suite
  Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp;

  // Setup the TensorCollectionGenerator
  TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator;

  // Setup the device
  Eigen::DefaultDevice device;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNLabels(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNLabels(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);

  // Make the expected tensor axes labels and tensor data
  Eigen::Tensor<int, 2> labels_t(1, t_dim_size);
  Eigen::Tensor<int, 4> values(t_dim_size, x_dim_size, y_dim_size, z_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    labels_t(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    Eigen::Tensor<int, 1> new_values(x_dim_size * y_dim_size * z_dim_size);
    new_values.setConstant(i * x_dim_size * y_dim_size * z_dim_size + 1);
    new_values = new_values.cumsum(0);
    values.slice(Eigen::array<Eigen::Index, 4>({ i, 0, 0, 0 }), Eigen::array<Eigen::Index, 4>({ 1, x_dim_size, y_dim_size, z_dim_size })) = new_values.reshape(Eigen::array<Eigen::Index, 4>({ 1, x_dim_size, y_dim_size, z_dim_size }));
  }
  Eigen::Tensor<int, 2> labels_x(1, x_dim_size);
  Eigen::Tensor<int, 2> labels_y(1, y_dim_size);
  Eigen::Tensor<int, 2> labels_z(1, z_dim_size);
  for (int i = 0; i < x_dim_size; ++i) {
    labels_x(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    labels_y(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
    labels_z(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
  }

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNLabels(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNLabels(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), t_dim_size);
  std::shared_ptr<int[]> labels_x_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getLabelsDataPointer(labels_x_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_x_insert_values(labels_x_insert_data.get(), 1, x_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < x_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_x_insert_values(i, j), labels_x(i, j));
    }
  }
  std::shared_ptr<int[]> labels_y_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getLabelsDataPointer(labels_y_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_y_insert_values(labels_y_insert_data.get(), 1, y_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < y_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_y_insert_values(i, j), labels_y(i, j));
    }
  }
  std::shared_ptr<int[]> labels_z_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getLabelsDataPointer(labels_z_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_z_insert_values(labels_z_insert_data.get(), 1, z_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < z_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_z_insert_values(i, j), labels_z(i, j));
    }
  }
  std::shared_ptr<int[]> labels_t_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getLabelsDataPointer(labels_t_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_t_insert_values(labels_t_insert_data.get(), 1, t_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_t_insert_values(i, j), labels_t(i, j));
    }
  }

  // Test the expected axis indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("x")->getTensorSize(), x_dim_size);
  for (int i = 0; i < x_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("x")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("x")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("x")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("x")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("x")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("x")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("y")->getTensorSize(), y_dim_size);
  for (int i = 0; i < y_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("y")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("y")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("y")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("y")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("y")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("y")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getTensorSize(), z_dim_size);
  for (int i = 0; i < z_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getTensorSize(), t_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getData()(i), i + 1);
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_insert_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 4>> data_insert_values(data_insert_data.get(), t_dim_size, x_dim_size, y_dim_size, z_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    for (int j = 0; j < x_dim_size; ++j) {
      for (int k = 0; k < y_dim_size; ++k) {
        for (int l = 0; l < z_dim_size; ++l) {
          BOOST_CHECK_EQUAL(data_insert_values(i, j, k, l), values(i, j, k, l));
        }
      }
    }
  }

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNLabels(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNLabels(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), t_dim_size);
  std::shared_ptr<int[]> labels_x_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getLabelsDataPointer(labels_x_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_x_update_values(labels_x_update_data.get(), 1, x_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < x_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_x_update_values(i, j), labels_x(i, j));
    }
  }
  std::shared_ptr<int[]> labels_y_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getLabelsDataPointer(labels_y_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_y_update_values(labels_y_update_data.get(), 1, y_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < y_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_y_update_values(i, j), labels_y(i, j));
    }
  }
  std::shared_ptr<int[]> labels_z_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getLabelsDataPointer(labels_z_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_z_update_values(labels_z_update_data.get(), 1, z_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < z_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_z_update_values(i, j), labels_z(i, j));
    }
  }
  std::shared_ptr<int[]> labels_t_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getLabelsDataPointer(labels_t_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_t_update_values(labels_t_update_data.get(), 1, t_dim_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < t_dim_size; ++j) {
      BOOST_CHECK_EQUAL(labels_t_update_values(i, j), labels_t(i, j));
    }
  }

  // Test the expected axis indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("x")->getTensorSize(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("x")->getTensorSize(), x_dim_size);
  for (int i = 0; i < x_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("x")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("x")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("x")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("x")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("x")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("x")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("y")->getTensorSize(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("y")->getTensorSize(), y_dim_size);
  for (int i = 0; i < y_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("y")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("y")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("y")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("y")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("y")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("y")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getTensorSize(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getTensorSize(), z_dim_size);
  for (int i = 0; i < z_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("z")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("z")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("z")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("z")->getData()(i), i + 1);
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getTensorSize(), t_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getTensorSize(), t_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndices().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIndicesView().at("t")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getIsModified().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getNotInMemory().at("t")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardId().at("t")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getShardIndices().at("t")->getData()(i), i + 1);
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 1296);
  std::shared_ptr<int[]> data_update_data;
  n_dim_tensor_collection->tables_.at("TTable")->getDataPointer(data_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 4>> data_update_values(data_update_data.get(), t_dim_size, x_dim_size, y_dim_size, z_dim_size);
  for (int i = 0; i < t_dim_size; ++i) {
    for (int j = 0; j < x_dim_size; ++j) {
      for (int k = 0; k < y_dim_size; ++k) {
        for (int l = 0; l < z_dim_size; ++l) {
          BOOST_CHECK_EQUAL(data_update_values(i, j, k, l), -1);
        }
      }
    }
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("x")->getNLabels(), x_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("y")->getNLabels(), y_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("z")->getNLabels(), z_dim_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().at("t")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 216);
}

BOOST_AUTO_TEST_SUITE_END()