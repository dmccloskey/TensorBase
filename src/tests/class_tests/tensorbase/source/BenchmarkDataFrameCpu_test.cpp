/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BenchmarkDataFrameCpu test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/benchmarks/BenchmarkDataFrameCpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;
using namespace std;

BOOST_AUTO_TEST_SUITE(benchmarkDataFrameCpu)

BOOST_AUTO_TEST_CASE(InsertUpdateDeleteCpu)
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
  BenchmarkDataFrame1TimePointCpu benchmark_1_tp;

  // Setup the DataFrameTensorCollectionGenerator
  DataFrameTensorCollectionGeneratorCpu tensor_collection_generator;

  // Setup the device
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::ThreadPoolDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, is_columnar, device);

  // Setup the transaction manager
  TransactionManager<Eigen::ThreadPoolDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("3_time")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("3_time")->getNLabels(), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardSpans().at("3_time"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getMaxDimSizeFromAxisName("3_time"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_x")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_y")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getMaxDimSizeFromAxisName("1_indices"), data_size);

  // Make the expected tensor axes labels and tensor data after insert
  DataFrameManagerTimeCpu dataframe_manager_time(data_size, false);
  DataFrameManagerLabelCpu dataframe_manager_labels(data_size, false);
  DataFrameManagerImage2DCpu dataframe_manager_image_2d(data_size, false);
  DataFrameManagerIsValidCpu dataframe_manager_is_valid(data_size, false);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_time_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_time_ptr;
  dataframe_manager_time.getInsertData(0, data_size, labels_time_ptr, values_time_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
  std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
  dataframe_manager_labels.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_image_2d_ptr;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_image_2d_ptr;
  dataframe_manager_image_2d.getInsertData(0, data_size, labels_image_2d_ptr, values_image_2d_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_is_valid_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_is_valid_ptr;
  dataframe_manager_is_valid.getInsertData(0, data_size, labels_is_valid_ptr, values_is_valid_ptr);

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_insert_values(i, j), labels_time_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 1 * data_size * 6);
  std::shared_ptr<int[]> data_insert_data_time;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataPointer(data_insert_data_time);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> data_insert_values_time(data_insert_data_time.get(), data_size, 1, 6);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 6; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values_time(i, 0, j), values_time_ptr->getData()(i, 0, j));
    }
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray32<char>[]> data_insert_data_labels;
  n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataPointer(data_insert_data_labels);
  Eigen::TensorMap<Eigen::Tensor<TensorArray32<char>, 2>> data_insert_values_labels(data_insert_data_labels.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_labels(i, 0), values_labels_ptr->getData()(i, 0));
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 1 * data_size * 28 * 28);
  std::shared_ptr<float[]> data_insert_data_image_2d;
  n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataPointer(data_insert_data_image_2d);
  Eigen::TensorMap<Eigen::Tensor<float, 4>> data_insert_values_image_2d(data_insert_data_image_2d.get(), data_size, 1, 28, 28);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 28; ++j) {
      for (int k = 0; k < 28; ++k) {
        BOOST_CHECK_EQUAL(data_insert_values_image_2d(i, 0, j, k), values_image_2d_ptr->getData()(i, 0, j, k));
      }
    }
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<int[]> data_insert_data_is_valid;
  n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataPointer(data_insert_data_is_valid);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values_is_valid(data_insert_data_is_valid.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_is_valid(i, 0), values_is_valid_ptr->getData()(i, 0));
  }

  // Query for the number of valid entries
  SelectTableDataIsValidCpu select_is_valid;
  select_is_valid(n_dim_tensor_collection, device);
  BOOST_CHECK_EQUAL(select_is_valid.result_->getData()(0), data_size / 2);

  // Query for the number of labels = "one"
  SelectTableDataLabelCpu select_label_ones;
  select_label_ones(n_dim_tensor_collection, device);
  BOOST_CHECK_EQUAL(select_label_ones.result_, 130);

  // Query for the average pixel intensity in the first two weeks of January
  SelectTableDataImage2DCpu select_2D_image;
  select_2D_image(n_dim_tensor_collection, device);
  BOOST_CHECK_CLOSE(select_2D_image.result_->getData()(0), 0, 1e-3);

  // Make the expected tensor axes labels and tensor data after update
  dataframe_manager_time.initTime();
  dataframe_manager_time.setUseRandomValues(true);
  dataframe_manager_labels.setUseRandomValues(true);
  dataframe_manager_image_2d.setUseRandomValues(true);
  dataframe_manager_is_valid.setUseRandomValues(true);
  labels_time_ptr.reset();
  values_time_ptr.reset();
  dataframe_manager_time.getInsertData(0, data_size, labels_time_ptr, values_time_ptr);
  labels_labels_ptr.reset();
  values_labels_ptr.reset();
  dataframe_manager_labels.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  labels_image_2d_ptr.reset();
  values_image_2d_ptr.reset();
  dataframe_manager_image_2d.getInsertData(0, data_size, labels_image_2d_ptr, values_image_2d_ptr);
  labels_is_valid_ptr.reset();
  values_is_valid_ptr.reset();
  dataframe_manager_is_valid.getInsertData(0, data_size, labels_is_valid_ptr, values_is_valid_ptr);

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_update_values(i, j), labels_time_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 1 * data_size * 6);
  std::shared_ptr<int[]> data_update_data_time;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataPointer(data_update_data_time);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> data_update_values(data_update_data_time.get(), data_size, 1, 6);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 6; ++j) {
      BOOST_CHECK_EQUAL(data_update_values(i, 0, j), values_time_ptr->getData()(i, 0, j));
    }
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray32<char>[]> data_update_data_labels;
  n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataPointer(data_update_data_labels);
  Eigen::TensorMap<Eigen::Tensor<TensorArray32<char>, 2>> data_update_values_labels(data_update_data_labels.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_labels(i, 0), values_labels_ptr->getData()(i, 0));
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 1 * data_size * 28 * 28);
  std::shared_ptr<float[]> data_update_data_image_2d;
  n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataPointer(data_update_data_image_2d);
  Eigen::TensorMap<Eigen::Tensor<float, 4>> data_update_values_image_2d(data_update_data_image_2d.get(), data_size, 1, 28, 28);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 28; ++j) {
      for (int k = 0; k < 28; ++k) {
        BOOST_CHECK_EQUAL(data_update_values_image_2d(i, 0, j, k), values_image_2d_ptr->getData()(i, 0, j, k));
      }
    }
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<int[]> data_update_data_is_valid;
  n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataPointer(data_update_data_is_valid);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values_is_valid(data_update_data_is_valid.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_is_valid(i, 0), values_is_valid_ptr->getData()(i, 0));
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 0);
}

// repeat with sharding
BOOST_AUTO_TEST_CASE(InsertUpdateDeleteShardingCpu)
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
  BenchmarkDataFrame1TimePointCpu benchmark_1_tp;

  // Setup the DataFrameTensorCollectionGenerator
  DataFrameTensorCollectionGeneratorCpu tensor_collection_generator;

  // Setup the device
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::ThreadPoolDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(data_size, shard_span_perc, is_columnar, device);

  // Setup the transaction manager
  TransactionManager<Eigen::ThreadPoolDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("3_time")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("3_time")->getNLabels(), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardSpans().at("3_time"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getMaxDimSizeFromAxisName("3_time"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_x")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("3_y")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getShardSpans().at("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getMaxDimSizeFromAxisName("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getMaxDimSizeFromAxisName("1_indices"), data_size);

  // Make the expected tensor axes labels and tensor data after insert
  DataFrameManagerTimeCpu dataframe_manager_time(data_size, false);
  DataFrameManagerLabelCpu dataframe_manager_labels(data_size, false);
  DataFrameManagerImage2DCpu dataframe_manager_image_2d(data_size, false);
  DataFrameManagerIsValidCpu dataframe_manager_is_valid(data_size, false);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_time_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_time_ptr;
  dataframe_manager_time.getInsertData(0, data_size, labels_time_ptr, values_time_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
  std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
  dataframe_manager_labels.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_image_2d_ptr;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_image_2d_ptr;
  dataframe_manager_image_2d.getInsertData(0, data_size, labels_image_2d_ptr, values_image_2d_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_is_valid_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_is_valid_ptr;
  dataframe_manager_is_valid.getInsertData(0, data_size, labels_is_valid_ptr, values_is_valid_ptr);

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_insert_values(i, j), labels_time_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  n_dim_tensor_collection->tables_.at("DataFrame_time")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 1 * data_size * 6);
  std::shared_ptr<int[]> data_insert_data_time;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataPointer(data_insert_data_time);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> data_insert_values_time(data_insert_data_time.get(), data_size, 1, 6);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 6; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values_time(i, 0, j), values_time_ptr->getData()(i, 0, j));
    }
  }
  n_dim_tensor_collection->tables_.at("DataFrame_label")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray32<char>[]> data_insert_data_labels;
  n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataPointer(data_insert_data_labels);
  Eigen::TensorMap<Eigen::Tensor<TensorArray32<char>, 2>> data_insert_values_labels(data_insert_data_labels.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_labels(i, 0), values_labels_ptr->getData()(i, 0));
  }
  n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 1 * data_size * 28 * 28);
  std::shared_ptr<float[]> data_insert_data_image_2d;
  n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataPointer(data_insert_data_image_2d);
  Eigen::TensorMap<Eigen::Tensor<float, 4>> data_insert_values_image_2d(data_insert_data_image_2d.get(), data_size, 1, 28, 28);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 28; ++j) {
      for (int k = 0; k < 28; ++k) {
        BOOST_CHECK_EQUAL(data_insert_values_image_2d(i, 0, j, k), values_image_2d_ptr->getData()(i, 0, j, k));
      }
    }
  }
  n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<int[]> data_insert_data_is_valid;
  n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataPointer(data_insert_data_is_valid);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values_is_valid(data_insert_data_is_valid.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_is_valid(i, 0), values_is_valid_ptr->getData()(i, 0));
  }

  // Query for the number of valid entries
  SelectTableDataIsValidCpu select_is_valid;
  select_is_valid(n_dim_tensor_collection, device);
  BOOST_CHECK_EQUAL(select_is_valid.result_->getData()(0), data_size / 2);

  // Query for the number of labels = "one"
  SelectTableDataLabelCpu select_label_ones;
  select_label_ones(n_dim_tensor_collection, device);
  BOOST_CHECK_EQUAL(select_label_ones.result_, 130);

  // Query for the average pixel intensity in the first two weeks of January
  SelectTableDataImage2DCpu select_2D_image;
  select_2D_image(n_dim_tensor_collection, device);
  BOOST_CHECK_CLOSE(select_2D_image.result_->getData()(0), 0, 1e-3);

  // Make the expected tensor axes labels and tensor data after update
  dataframe_manager_time.initTime();
  dataframe_manager_time.setUseRandomValues(true);
  dataframe_manager_labels.setUseRandomValues(true);
  dataframe_manager_image_2d.setUseRandomValues(true);
  dataframe_manager_is_valid.setUseRandomValues(true);
  labels_time_ptr.reset();
  values_time_ptr.reset();
  dataframe_manager_time.getInsertData(0, data_size, labels_time_ptr, values_time_ptr);
  labels_labels_ptr.reset();
  values_labels_ptr.reset();
  dataframe_manager_labels.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  labels_image_2d_ptr.reset();
  values_image_2d_ptr.reset();
  dataframe_manager_image_2d.getInsertData(0, data_size, labels_image_2d_ptr, values_image_2d_ptr);
  labels_is_valid_ptr.reset();
  values_is_valid_ptr.reset();
  dataframe_manager_is_valid.getInsertData(0, data_size, labels_is_valid_ptr, values_is_valid_ptr);

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(transaction_manager, data_size, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_update_values(i, j), labels_time_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getIsModified().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getNotInMemory().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  n_dim_tensor_collection->tables_.at("DataFrame_time")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 1 * data_size * 6);
  std::shared_ptr<int[]> data_update_data_time;
  n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataPointer(data_update_data_time);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> data_update_values(data_update_data_time.get(), data_size, 1, 6);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 6; ++j) {
      BOOST_CHECK_EQUAL(data_update_values(i, 0, j), values_time_ptr->getData()(i, 0, j));
    }
  }
  n_dim_tensor_collection->tables_.at("DataFrame_label")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray32<char>[]> data_update_data_labels;
  n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataPointer(data_update_data_labels);
  Eigen::TensorMap<Eigen::Tensor<TensorArray32<char>, 2>> data_update_values_labels(data_update_data_labels.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_labels(i, 0), values_labels_ptr->getData()(i, 0));
  }
  n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 1 * data_size * 28 * 28);
  std::shared_ptr<float[]> data_update_data_image_2d;
  n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataPointer(data_update_data_image_2d);
  Eigen::TensorMap<Eigen::Tensor<float, 4>> data_update_values_image_2d(data_update_data_image_2d.get(), data_size, 1, 28, 28);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 28; ++j) {
      for (int k = 0; k < 28; ++k) {
        BOOST_CHECK_EQUAL(data_update_values_image_2d(i, 0, j, k), values_image_2d_ptr->getData()(i, 0, j, k));
      }
    }
  }
  n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<int[]> data_update_data_is_valid;
  n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataPointer(data_update_data_is_valid);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values_is_valid(data_update_data_is_valid.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_is_valid(i, 0), values_is_valid_ptr->getData()(i, 0));
  }

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(transaction_manager, data_size, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_time")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_label")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_image_2D")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("DataFrame_is_valid")->getDataTensorSize(), 0);
}

BOOST_AUTO_TEST_SUITE_END()