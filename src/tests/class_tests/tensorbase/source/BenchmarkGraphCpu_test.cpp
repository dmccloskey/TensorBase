/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BenchmarkGraphCpu test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/benchmarks/BenchmarkGraphCpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;
using namespace std;

BOOST_AUTO_TEST_SUITE(benchmarkGraphCpu)

BOOST_AUTO_TEST_CASE(InsertUpdateDeleteCpu)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 0;
  const int scale = 8; const edge_factor = 16;
  const bool in_memory = true;
  const int data_size = std::pow(2, scale) * edge_factor;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  BenchmarkGraph1LinkCpu benchmark_1_link;

  // Setup the GraphTensorCollectionGenerator
  GraphTensorCollectionGeneratorCpu tensor_collection_generator;

  // Setup the device
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::ThreadPoolDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(scale, edge_factor, shard_span_perc, device);

  // Setup the transaction manager
  TransactionManager<Eigen::ThreadPoolDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("3_sparse_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("3_sparse_indices")->getNLabels(), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("3_sparse_indices"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("3_sparse_indices"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_x")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_y")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("1_indices"), data_size);

  // Make the expected tensor axes labels and tensor data after insert
  GraphManagerSparseIndicesCpu graph_manager_sparse_indices(data_size, false);
  GraphManagerWeightsCpu graph_manager_weights(data_size, false);
  GraphManagerNodePropertyCpu graph_manager_node_property(data_size, false);
  GraphManagerLinkPropertyCpu graph_manager_link_property(data_size, false);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_sparse_indices_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_sparse_indices_ptr;
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
  std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_node_property_ptr;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_node_property_ptr;
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_link_property_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_link_property_ptr;
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr);

  // Test the expected tensor collection after insert
  benchmark_1_link.insert1Link(transaction_manager, scale, edge_factor, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_insert_values(i, j), labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<int[]> data_insert_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_insert_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values_sparse_indices(data_insert_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values_sparse_indices(i, j), values_sparse_indices_ptr->getData()(i, j));
    }
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<float[]> data_insert_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_insert_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_insert_values_weights(data_insert_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_weights(i, 0), values_labels_ptr->getData()(i, 0));
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), data_size * 1);
  std::shared_ptr<TensorArray8<char>[]> data_insert_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_insert_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_insert_values_node_property(data_insert_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_node_property(i, 0), values_node_property_ptr->getData()(i, 0));
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray8<char>[]> data_insert_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_insert_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_insert_values_link_property(data_insert_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_link_property(i, 0), values_link_property_ptr->getData()(i, 0));
  }

  // Query for the number of black nodes

  // Query for the number of dashed links

  // Query for the adjacency matrix

  // Query for the BFS

  // Query for the SSSP

  // Make the expected tensor axes labels and tensor data after update
  graph_manager_sparse_indices.initTime();
  graph_manager_sparse_indices.setUseRandomValues(true);
  graph_manager_weights.setUseRandomValues(true);
  graph_manager_node_property.setUseRandomValues(true);
  graph_manager_link_property.setUseRandomValues(true);
  labels_sparse_indices_ptr.reset();
  values_sparse_indices_ptr.reset();
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr);
  labels_labels_ptr.reset();
  values_labels_ptr.reset();
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  labels_node_property_ptr.reset();
  values_node_property_ptr.reset();
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr);
  labels_link_property_ptr.reset();
  values_link_property_ptr.reset();
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr);

  // Test the expected tensor collection after update
  benchmark_1_link.update1Link(transaction_manager, scale, edge_factor, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_update_values(i, j), labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<int[]> data_update_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_update_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_EQUAL(data_update_values(i, j), values_sparse_indices_ptr->getData()(i, j));
    }
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<float[]> data_update_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_update_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_update_values_weights(data_update_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_weights(i, 0), values_labels_ptr->getData()(i, 0));
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), data_size * 1);
  std::shared_ptr<TensorArray8<char>[]> data_update_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_update_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_update_values_node_property(data_update_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_node_property(i, 0), values_node_property_ptr->getData()(i, 0));
  }
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray8<char>[]> data_update_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_update_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_update_values_link_property(data_update_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_link_property(i, 0), values_link_property_ptr->getData()(i, 0));
  }

  // Test the expected tensor collection after deletion
  benchmark_1_link.delete1Link(transaction_manager, scale, edge_factor, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 0);
}

// repeat with sharding
BOOST_AUTO_TEST_CASE(InsertUpdateDeleteShardingCpu)
{
  // Parameters for the test
  std::string data_dir = "";
  const int n_dims = 0;
  const int scale = 8; const edge_factor = 16;
  const bool in_memory = false;
  const int data_size = std::pow(2, scale) * edge_factor;  
  const double shard_span_perc = 0.05;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  BenchmarkGraph1LinkCpu benchmark_1_link;

  // Setup the GraphTensorCollectionGenerator
  GraphTensorCollectionGeneratorCpu tensor_collection_generator;

  // Setup the device
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::ThreadPoolDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(scale, edge_factor, shard_span_perc, device);

  // Setup the transaction manager
  TransactionManager<Eigen::ThreadPoolDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("3_sparse_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("3_sparse_indices")->getNLabels(), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("3_sparse_indices"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("3_sparse_indices"), 6);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_x")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_x")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_y")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("3_y")->getNLabels(), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("1_indices"), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("3_x"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("3_y"), 28);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("1_indices"), TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("2_columns"), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("1_indices"), data_size);

  // Make the expected tensor axes labels and tensor data after insert
  GraphManagerSparseIndicesCpu graph_manager_sparse_indices(data_size, false);
  GraphManagerWeightsCpu graph_manager_weights(data_size, false);
  GraphManagerNodePropertyCpu graph_manager_node_property(data_size, false);
  GraphManagerLinkPropertyCpu graph_manager_link_property(data_size, false);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_sparse_indices_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> values_sparse_indices_ptr;
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_labels_ptr;
  std::shared_ptr<TensorData<TensorArray32<char>, Eigen::ThreadPoolDevice, 2>> values_labels_ptr;
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_node_property_ptr;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 4>> values_node_property_ptr;
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> labels_link_property_ptr;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> values_link_property_ptr;
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr);

  // Test the expected tensor collection after insert
  benchmark_1_link.insert1Link(transaction_manager, scale, edge_factor, in_memory, device);

  // Test the expected tensor axes after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_insert_values(i, j), labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after insert
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<int[]> data_insert_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_insert_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values_sparse_indices(data_insert_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_EQUAL(data_insert_values_sparse_indices(i, j), values_sparse_indices_ptr->getData()(i, j));
    }
  }
  n_dim_tensor_collection->tables_.at("Graph_weights")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_weights")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<float[]> data_insert_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_insert_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_insert_values_weights(data_insert_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_weights(i, 0), values_labels_ptr->getData()(i, 0));
  }
  n_dim_tensor_collection->tables_.at("Graph_node_property")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), data_size * 1);
  std::shared_ptr<TensorArray8<char>[]> data_insert_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_insert_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_insert_values_node_property(data_insert_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_node_property(i, 0), values_node_property_ptr->getData()(i, 0));
  }
  n_dim_tensor_collection->tables_.at("Graph_link_property")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray8<char>[]> data_insert_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_insert_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_insert_values_link_property(data_insert_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_insert_values_link_property(i, 0), values_link_property_ptr->getData()(i, 0));
  }

  // Query for the number of black nodes

  // Query for the number of dashed links

  // Query for the adjacency matrix

  // Query for the BFS

  // Query for the SSSP

  // Make the expected tensor axes labels and tensor data after update
  graph_manager_sparse_indices.initTime();
  graph_manager_sparse_indices.setUseRandomValues(true);
  graph_manager_weights.setUseRandomValues(true);
  graph_manager_node_property.setUseRandomValues(true);
  graph_manager_link_property.setUseRandomValues(true);
  labels_sparse_indices_ptr.reset();
  values_sparse_indices_ptr.reset();
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr);
  labels_labels_ptr.reset();
  values_labels_ptr.reset();
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr);
  labels_node_property_ptr.reset();
  values_node_property_ptr.reset();
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr);
  labels_link_property_ptr.reset();
  values_link_property_ptr.reset();
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr);

  // Test the expected tensor collection after update
  benchmark_1_link.update1Link(transaction_manager, scale, edge_factor, in_memory, device);

  // Test the expected tensor axes after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), data_size);
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      BOOST_CHECK_EQUAL(labels_indices_update_values(i, j), labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_indices after update
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getTensorSize(), data_size);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getTensorSize(), data_size);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_indices")->getData()(i), TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<int[]> data_update_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_update_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_EQUAL(data_update_values(i, j), values_sparse_indices_ptr->getData()(i, j));
    }
  }
  n_dim_tensor_collection->tables_.at("Graph_weights")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_weights")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 2 * data_size);
  std::shared_ptr<float[]> data_update_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_update_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_update_values_weights(data_update_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_weights(i, 0), values_labels_ptr->getData()(i, 0));
  }
  n_dim_tensor_collection->tables_.at("Graph_node_property")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), data_size * 1);
  std::shared_ptr<TensorArray8<char>[]> data_update_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_update_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_update_values_node_property(data_update_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_node_property(i, 0), values_node_property_ptr->getData()(i, 0));
  }
  n_dim_tensor_collection->tables_.at("Graph_link_property")->loadTensorTableBinary(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDir(), device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 1 * data_size);
  std::shared_ptr<TensorArray8<char>[]> data_update_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_update_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 2>> data_update_values_link_property(data_update_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    BOOST_CHECK_EQUAL(data_update_values_link_property(i, 0), values_link_property_ptr->getData()(i, 0));
  }

  // Test the expected tensor collection after deletion
  benchmark_1_link.delete1Link(transaction_manager, scale, edge_factor, in_memory, device);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_indices")->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize(), 0);
}

BOOST_AUTO_TEST_SUITE_END()