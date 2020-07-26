/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/benchmarks/BenchmarkGraphGpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;
using namespace std;

void test_InsertUpdateDeleteGpu()
{
  // Parameters for the test
  std::string data_dir = "";
  const int scale = 8;
  const int edge_factor = 16;
  const bool in_memory = true;
  const int data_size = std::pow(2, scale) * edge_factor;
  const double shard_span_perc = 1;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  BenchmarkGraph1LinkGpu benchmark_1_link;

  // Setup the GraphTensorCollectionGenerator
  GraphTensorCollectionGeneratorGpu tensor_collection_generator;

  // Setup the device
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(scale, edge_factor, shard_span_perc, device);

  // Setup the transaction manager
  TransactionManager<Eigen::GpuDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("1_links") == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("1_links") == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("1_links") == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("1_links") == data_size);

  // Make the expected tensor axes labels and tensor data after insert
  benchmark_1_link.makeKroneckerGraph(scale, edge_factor, device);
  GraphManagerSparseIndicesGpu<int, float, int, int> graph_manager_sparse_indices(false);
  GraphManagerWeightsGpu<int, float, int, float> graph_manager_weights(false);
  GraphManagerNodePropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_node_property(false);
  GraphManagerLinkPropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_link_property(false);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_sparse_indices_ptr;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_sparse_indices_ptr;
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_labels_ptr;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> values_labels_ptr;
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_node_property_ptr;
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_node_property_ptr;
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_link_property_ptr;
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_link_property_ptr;
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);

  // Test the expected tensor collection after insert
  benchmark_1_link.insert1Link(transaction_manager, scale, edge_factor, in_memory, device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (auto& table_map : n_dim_tensor_collection->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }

  // Test the expected tensor axes after insert
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == data_size);
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      assert(labels_indices_insert_values(i, j) == labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_links after insert
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getTensorSize() == data_size);
  for (int i = 0; i < data_size; ++i) {
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getData()(i) == 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getData()(i) == 0);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 2 * data_size);
  std::shared_ptr<int[]> data_insert_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_insert_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values_sparse_indices(data_insert_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      assert(data_insert_values_sparse_indices(i, j) == values_sparse_indices_ptr->getData()(i, j));
    }
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<float[]> data_insert_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_insert_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_insert_values_weights(data_insert_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_insert_values_weights(i, 0) == values_labels_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() <= 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_insert_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_insert_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_insert_values_node_property(data_insert_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_insert_values_node_property(i, 0) == values_node_property_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_insert_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_insert_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_insert_values_link_property(data_insert_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_insert_values_link_property(i, 0) == values_link_property_ptr->getData()(i, 0));
  }

  // Query for the number of black nodes

  // Query for the number of dashed links

  // Query for the adjacency matrix

  // Query for the BFS

  // Query for the SSSP

  // Make the expected tensor axes labels and tensor data after update
  graph_manager_sparse_indices.setUseRandomValues(true);
  graph_manager_weights.setUseRandomValues(true);
  graph_manager_node_property.setUseRandomValues(true);
  graph_manager_link_property.setUseRandomValues(true);
  labels_sparse_indices_ptr.reset();
  values_sparse_indices_ptr.reset();
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  labels_labels_ptr.reset();
  values_labels_ptr.reset();
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  labels_node_property_ptr.reset();
  values_node_property_ptr.reset();
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  labels_link_property_ptr.reset();
  values_link_property_ptr.reset();
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);

  // Test the expected tensor collection after update
  benchmark_1_link.update1Link(transaction_manager, scale, edge_factor, in_memory, device);
  for (auto& table_map : n_dim_tensor_collection->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }

  // Test the expected tensor axes after update
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == data_size);
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      assert(labels_indices_update_values(i, j) == labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_links after update
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getTensorSize() == data_size);
  for (int i = 0; i < data_size; ++i) {
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getData()(i) == 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getData()(i) == 0);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 2 * data_size);
  std::shared_ptr<int[]> data_update_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_update_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      assert(data_update_values(i, j) == values_sparse_indices_ptr->getData()(i, j));
    }
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<float[]> data_update_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_update_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_update_values_weights(data_update_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_update_values_weights(i, 0) == values_labels_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() <= 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_update_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_update_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_update_values_node_property(data_update_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_update_values_node_property(i, 0) == values_node_property_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_update_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_update_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_update_values_link_property(data_update_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_update_values_link_property(i, 0) == values_link_property_ptr->getData()(i, 0));
  }

  // Test the expected tensor collection after deletion
  benchmark_1_link.delete1Link(transaction_manager, scale, edge_factor, in_memory, device);
  for (auto& table_map : n_dim_tensor_collection->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 0);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_InsertUpdateDeleteShardingGpu()
{
  // Parameters for the test
  std::string data_dir = "";
  const int scale = 8; 
  const int edge_factor = 16;
  const bool in_memory = true;
  const int data_size = std::pow(2, scale) * edge_factor;
  const double shard_span_perc = 0.05;
  const int n_engines = 1;
  const int dim_span = std::pow(data_size, 0.25);

  // Setup the Benchmarking suite
  BenchmarkGraph1LinkGpu benchmark_1_link;

  // Setup the GraphTensorCollectionGenerator
  GraphTensorCollectionGeneratorGpu tensor_collection_generator;

  // Setup the device
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(scale, edge_factor, shard_span_perc, device);

  // Setup the transaction manager
  TransactionManager<Eigen::GpuDevice> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getMaxDimSizeFromAxisName("1_links") == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getMaxDimSizeFromAxisName("1_links") == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getMaxDimSizeFromAxisName("1_links") == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("2_columns")->getNLabels() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getShardSpans().at("1_links") == TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("2_columns") == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getMaxDimSizeFromAxisName("1_links") == data_size);

  // Make the expected tensor axes labels and tensor data after insert
  benchmark_1_link.makeKroneckerGraph(scale, edge_factor, device);
  GraphManagerSparseIndicesGpu<int, float, int, int> graph_manager_sparse_indices(false);
  GraphManagerWeightsGpu<int, float, int, float> graph_manager_weights(false);
  GraphManagerNodePropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_node_property(false);
  GraphManagerLinkPropertyGpu<int, float, int, TensorArrayGpu8<char>> graph_manager_link_property(false);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_sparse_indices_ptr;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> values_sparse_indices_ptr;
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_labels_ptr;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> values_labels_ptr;
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_node_property_ptr;
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_node_property_ptr;
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_link_property_ptr;
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_link_property_ptr;
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);

  // Test the expected tensor collection after insert
  benchmark_1_link.insert1Link(transaction_manager, scale, edge_factor, in_memory, device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (auto& table_map : n_dim_tensor_collection->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }

  // Test the expected tensor axes after insert
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == data_size);
  std::shared_ptr<int[]> labels_indices_insert_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getLabelsDataPointer(labels_indices_insert_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_insert_values(labels_indices_insert_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      assert(labels_indices_insert_values(i, j) == labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_links after insert
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getTensorSize() == data_size);
  for (int i = 0; i < data_size; ++i) {
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getData()(i) == 0);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getData()(i) == 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after insert
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 2 * data_size);
  std::shared_ptr<int[]> data_insert_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_insert_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_insert_values_sparse_indices(data_insert_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      assert(data_insert_values_sparse_indices(i, j) == values_sparse_indices_ptr->getData()(i, j));
    }
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<float[]> data_insert_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_insert_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_insert_values_weights(data_insert_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_insert_values_weights(i, 0) == values_labels_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() <= 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_insert_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_insert_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_insert_values_node_property(data_insert_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_insert_values_node_property(i, 0) == values_node_property_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_insert_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_insert_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_insert_values_link_property(data_insert_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_insert_values_link_property(i, 0) == values_link_property_ptr->getData()(i, 0));
  }

  // Query for the number of black nodes

  // Query for the number of dashed links

  // Query for the adjacency matrix

  // Query for the BFS

  // Query for the SSSP

  // Make the expected tensor axes labels and tensor data after update
  graph_manager_sparse_indices.setUseRandomValues(true);
  graph_manager_weights.setUseRandomValues(true);
  graph_manager_node_property.setUseRandomValues(true);
  graph_manager_link_property.setUseRandomValues(true);
  labels_sparse_indices_ptr.reset();
  values_sparse_indices_ptr.reset();
  graph_manager_sparse_indices.getInsertData(0, data_size, labels_sparse_indices_ptr, values_sparse_indices_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  labels_labels_ptr.reset();
  values_labels_ptr.reset();
  graph_manager_weights.getInsertData(0, data_size, labels_labels_ptr, values_labels_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  labels_node_property_ptr.reset();
  values_node_property_ptr.reset();
  graph_manager_node_property.getInsertData(0, data_size, labels_node_property_ptr, values_node_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);
  labels_link_property_ptr.reset();
  values_link_property_ptr.reset();
  graph_manager_link_property.getInsertData(0, data_size, labels_link_property_ptr, values_link_property_ptr, benchmark_1_link.graph_manager_helper_.kronecker_graph_indices_, benchmark_1_link.graph_manager_helper_.kronecker_graph_weights_, benchmark_1_link.graph_manager_helper_.kronecker_graph_node_ids_, benchmark_1_link.graph_manager_helper_.kronecker_graph_link_ids_, device);

  // Test the expected tensor collection after update
  benchmark_1_link.update1Link(transaction_manager, scale, edge_factor, in_memory, device);
  for (auto& table_map : n_dim_tensor_collection->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }

  // Test the expected tensor axes after update
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == data_size);
  std::shared_ptr<int[]> labels_indices_update_data;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getLabelsDataPointer(labels_indices_update_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_indices_update_values(labels_indices_update_data.get(), 1, data_size);
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < data_size; ++j) {
      assert(labels_indices_update_values(i, j) == labels_sparse_indices_ptr->getData()(i, j));
    }
  }

  // Test the expected axis 1_links after update
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getTensorSize() == data_size);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getTensorSize() == data_size);
  for (int i = 0; i < data_size; ++i) {
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndices().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIndicesView().at("1_links")->getData()(i) == i + 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getIsModified().at("1_links")->getData()(i) == 0);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getNotInMemory().at("1_links")->getData()(i) == 1);
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardId().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_id(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
    assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getShardIndices().at("1_links")->getData()(i) == TensorCollectionShardHelper::calc_shard_index(TensorCollectionShardHelper::round_1(data_size, shard_span_perc), i));
  }

  // Test the expected data after update
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 2 * data_size);
  std::shared_ptr<int[]> data_update_data_sparse_indices;
  n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataPointer(data_update_data_sparse_indices);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> data_update_values(data_update_data_sparse_indices.get(), data_size, 2);
  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      assert(data_update_values(i, j) == values_sparse_indices_ptr->getData()(i, j));
    }
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<float[]> data_update_data_weights;
  n_dim_tensor_collection->tables_.at("Graph_weights")->getDataPointer(data_update_data_weights);
  Eigen::TensorMap<Eigen::Tensor<float, 2>> data_update_values_weights(data_update_data_weights.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_update_values_weights(i, 0) == values_labels_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() <= 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_update_data_node_property;
  n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataPointer(data_update_data_node_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_update_values_node_property(data_update_data_node_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_update_values_node_property(i, 0) == values_node_property_ptr->getData()(i, 0));
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 1 * data_size);
  std::shared_ptr<TensorArrayGpu8<char>[]> data_update_data_link_property;
  n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataPointer(data_update_data_link_property);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 2>> data_update_values_link_property(data_update_data_link_property.get(), data_size, 1);
  for (int i = 0; i < data_size; ++i) {
    assert(data_update_values_link_property(i, 0) == values_link_property_ptr->getData()(i, 0));
  }

  // Test the expected tensor collection after deletion
  benchmark_1_link.delete1Link(transaction_manager, scale, edge_factor, in_memory, device);
  for (auto& table_map : n_dim_tensor_collection->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_sparse_indices")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_weights")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getAxes().at("1_nodes")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_node_property")->getDataTensorSize() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNDimensions() == 1);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getAxes().at("1_links")->getNLabels() == 0);
  assert(n_dim_tensor_collection->tables_.at("Graph_link_property")->getDataTensorSize() == 0);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  test_InsertUpdateDeleteGpu();
  test_InsertUpdateDeleteShardingGpu();
  return 0;
}
#endif