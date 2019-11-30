/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE BenchmarkPixelsDefaultDevice test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/benchmarks/BenchmarkPixelsDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(benchmarkPixelsDefaultDevice)

BOOST_AUTO_TEST_CASE(InsertUpdateDelete0DDefaultDevice) 
{
  // Parameters for the test
  std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
  int n_dims = 1;
  int data_size = 1296;
  bool in_memory = true;
  double shard_span_perc = 1;
  int n_engines = 1;

  // Setup the Benchmarking suite
  Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp;

  // Setup the TensorCollectionGenerator
  TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator;

  // Make the nD TensorTables
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

  // Setup the transaction manager
  TransactionManager<DeviceT> transaction_manager;
  transaction_manager.setMaxOperations(data_size + 1);
  transaction_manager.setTensorCollection(n_dim_tensor_collection);

  // Test the initial tensor collection
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().second->getName(), "xyzt");
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().second->getNDimensions(), 2);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().second->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().second->getName(), "values");
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().second->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getAxes().second->getNLabels(), 0);
  BOOST_CHECK_EQUAL(n_dim_tensor_collection->tables_.at("TTable")->getDataTensorSize(), 0);

  // Test the expected tensor collection after insert
  benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, device);
  // TODO: test the expected axis labels
  // TODO: test the expected axis indices
  // TODO: test the expected data

  // Test the expected tensor collection after update
  benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, device);
  // TODO: test the expected axis labels
  // TODO: test the expected axis indices
  // TODO: test the expected data

  // Test the expected tensor collection after deletion
  benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, device);
  // TODO: test the expected axis labels
  // TODO: test the expected axis indices
  // TODO: test the expected data
}

BOOST_AUTO_TEST_SUITE_END()