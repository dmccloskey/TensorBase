/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorTableFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/TensorTableFile.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorTableFile1)

BOOST_AUTO_TEST_CASE(constructor)
{
  TensorTableFile<float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorTableFile<float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
  ptr = new TensorTableFile<float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorTableFile<float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  ptr = new TensorTableFile<float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(makeTensorTableShardFilename)
{
  TensorTableFile<float, Eigen::DefaultDevice, 3> data;
  BOOST_CHECK_EQUAL(data.makeTensorTableShardFilename("dir/", "table1", 1), "dir/table1_1.tts");
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes();

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // Setup the device
  Eigen::DefaultDevice device;

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Store the data
  TensorTableFile<float, Eigen::DefaultDevice, 3> data;
  data.storeTensorTableBinary("", tensorTable, device);

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
  }

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }

  // Load the data
  data.loadTensorTableBinary("", tensorTable, device);

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
  }

  // Test for the original data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i,j,k), tensor_values(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()