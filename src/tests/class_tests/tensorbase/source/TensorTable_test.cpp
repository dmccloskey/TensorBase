/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorTable test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorTable)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
  TensorTableDefaultDevice<float, 3>* ptr = nullptr;
  TensorTableDefaultDevice<float, 3>* nullPointer = nullptr;
	ptr = new TensorTableDefaultDevice<float, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
  TensorTableDefaultDevice<float, 3>* ptr = nullptr;
	ptr = new TensorTableDefaultDevice<float, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructorNameAndAxesDefaultDevice)
{
  TensorTableDefaultDevice<float, 3> tensorTable("1");

  BOOST_CHECK_EQUAL(tensorTable.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "1");
  BOOST_CHECK_EQUAL(tensorTable.getDir(), "");

  TensorTableDefaultDevice<float, 3> tensorTable2("1", "dir");

  BOOST_CHECK_EQUAL(tensorTable2.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable2.getName(), "1");
  BOOST_CHECK_EQUAL(tensorTable2.getDir(), "dir");
}

BOOST_AUTO_TEST_CASE(comparatorDefaultDevice)
{
  Eigen::DefaultDevice device;
  // Set the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

  // Make the test tables
  TensorTableDefaultDevice<float, 3> tensorTable_test("1");
  tensorTable_test.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable_test.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable_test.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable_test.setAxes(device);

  // Test expected comparison
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);

  BOOST_CHECK(tensorTable_test == tensorTable1); // expected
  tensorTable1.setName("1.1");
  BOOST_CHECK(tensorTable_test != tensorTable1); // difference names but same axes

  // Test differen axes but same name
  TensorTableDefaultDevice<float, 3> tensorTable2("1");
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("4", dimensions1, labels1)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("5", dimensions2, labels2)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("6", dimensions3, labels3)));
  tensorTable2.setAxes(device);

  BOOST_CHECK(tensorTable_test != tensorTable2); // different axes
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;
  // Check defaults
  BOOST_CHECK_EQUAL(tensorTable.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDir(), "");
  BOOST_CHECK_EQUAL(tensorTable.getTensorSize(), 0);

  // Check getters/setters
  tensorTable.setId(1);
  tensorTable.setName("1");
  std::map<std::string, int> shard_span = {
    {"1", 2}, {"2", 2}, {"3", 3} };
  tensorTable.setShardSpans(shard_span);
  tensorTable.setDir("dir");

  BOOST_CHECK_EQUAL(tensorTable.getId(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "1");
  BOOST_CHECK(tensorTable.getShardSpans() == shard_span);
  BOOST_CHECK_EQUAL(tensorTable.getDir(), "dir");

  // SetAxes associated getters/setters
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  //Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  //labels1.setConstant("x-axis");
  //labels2.setConstant("y-axis");
  //labels3.setConstant("z-axis");
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setShardSpans(std::map<std::string, int>()); // reset the shard spans to zero
  tensorTable.setAxes(device);

  // Test expected axes values
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getName(), "1");
  //BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getLabels()(0, 0), 1);
  ////BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getLabels()(0,0), "x-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(nlabels1 -1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(nlabels1 - 1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(nlabels1 - 1), nlabels1);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getName(), "2");
  //BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getLabels()(0, 0), 2);
  ////BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getLabels()(0, 0), "y-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNLabels(), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardId().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(nlabels2 - 1), nlabels2);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getName(), "3");
  //BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getLabels()(0, 0), 3);
  ////BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getLabels()(0, 0), "z-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNLabels(), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardId().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(nlabels3 - 1), nlabels3);

  // Test expected axis to dims mapping
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("1"), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("2"), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor shard spans
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("1"), 2);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("2"), 3);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("3"), 5);

  // Test expected tensor dimensions
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 5);
  BOOST_CHECK_EQUAL(tensorTable.getTensorSize(), 30);

  // Test expected tensor data values
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(2), 5);
  size_t test = 2 * 3 * 5 * sizeof(float);
  BOOST_CHECK_EQUAL(tensorTable.getDataTensorBytes(), test);

  // Test expected CSV shard shape dimensions
  BOOST_CHECK_EQUAL(tensorTable.getCsvShardSpans().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getCsvShardSpans().at(1), 15);

  // Test expected CSV data dimensions
  BOOST_CHECK_EQUAL(tensorTable.getCsvDataDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getCsvDataDimensions().at(1), 15);

  // Test setting the data
  Eigen::Tensor<float, 3> tensor_data(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_data(i, j, k) = i + j + k;
      }
    }
  }
  tensorTable.setData(tensor_data);
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
  }
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
  }
  for (int i = 0; i < nlabels3; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
  }

  // Test setting the data
  tensorTable.setData();
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
  }
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }

  // Test clear
  tensorTable.clear();
  BOOST_CHECK_EQUAL(tensorTable.getAxes().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getShardId().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 0);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().size(), 0);
}

BOOST_AUTO_TEST_CASE(initDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // Check the dimensions and expected not_in_memory values
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }
  BOOST_CHECK_EQUAL(tensorTable.getDataTensorSize(), nlabels*nlabels*nlabels);

  // Reset the not_in_memory to false
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }

  // Resize the tensor data
  Eigen::array<Eigen::Index, 3> new_dimensions = { 2, 2, 2 };
  tensorTable.initData(new_dimensions, device);

  // Check the dimensions and expected not_in_memory values
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }
  BOOST_CHECK_EQUAL(tensorTable.getDataTensorSize(), 8);

  // Reset the not_in_memory to false
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }

  // Resize the tensor data to 0
  tensorTable.initData(device);

  // Check the dimensions and expected not_in_memory values
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }
  BOOST_CHECK_EQUAL(tensorTable.getDataTensorSize(), 0);
}

BOOST_AUTO_TEST_CASE(reShardIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test the default shard span
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("1"), 4);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("2"), 4);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("3"), 4);

  // Reset the shard span
  int shard_span = 3;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("1"), 3);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("2"), 3);
  BOOST_CHECK_EQUAL(tensorTable.getShardSpans().at("3"), 3);
  tensorTable.reShardIndices(device);
  for (int i = 0; i < nlabels; ++i) {
    if (i < shard_span) {
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("3")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i - shard_span + 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("2")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(i), i - shard_span + 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("3")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(i), i - shard_span + 1);
    }
  }
}

BOOST_AUTO_TEST_CASE(tensorDataWrappersDefaultDevice)
{
  // Set up the device
  Eigen::DefaultDevice device;

  // Set up the tensor table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test indices wrappers
  std::map<std::string, std::pair<bool, bool>> statuses;
  tensorTable.setIndicesDataStatus(false, false);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(!index_status.second.first);
    BOOST_CHECK(!index_status.second.second);
  }
  tensorTable.syncIndicesHAndDData(device);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setIndicesViewDataStatus(false, false);
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(!index_status.second.first);
    BOOST_CHECK(!index_status.second.second);
  }
  tensorTable.syncIndicesViewHAndDData(device);
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }

  // Test in memory wrappers
  tensorTable.setIsModifiedDataStatus(false, false);
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(!index_status.second.first);
    BOOST_CHECK(!index_status.second.second);
  }
  tensorTable.syncIsModifiedHAndDData(device);
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setNotInMemoryDataStatus(false, false);
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(!index_status.second.first);
    BOOST_CHECK(!index_status.second.second);
  }
  tensorTable.syncNotInMemoryHAndDData(device);
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setShardIdDataStatus(false, false);
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(!index_status.second.first);
    BOOST_CHECK(!index_status.second.second);
  }
  tensorTable.syncShardIdHAndDData(device);
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setShardIndicesDataStatus(false, false);
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(!index_status.second.first);
    BOOST_CHECK(!index_status.second.second);
  }
  tensorTable.syncShardIndicesHAndDData(device);
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }

  // Test axes wrappers
  tensorTable.setAxesDataStatus(false, false);
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    BOOST_CHECK(!axis_status.second.first);
    BOOST_CHECK(!axis_status.second.second);
  }
  tensorTable.syncAxesHAndDData(device);
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    BOOST_CHECK(axis_status.second.first);
    BOOST_CHECK(axis_status.second.second);
  }

  // Test bulk host/device transfers
  tensorTable.setIndicesDataStatus(false, false);
  tensorTable.setIndicesViewDataStatus(false, false);
  tensorTable.setIsModifiedDataStatus(false, false);
  tensorTable.setNotInMemoryDataStatus(false, false);
  tensorTable.setShardIdDataStatus(false, false);
  tensorTable.setShardIndicesDataStatus(false, false);
  tensorTable.setAxesDataStatus(false, false);

  // Test to Device
  tensorTable.syncAxesAndIndicesDData(device);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    BOOST_CHECK(axis_status.second.first);
    BOOST_CHECK(axis_status.second.second);
  }

  // Test to host
  tensorTable.setIndicesDataStatus(false, false);
  tensorTable.setIndicesViewDataStatus(false, false);
  tensorTable.setIsModifiedDataStatus(false, false);
  tensorTable.setNotInMemoryDataStatus(false, false);
  tensorTable.setShardIdDataStatus(false, false);
  tensorTable.setShardIndicesDataStatus(false, false);
  tensorTable.setAxesDataStatus(false, false);
  tensorTable.syncAxesAndIndicesHData(device);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    BOOST_CHECK(index_status.second.first);
    BOOST_CHECK(index_status.second.second);
  }
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    BOOST_CHECK(axis_status.second.first);
    BOOST_CHECK(axis_status.second.second);
  }
}

BOOST_AUTO_TEST_CASE(zeroIndicesViewAndResetIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // test null
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_1(tensorTable.getIndicesView().at("1")->getDataPointer().get(), tensorTable.getIndicesView().at("1")->getDimensions());
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i+1);
  }

  // test zero
  tensorTable.zeroIndicesView("1", device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }
  // test reset
  tensorTable.resetIndicesView("1", device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i+1);
  }
}

BOOST_AUTO_TEST_CASE(selectIndicesView1DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(nlabels / 2);
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0) {
      select_labels_values(iter) = i;
      ++iter;
    }
  }
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ nlabels/2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);

  // test the updated view
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i%2==0)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }
}

BOOST_AUTO_TEST_CASE(selectIndicesView2DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(2), dimensions2(1), dimensions3(1);
  dimensions1(0) = "a"; dimensions1(1) = "b";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(2, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3}, {4, 5, 6, 7} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(2, 2);
  select_labels_values.setValues({ {0, 2}, {4, 6} });
  TensorDataDefaultDevice<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 2, 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(select_labels);

  // test the updated view
  tensorTable.selectIndicesView("1", select_labels_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }
}

BOOST_AUTO_TEST_CASE(broadcastSelectIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the indices test
  Eigen::Tensor<int, 3> indices_test(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = i + 1;
      }
    }
  }

  // test the broadcast indices values
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_view_bcast;
  tensorTable.broadcastSelectIndicesView(indices_view_bcast, "1", device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_view_bcast->getData()(i,j,k), indices_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(extractTensorDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data, selection indices, and test selection data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> tensor_test(Eigen::array<Eigen::Index, 3>({ nlabels / 2, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        float value = i * nlabels + j * nlabels + k;
        tensor_values(i, j, k) = value;
        if (i % 2 == 0) {
          indices_values(i, j, k) = 1;
          tensor_test(i/2, j, k) = value;
        }
        else {
          indices_values(i, j, k) = 0;
        }
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataDefaultDevice<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_values);
  
  // test
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensor_select;
  tensorTable.reduceTensorDataToSelectIndices(std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_select), 
    tensor_select, "1", nlabels / 2, device);
  for (int i = 0; i < nlabels/2; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_CLOSE(tensor_select->getData()(i, j, k), tensor_test(i, j, k), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(selectTensorIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 2;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor select and values select data
  Eigen::Tensor<float, 3> tensor_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 1> values_select_values(Eigen::array<Eigen::Index, 1>({ nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    values_select_values(i) = 2.0;
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_select_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  TensorDataDefaultDevice<float, 3> tensor_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  tensor_select.setData(tensor_select_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensor_select_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(tensor_select);
  TensorDataDefaultDevice<float, 1> values_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  values_select.setData(values_select_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> values_select_ptr = std::make_shared<TensorDataDefaultDevice<float, 1>>(values_select);

  // test inequality
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_select;
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::NOT_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
      }
    }
  }

  // test equality
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test less than
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN, logicalModifiers::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) < 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test less than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN_OR_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) <= 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test greater than
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN, logicalModifiers::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) > 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test greater than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN_OR_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) >= 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(applyIndicesSelectToIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the indices select
  Eigen::Tensor<int, 3> indices_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == j && j == k && k == i
          && i < nlabels - 1 && j < nlabels - 1 && k < nlabels - 1) // the first 2 diagonal elements
          indices_select_values(i, j, k) = 1;
        else
          indices_select_values(i, j, k) = 0;
      }
    }
  }
  TensorDataDefaultDevice<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_select);

  // test using the second indices view
  tensorTable.getIndicesView().at("2")->getData()(nlabels - 1) = 0;
  // test for OR within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::OR, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i == nlabels - 1)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
  }

  tensorTable.resetIndicesView("2", device);
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  // test for AND within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::OR, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i == 0)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
  }

  tensorTable.resetIndicesView("2", device);
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  // test for OR within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i != 0 && i < nlabels - 1)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);
  }

  tensorTable.resetIndicesView("2", device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == j && j == k && k == i
          && i < nlabels - 1 && j < nlabels - 1 && k < nlabels - 1) // the first 2 diagonal elements
          indices_select_ptr->getData()(i, j, k) = 1;
        else if (j == 0)
          indices_select_ptr->getData()(i, j, k) = 1; // all elements along the first index of the selection dim
        else
          indices_select_ptr->getData()(i, j, k) = 0;
      }
    }
  }
  // test for AND within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i==0)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i+1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);
  }

  // TODO: lacking code coverage for the case of TDim = 2
}

BOOST_AUTO_TEST_CASE(whereIndicesViewDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(2);
  select_labels_values(0) = 0; select_labels_values(1) = 2;
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);

  // set up the selection values
  Eigen::Tensor<float, 1> select_values_values(2);
  select_values_values(0) = 9; select_values_values(1) = 9;
  TensorDataDefaultDevice<float, 1> select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_values.setData(select_values_values);

  // test whereIndicesView
  tensorTable.whereIndicesView("1", 0, select_labels_ptr,
    std::make_shared<TensorDataDefaultDevice<float, 1>>(select_values), logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);

    // indices view 2
    if (i == 1)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), 0);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // test whereIndicesView
  tensorTable.whereIndicesView("1", 0, select_labels_ptr,
    std::make_shared<TensorDataDefaultDevice<float, 1>>(select_values), logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);

    // indices view 2
    if (i == 1)
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    else
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), 0);
  }
}

BOOST_AUTO_TEST_CASE(sliceTensorForSortDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // test sliceTensorForSort for axis 2
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> tensor_sort;
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "2", device);
  std::vector<float> tensor_slice_2_test = {9, 12, 15};
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_CLOSE(tensor_sort->getData()(i), tensor_slice_2_test.at(i), 1e-3);
  }

  // test sliceTensorForSort for axis 2
  tensor_sort.reset();
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "3", device);
  std::vector<float> tensor_slice_3_test = { 9, 10, 11 };
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_CLOSE(tensor_sort->getData()(i), tensor_slice_3_test.at(i), 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(sortIndicesViewData1DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(1);
  select_labels_values(0) = 1;
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);

  // test sort ASC
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::ASC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
  }

  // test sort DESC
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), nlabels - i);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), nlabels - i);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // test sort ASC
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::ASC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
  }

  // test sort DESC
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), nlabels - i);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), nlabels - i);
  }
}

BOOST_AUTO_TEST_CASE(sortIndicesViewData2DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(1, 1);
  select_labels_values(0, 0) = 1;
  TensorDataDefaultDevice<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(select_labels);

  // test sort ASC
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::ASC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
  }

  // test sort DESC
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::DESC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), nlabels - i);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), nlabels - i);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // test sort ASC
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::ASC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
  }

  // test sort DESC
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::DESC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), nlabels - i);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), nlabels - i);
  }
}

BOOST_AUTO_TEST_CASE(makeSelectIndicesFromIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test null
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_select;
  tensorTable.makeSelectIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_select, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
      }
    }
  }

  // make the expected indices tensor
  Eigen::Tensor<int, 3> indices_select_test(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == 1)
          indices_select_test(i, j, k) = 1;
        else
          indices_select_test(i, j, k) = 0;
      }
    }
  }

  // select
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int,1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({1});
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test selected
  indices_select.reset();
  tensorTable.makeSelectIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_select, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), indices_select_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(getSelectTensorDataFromIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // select label 1 from axis 1
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // make the expected dimensions
  Eigen::array<Eigen::Index, 3> select_dimensions = { 1, 3, 3 };

  // make the indices_select
  Eigen::Tensor<float, 3> tensor_select_test(select_dimensions);
  Eigen::Tensor<int, 3> indices_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == 1) {
          indices_select_values(i, j, k) = 1;
          tensor_select_test(0, j, k) = float(iter);
        }
        else {
          indices_select_values(i, j, k) = 0;
        }
        ++iter;
      }
    }
  }
  TensorDataDefaultDevice<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);

  // test for the selected data
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensor_select_ptr;
  tensorTable.getSelectTensorDataFromIndicesView(tensor_select_ptr, std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_select), device);
  BOOST_CHECK(tensor_select_ptr->getDimensions() == select_dimensions);
  for (int j = 0; j < nlabels; ++j) {
    for (int k = 0; k < nlabels; ++k) {
      BOOST_CHECK_CLOSE(tensor_select_ptr->getData()(0, j, k), tensor_select_test(0, j, k), 1e-3);
    }
  }
}

BOOST_AUTO_TEST_CASE(selectTensorDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // select label 1 from axis 1
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test `selectTensorData`
  tensorTable.selectTensorData(device);

  // Test expected axes values
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getName(), "1");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(0), 1);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getName(), "2");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNLabels(), nlabels);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
  }

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getName(), "3");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNLabels(), nlabels);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test expected axis to dims mapping
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("1"), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("2"), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor dimensions
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 3);

  // Test expected tensor data values
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(2), 3);
  size_t test = 1 * 3 * 3 * sizeof(float);
  BOOST_CHECK_EQUAL(tensorTable.getDataTensorBytes(), test);  

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  tensorTable.selectTensorData(device);

  // Test expected axes values
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getName(), "1");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(0), 1);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getName(), "2");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNLabels(), nlabels);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
  }

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getName(), "3");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNLabels(), nlabels);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test expected axis to dims mapping
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("1"), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("2"), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor dimensions
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 3);

  // Test expected tensor data values
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDataDimensions().at(2), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDataTensorBytes(), test);
}

BOOST_AUTO_TEST_CASE(makeSortIndicesViewFromIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // make the expected tensor indices
  Eigen::Tensor<int, 3> indices_test(nlabels, nlabels, nlabels);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = i + j*nlabels + k*nlabels*nlabels + 1;
      }
    }
  }

  // Test for the sort indices
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_sort_ptr;
  tensorTable.makeSortIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_sort_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_sort_ptr->getData()(i, j, k), indices_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(sortTensorDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(1);
  select_labels_values(0) = 0;
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);

  // sort each of the axes
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);

  // make the expected sorted tensor
  float sorted_data[] = { 24, 25, 26, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2 };
  Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor_sorted_values(sorted_data, Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));

  // Test for sorted tensor data and reset indices view
  tensorTable.sortTensorData(device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), i);
    BOOST_CHECK_EQUAL(axis_2_ptr->getLabels()(0, i), nlabels - i - 1);
    BOOST_CHECK_EQUAL(axis_3_ptr->getLabels()(0, i), nlabels - i - 1);
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_CLOSE(tensorTable.getData()(i,j,k), tensor_sorted_values(i, j, k), 1e-3);
      }
    }
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Test for sorted tensor data and reset indices view
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);
  tensorTable.sortTensorData(device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), i);
    BOOST_CHECK_EQUAL(axis_2_ptr->getLabels()(0, i), nlabels - i - 1);
    BOOST_CHECK_EQUAL(axis_3_ptr->getLabels()(0, i), nlabels - i - 1);
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_CLOSE(tensorTable.getData()(i, j, k), tensor_sorted_values(i, j, k), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(updateSelectTensorDataValues1DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the tensor data and the update values
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        update_values(i, j, k) = 100;
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

  // Test update
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_old_ptr;
  tensorTable.updateSelectTensorDataValues(values_new_ptr, values_old_ptr, device);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(values_old_ptr->getData()(i, j, k), float(iter));
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Test update
  values_old_ptr.reset();
  tensorTable.updateSelectTensorDataValues(values_new_ptr, values_old_ptr, device);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(values_old_ptr->getData()(i, j, k), float(iter));
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }
}

BOOST_AUTO_TEST_CASE(updateSelectTensorDataValues2DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the tensor data and the update values
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        update_values(i, j, k) = 100;
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

  // Test update
  TensorDataDefaultDevice<float, 3> values_old(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_old.setData();
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_old_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_old);
  tensorTable.updateSelectTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr->getDataPointer(), device);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(values_old_ptr->getData()(i, j, k), float(iter));
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Test update
  values_old_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_old);
  tensorTable.updateSelectTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr->getDataPointer(), device);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(values_old_ptr->getData()(i, j, k), float(iter));
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }
}

BOOST_AUTO_TEST_CASE(updateTensorDataValuesDefaultDevice)
{
	// setup the table
	TensorTableDefaultDevice<float, 3> tensorTable;
	Eigen::DefaultDevice device;

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
	tensorTable.setAxes(device);

	// setup the tensor data and the update values
	Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	int iter = 0;
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				tensor_values(i, j, k) = float(iter);
				update_values(i, j, k) = 100;
				++iter;
			}
		}
	}
	tensorTable.setData(tensor_values);
	TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	values_new.setData(update_values);
	std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

	// Test update
	std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 2>> values_old_ptr;
	tensorTable.updateTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr, device);
	iter = 0;
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				BOOST_CHECK_EQUAL(values_old_ptr->getData()(i + j * nlabels + k * nlabels * nlabels), tensor_values(i, j, k));
				BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
				++iter;
			}
		}
	}

	// Test for the in_memory and is_modified attributes
	for (int i = 0; i < nlabels; ++i) {
		BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
		BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
		BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
		BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
		BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
		BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
	}

	// Write the original data to disk, clear the data, and repeat the tests
	tensorTable.clear();
	axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
	axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
	axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
	tensorTable.addTensorAxis(axis_1_ptr);
	tensorTable.addTensorAxis(axis_2_ptr);
	tensorTable.addTensorAxis(axis_3_ptr);
	tensorTable.setAxes(device);
	tensorTable.setData(tensor_values);
	tensorTable.storeTensorTableBinary("", device);
	tensorTable.setData();

	// Test update
	values_old_ptr.reset();
	tensorTable.updateTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr, device);
	iter = 0;
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				BOOST_CHECK_EQUAL(values_old_ptr->getData()(i + j * nlabels + k * nlabels * nlabels), tensor_values(i, j, k));
				BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
				++iter;
			}
		}
	}

	// Test for the in_memory and is_modified attributes
	for (int i = 0; i < nlabels; ++i) {
		BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
		BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
		BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
		BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
		BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
		BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
	}
}

BOOST_AUTO_TEST_CASE(makeAppendIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  auto axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", 1, 0));
  axis_3_ptr->setDimensions(dimensions3);
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // test the making the append indices
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_ptr;
  tensorTable.makeAppendIndices("1", nlabels, indices_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_ptr->getData()(i), nlabels + i + 1);
  }

  // test the making the append indices on a zero axis
  indices_ptr.reset();
  tensorTable.makeAppendIndices("3", nlabels, indices_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_ptr->getData()(i), i + 1);
  }
}

BOOST_AUTO_TEST_CASE(appendToIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the new indices
  Eigen::Tensor<int, 1> indices_new_values(nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    indices_new_values(i) = nlabels + i + 1;
  }
  TensorDataDefaultDevice<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ nlabels - 1 }));
  indices_new.setData(indices_new_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_new);

  // test appendToIndices
  tensorTable.appendToIndices("1", indices_new_ptr, device);

  // check the new indices [MODIFIED]
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")), nlabels + nlabels - 1);
  for (int i = 0; i < nlabels + nlabels - 1; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 1);
    if (i < nlabels) {
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), 0);
    }
  }

  // check the existing indices [NEW]
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
  }
}

BOOST_AUTO_TEST_CASE(appendToAxis1DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new tensor data
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      update_values(0, i, j) = i;
    }
  }
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

  // setup the new axis labels
  Eigen::Tensor<int, 2> labels_values(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_values(0, 0) = 3;
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new);

  // setup the new indices
  TensorDataDefaultDevice<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_new.setData();
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_new);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), labels1(i));
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, nlabels), 3);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      BOOST_CHECK_EQUAL(tensorTable.getData()(nlabels, i, j), update_values(0, i, j));
    }
  }
  BOOST_CHECK_EQUAL(indices_new_ptr->getData()(0), nlabels + 1);

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), labels1(i));
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, nlabels), 3);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      BOOST_CHECK_EQUAL(tensorTable.getData()(nlabels, i, j), update_values(0, i, j));
    }
  }
  BOOST_CHECK_EQUAL(indices_new_ptr->getData()(0), nlabels + 1);

  // Check that the binarized data was written correctly [NEW]
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }

  tensorTable.loadTensorTableBinary("", device);
  // Test the new TensorTable
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      BOOST_CHECK_EQUAL(tensorTable.getData()(nlabels, i, j), update_values(0, i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(appendToAxis2DefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  auto axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", 1, 0));
  axis_1_ptr->setDimensions(dimensions1);
  auto axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData();

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j*nlabels + k*nlabels*nlabels;
      }
    }
  }
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(tensor_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

  // setup the new axis labels
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, nlabels }));
  labels_new.setData(labels1);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new);

  // setup the new indices
  TensorDataDefaultDevice<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ nlabels }));
  indices_new.setData();
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_new);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), labels1(i));
    BOOST_CHECK_EQUAL(indices_new_ptr->getData()(i), i + 1);
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Check that the binarized data was written correctly
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }

  tensorTable.loadTensorTableBinary("", device);
  // Test the new TensorTable
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), labels1(i));
    BOOST_CHECK_EQUAL(indices_new_ptr->getData()(i), i + 1);
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeIndicesViewSelectFromIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  indices_to_select_values.setValues({1, 2});
  TensorDataDefaultDevice<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ 2 }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_to_select);

  // test makeIndicesViewSelectFromIndices
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_select_ptr;
  tensorTable.makeIndicesViewSelectFromIndices("1", indices_select_ptr, indices_to_select_ptr, true, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i > 1)
      BOOST_CHECK_EQUAL(indices_select_ptr->getData()(i), 1);
    else
      BOOST_CHECK_EQUAL(indices_select_ptr->getData()(i), 0);
  }
  indices_select_ptr.reset();
  tensorTable.makeIndicesViewSelectFromIndices("1", indices_select_ptr, indices_to_select_ptr, false, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i <= 1)
      BOOST_CHECK_EQUAL(indices_select_ptr->getData()(i), 1);
    else
      BOOST_CHECK_EQUAL(indices_select_ptr->getData()(i), 0);
  }
}

BOOST_AUTO_TEST_CASE(deleteFromIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select_values.setValues({ 2 });
  TensorDataDefaultDevice<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_to_select);

  // test deleteFromIndices
  tensorTable.deleteFromIndices("1", indices_to_select_ptr, device);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")), nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    if (i == 0) {
      BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 2);
    }
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 1);
  }
}

BOOST_AUTO_TEST_CASE(makeSelectIndicesFromIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0) indices_to_select_values(i) = i + 1;
    else indices_to_select_values(i) = 0;
  }
  TensorDataDefaultDevice<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_to_select);

  // test the selection indices
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_select_ptr;
  tensorTable.makeSelectIndicesFromIndices("1", indices_to_select_ptr, indices_select_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i % 2 == 0)
          BOOST_CHECK_EQUAL(indices_select_ptr->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select_ptr->getData()(i, j, k), 0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(deleteFromAxisDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> new_values(Eigen::array<Eigen::Index, 3>({ nlabels - 1, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
        if (i != 1) {
          new_values(iter, j, k) = i + j * nlabels + k * nlabels*nlabels;
        }
      }
    }
    if (i != 1) ++iter;
  }
  tensorTable.setData(tensor_values);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select_values.setValues({ 2 });
  TensorDataDefaultDevice<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_to_select);

  // test deleteFromAxis
  TensorDataDefaultDevice<float, 3> values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values.setData();
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_ptr;
  tensorTable.deleteFromAxis("1", indices_to_select_ptr, labels_ptr, values_ptr->getDataPointer(), device);

  // test the expected indices sizes and values
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")), nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    if (i == 0) {
      BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 2);
    }
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 1);
  }

  // Test the expected data values
  for (int i = 0; i < nlabels - 1; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), new_values(i, j, k));
      }
    }
  }

  // Test the expected axis values
  std::vector<int> expected_labels = {0, 2};
  for (int i = 0; i < nlabels - 1; ++i) {
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), expected_labels.at(i));
  }

  // Test the expected returned labels
  BOOST_CHECK_EQUAL(labels_ptr->getData()(0,0), 1);

  // Test the expected returned data
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(values_ptr->getData()(i, j, k), tensor_values(1, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeIndicesFromIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // modify the indices view for axis 1
  tensorTable.getIndicesView().at("1")->getData()(0) = 0;

  // test makeIndicesFromIndicesView
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_ptr;
  tensorTable.makeIndicesFromIndicesView("1", indices_ptr, device);
  for (int i = 0; i < nlabels - 1; ++i) {
    BOOST_CHECK_EQUAL(indices_ptr->getData()(i), i + 2);
  }
}

BOOST_AUTO_TEST_CASE(insertIntoAxisDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

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

  // setup the new tensor data
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      update_values(0, i, j) = 100;
    }
  }
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

  // setup the new axis labels
  Eigen::Tensor<int, 2> labels_values(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_values(0, 0) = 100;
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new);

  // setup the new indices
  Eigen::Tensor<int, 1> indices_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_values(0) = 3;
  TensorDataDefaultDevice<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_new.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_new);

  // Change the indices and indices view to simulate a deletion
  tensorTable.getIndices().at("1")->getData()(nlabels - 1) = 4;
  tensorTable.getIndicesView().at("1")->getData()(nlabels - 1) = 4;

  // test insertIntoAxis
  tensorTable.insertIntoAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  int iter = 0;
  for (int i = 0; i < nlabels + 1; ++i) {
    // check the axis
    if (i == 2)
      BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), 100);
    else
      BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), labels1(iter));

    // check the indices
    BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    if (i >= nlabels) {
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i - nlabels + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getShardId().at("1")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }

    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        // check the tensor data
        if (i == 2)
          BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
        else
          BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(iter, j, k));
      }
    }
    if (i != 2) ++iter;
  }
}

BOOST_AUTO_TEST_CASE(makeSparseAxisLabelsFromIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<int, 2> expected_values(Eigen::array<Eigen::Index, 2>({ 3, nlabels*nlabels*nlabels }));
  expected_values.setValues({ 
    {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3 },
    {1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3 },
    {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3 } });

  // Test
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_ptr;
  tensorTable.makeSparseAxisLabelsFromIndicesView(labels_ptr, device);
  BOOST_CHECK_EQUAL(labels_ptr->getDimensions().at(0), 3);
  BOOST_CHECK_EQUAL(labels_ptr->getDimensions().at(1), nlabels*nlabels*nlabels);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < nlabels*nlabels*nlabels; ++j) {
      BOOST_CHECK_EQUAL(labels_ptr->getData()(i,j), expected_values(i,j));
    }
  }
}

BOOST_AUTO_TEST_CASE(makeSparseTensorTableDefaultDevice)
{
  // setup the device
  Eigen::DefaultDevice device;

  // setup the expected axes
  Eigen::Tensor<std::string, 1> dimensions1(3), dimensions2(1);
  dimensions1.setValues({"0", "1", "2"});
  dimensions2(0) = "Values";

  // setup the expected labels
  int nlabels1 = 27;
  Eigen::Tensor<int, 2> labels1(3, nlabels1), labels2(1, 1);
  labels1.setValues({
    {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3 },
    {1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3 },
    {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3 } });
  TensorDataDefaultDevice<int, 2> sparse_labels(Eigen::array<Eigen::Index, 2>({ 3, nlabels1 }));
  sparse_labels.setData(labels1);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> sparse_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(sparse_labels);

  labels2.setConstant(0);

  // setup the expected data
  int nlabels = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
      }
    }
  }
  TensorDataDefaultDevice<float, 3> sparse_data(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  sparse_data.setData(tensor_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> sparse_data_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(sparse_data);

  // Test
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 2>> sparse_table_ptr;
  TensorTableDefaultDevice<float, 3> tensorTable;
  tensorTable.makeSparseTensorTable(dimensions1, sparse_labels_ptr, sparse_data_ptr, sparse_table_ptr, device);

  // Check for the correct dimensions
  BOOST_CHECK_EQUAL(sparse_table_ptr->getDimensions().at(0), nlabels1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getDimensions().at(1), 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getName(), "Indices");
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getNDimensions(), 3);
  std::shared_ptr<int[]> labels1_ptr;
  sparse_table_ptr->getAxes().at("Indices")->getLabelsDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i), std::to_string(i));
    for (int j = 0; j < nlabels1; ++j) {
      BOOST_CHECK_EQUAL(labels_values(i,j), labels1(i,j));
    }
  }

  // Check the Values axes
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getName(), "Values");
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getNDimensions(), 1);
  std::shared_ptr<int[]> labels2_ptr;
  sparse_table_ptr->getAxes().at("Values")->getLabelsDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0), "Values");
  BOOST_CHECK_EQUAL(labels2_values(0,0), 0);

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndices().at("Indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardId().at("Indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIsModified().at("Indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i), i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndices().at("Values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndicesView().at("Values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardId().at("Values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIsModified().at("Values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i), 0);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardIndices().at("Values")->getData()(i), i + 1);
  }
}

BOOST_AUTO_TEST_CASE(getSelectTensorDataAsSparseTensorTableDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

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

  // setup the expected labels
  int nlabels1 = 27;
  Eigen::Tensor<int, 2> labels1_expected(3, nlabels1);
  labels1_expected.setValues({
    {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3 },
    {1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3 },
    {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3 } });

  // Test
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 2>> sparse_table_ptr;
  tensorTable.getSelectTensorDataAsSparseTensorTable(sparse_table_ptr, device);

  // Check for the correct dimensions
  BOOST_CHECK_EQUAL(sparse_table_ptr->getDimensions().at(0), nlabels1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getDimensions().at(1), 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getName(), "Indices");
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getNDimensions(), 3);
  std::shared_ptr<int[]> labels1_ptr;
  sparse_table_ptr->getAxes().at("Indices")->getLabelsDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i), std::to_string(i + 1));
    for (int j = 0; j < nlabels1; ++j) {
      BOOST_CHECK_EQUAL(labels_values(i, j), labels1_expected(i, j));
    }
  }

  // Check the Values axes
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getName(), "Values");
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getNDimensions(), 1);
  std::shared_ptr<int[]> labels2_ptr;
  sparse_table_ptr->getAxes().at("Values")->getLabelsDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0), "Values");
  BOOST_CHECK_EQUAL(labels2_values(0, 0), 0);

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndices().at("Indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardId().at("Indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIsModified().at("Indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i), i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndices().at("Values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndicesView().at("Values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardId().at("Values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIsModified().at("Values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i), 0);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardIndices().at("Values")->getData()(i), i + 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Test
  sparse_table_ptr.reset();
  tensorTable.getSelectTensorDataAsSparseTensorTable(sparse_table_ptr, device);

  // Check for the correct dimensions
  BOOST_CHECK_EQUAL(sparse_table_ptr->getDimensions().at(0), nlabels1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getDimensions().at(1), 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getName(), "Indices");
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getNDimensions(), 3);
  labels1_ptr.reset();
  sparse_table_ptr->getAxes().at("Indices")->getLabelsDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels1b_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i), std::to_string(i + 1));
    for (int j = 0; j < nlabels1; ++j) {
      BOOST_CHECK_EQUAL(labels1b_values(i, j), labels1_expected(i, j));
    }
  }

  // Check the Values axes
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getName(), "Values");
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getNLabels(), 1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getNDimensions(), 1);
  labels2_ptr.reset();
  sparse_table_ptr->getAxes().at("Values")->getLabelsDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2b_values(labels2_ptr.get(), 1, 1);
  BOOST_CHECK_EQUAL(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0), "Values");
  BOOST_CHECK_EQUAL(labels2b_values(0, 0), 0);

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndices().at("Indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardId().at("Indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIsModified().at("Indices")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i), 0);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i), i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndices().at("Values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIndicesView().at("Values")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardId().at("Values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getIsModified().at("Values")->getData()(i), 1);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i), 0);
    BOOST_CHECK_EQUAL(sparse_table_ptr->getShardIndices().at("Values")->getData()(i), i + 1);
  }
}

BOOST_AUTO_TEST_CASE(updateTensorDataConstantDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

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

  // setup the update values
  TensorDataDefaultDevice<float, 1> values_new(Eigen::array<Eigen::Index, 1>({1}));
  values_new.setData();
  values_new.getData()(0) = 100;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 1>>(values_new);

  // Test update
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 2>> values_old_ptr;
  tensorTable.updateTensorDataConstant(values_new_ptr, values_old_ptr, device);

  // Test the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(values_old_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i,j,k));
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // reset is_modified attribute
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    is_modified_map.second->getData() = is_modified_map.second->getData().constant(0);
  }

  // Revert the operation and test
  tensorTable.updateTensorDataFromSparseTensorTable(values_old_ptr, device);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // TODO: Test after a selection (see test for TensorOperation TensorUpdateConstant)

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.setData(tensor_values);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();

  // Test update
  values_old_ptr.reset();
  tensorTable.updateTensorDataConstant(values_new_ptr, values_old_ptr, device);

  // Test the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(values_old_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 100);
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // clear the data
  tensorTable.setData();

  // Revert the operation and test
  tensorTable.updateTensorDataFromSparseTensorTable(values_old_ptr, device);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }
}

BOOST_AUTO_TEST_CASE(makeShardIndicesFromShardIDsDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 6;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3, 4, 5} });
  labels2.setValues({ {0, 1, 2, 3, 4, 5} });
  labels3.setValues({ {0, 1, 2, 3, 4, 5} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);

  // Test for the shard indices
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_shard_ptr;
  tensorTable.makeShardIndicesFromShardIDs(indices_shard_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_shard_ptr->getData()(i, j, k), 1);
      }
    }
  }

  // make the expected tensor indices
  int shard_n_indices = 3;
  std::vector<int> shard_id_indices = { 0, 0, 1, 1, 2, 2 };
  Eigen::Tensor<int, 3> indices_test(nlabels, nlabels, nlabels);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = shard_id_indices.at(i) + shard_id_indices.at(j) * shard_n_indices + shard_id_indices.at(k) * shard_n_indices*shard_n_indices + 1;
      }
    }
  }

  // Test for the shard indices
  tensorTable.reShardIndices(device);
  indices_shard_ptr.reset();
  tensorTable.makeShardIndicesFromShardIDs(indices_shard_ptr, device); 
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_shard_ptr->getData()(i, j, k), indices_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(makeModifiedShardIDTensorDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Test the unmodified case
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> shard_id_indices_ptr;
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 0);

  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices;
  Eigen::array<Eigen::Index, 3> shard_data_dimensions;
  int shard_data_size = 0;
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  BOOST_CHECK_EQUAL(slice_indices.size(), 0);
  BOOST_CHECK_EQUAL(shard_data_size, 0);
  // Test the fully modified case
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    is_modified_map.second->getData() = is_modified_map.second->getData().constant(1);
  }
  shard_id_indices_ptr.reset();
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 8);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    BOOST_CHECK_EQUAL(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices_test;
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  slice_indices_test.emplace(2, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,0 }), Eigen::array<Eigen::Index, 3>({ 1,2,2 })));
  slice_indices_test.emplace(3, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,0 }), Eigen::array<Eigen::Index, 3>({ 2,1,2 })));
  slice_indices_test.emplace(4, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,0 }), Eigen::array<Eigen::Index, 3>({ 1,1,2 })));
  slice_indices_test.emplace(5, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,2 }), Eigen::array<Eigen::Index, 3>({ 2,2,1 })));
  slice_indices_test.emplace(6, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,2 }), Eigen::array<Eigen::Index, 3>({ 1,2,1 })));
  slice_indices_test.emplace(7, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,2 }), Eigen::array<Eigen::Index, 3>({ 2,1,1 })));
  slice_indices_test.emplace(8, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,2 }), Eigen::array<Eigen::Index, 3>({ 1,1,1 })));
  Eigen::array<Eigen::Index, 3> shard_data_dimensions_test = { nlabels, nlabels, nlabels };
  int iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    BOOST_CHECK_EQUAL(slice_indices_map.first, iter);
    BOOST_CHECK(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    BOOST_CHECK(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  BOOST_CHECK_EQUAL(shard_data_size, nlabels * nlabels * nlabels);

  // Test the partially modified case
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        is_modified_map.second->getData()(i) = 1;
      else
        is_modified_map.second->getData()(i) = 0;
    }
  }
  shard_id_indices_ptr.reset();
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    BOOST_CHECK_EQUAL(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  shard_data_dimensions_test = Eigen::array<Eigen::Index, 3>({ 2, 2, 2 });
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    BOOST_CHECK_EQUAL(slice_indices_map.first, iter);
    BOOST_CHECK(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    BOOST_CHECK(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  BOOST_CHECK_EQUAL(shard_data_size, 8);
}

BOOST_AUTO_TEST_CASE(makeNotInMemoryShardIDTensorDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Test all in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> shard_id_indices_ptr;
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 0);

  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices;
  Eigen::array<Eigen::Index, 3> shard_data_dimensions;
  int shard_data_size = 0;
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  BOOST_CHECK_EQUAL(slice_indices.size(), 0);
  BOOST_CHECK_EQUAL(shard_data_size, 0);

  // Test not all in memory case and none selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  for (auto& indices_view_map : tensorTable.getIndicesView()) {
    indices_view_map.second->getData() = indices_view_map.second->getData().constant(0);
  }
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 0);

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  BOOST_CHECK_EQUAL(slice_indices.size(), 0);
  BOOST_CHECK_EQUAL(shard_data_size, 0);

  // Test all not in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.resetIndicesView("1", device);
  tensorTable.resetIndicesView("2", device);
  tensorTable.resetIndicesView("3", device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 8);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    BOOST_CHECK_EQUAL(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices_test;
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  slice_indices_test.emplace(2, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,0 }), Eigen::array<Eigen::Index, 3>({ 1,2,2 })));
  slice_indices_test.emplace(3, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,0 }), Eigen::array<Eigen::Index, 3>({ 2,1,2 })));
  slice_indices_test.emplace(4, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,0 }), Eigen::array<Eigen::Index, 3>({ 1,1,2 })));
  slice_indices_test.emplace(5, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,2 }), Eigen::array<Eigen::Index, 3>({ 2,2,1 })));
  slice_indices_test.emplace(6, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,2 }), Eigen::array<Eigen::Index, 3>({ 1,2,1 })));
  slice_indices_test.emplace(7, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,2 }), Eigen::array<Eigen::Index, 3>({ 2,1,1 })));
  slice_indices_test.emplace(8, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,2 }), Eigen::array<Eigen::Index, 3>({ 1,1,1 })));
  Eigen::array<Eigen::Index, 3> shard_data_dimensions_test = {nlabels, nlabels, nlabels};
  int iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    BOOST_CHECK_EQUAL(slice_indices_map.first, iter);
    BOOST_CHECK(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    BOOST_CHECK(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  BOOST_CHECK_EQUAL(shard_data_size, nlabels * nlabels * nlabels);

  // Test the partially in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        in_memory_map.second->getData()(i) = 1;
      else
        in_memory_map.second->getData()(i) = 0;
    }
  }
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    BOOST_CHECK_EQUAL(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  shard_data_dimensions_test = Eigen::array<Eigen::Index, 3>({ 2, 2, 2 });
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    BOOST_CHECK_EQUAL(slice_indices_map.first, iter);
    BOOST_CHECK(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    BOOST_CHECK(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  BOOST_CHECK_EQUAL(shard_data_size, 8);

  // Test the partially in memory case and partially selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        in_memory_map.second->getData()(i) = 1;
      else
        in_memory_map.second->getData()(i) = 0;
    }
  }
  for (auto& indices_view_map : tensorTable.getIndicesView()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < 1)
        indices_view_map.second->getData()(i) = i + 1;
      else
        indices_view_map.second->getData()(i) = 0;
    }
  }
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  BOOST_CHECK_EQUAL(shard_id_indices_ptr->getTensorSize(), 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    BOOST_CHECK_EQUAL(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  shard_data_dimensions_test = Eigen::array<Eigen::Index, 3>({ 2, 2, 2 });
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    BOOST_CHECK_EQUAL(slice_indices_map.first, iter);
    BOOST_CHECK(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    BOOST_CHECK(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  BOOST_CHECK_EQUAL(shard_data_size, 8);
}

BOOST_AUTO_TEST_CASE(makeTensorTableShardFilenameDefaultDevice)
{
  TensorTableDefaultDevice<float, 3> tensorTable;
  BOOST_CHECK_EQUAL(tensorTable.makeTensorTableShardFilename("dir/", "table1", 1), "dir/table1_1.tts");
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinaryDefaultDevice)
{
  // Set up the device
  Eigen::DefaultDevice device;

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
  tensorTable.setAxes(device);

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

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Test store/load for the case of all `is_modified`, all not `not_in_memory`, and selected `indices_view`
  tensorTable.storeTensorTableBinary("", device);

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
  }

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }

  // Load the data
  tensorTable.loadTensorTableBinary("", device);

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
  }

  // Test for the original data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Test store/load for the case of partially `is_modified`, all not `not_in_memory`, and partially selected `indices_view`
  for (auto& is_modified_map : tensorTable.getIsModified()) { // Shards 1-7
    for (int i = 0; i < nlabels; ++i) {
      if (i < 2)
        is_modified_map.second->getData()(i) = 1;
      else
        is_modified_map.second->getData()(i) = 0;
    }
  }
  for (auto& indices_view_map : tensorTable.getIndicesView()) { // Shard 1
    for (int i = 0; i < nlabels; ++i) {
      if (i < 1)
        indices_view_map.second->getData()(i) = i + 1;
      else
        indices_view_map.second->getData()(i) = 0;
    }
  }

  // Test for the in_memory, is_modified, and indices_view attributes before store
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    if (i < 2) {
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
    }
    if (i < 1) {
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), 0);
    }
  }

  // Test for the in_memory, is_modified, and indices_view attributes after store
  tensorTable.storeTensorTableBinary("", device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 0);
    if (i < 1) {
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(i), 0);
    }
  }

  // Reset the in_memory values and Zero the TensorData
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.getData() = tensorTable.getData().constant(0);

  // Load the data
  tensorTable.loadTensorTableBinary("", device);

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    if (i < shard_span) {
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
    }
  }

  // Test for the original data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        if (i < shard_span && j < shard_span && k < shard_span)
          BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
        else
          BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), 0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(storeAndLoadTensorTableAxesDefaultDevice)
{
  // Set up the device
  Eigen::DefaultDevice device;

  // Set the axes data
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

  // Make the axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);

  // Store the axes
  tensorTable1.storeTensorTableAxesBinary("", device);

  // Remake empty axes
  tensorTable1.clear();
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", 1, nlabels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", 1, nlabels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", 1, nlabels3)));
  tensorTable1.setAxes(device);

  // Load the axes
  tensorTable1.loadTensorTableAxesBinary("", device);

  // Test for the correct axes data
  std::shared_ptr<int[]> labels1_ptr;
  tensorTable1.getAxes().at("1")->getLabelsDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels1_values(labels1_ptr.get(), 1, nlabels1);
  for (int j = 0; j < nlabels1; ++j) {
    BOOST_CHECK_EQUAL(labels1_values(0, j), labels1(0, j));
  }

  std::shared_ptr<int[]> labels2_ptr;
  tensorTable1.getAxes().at("2")->getLabelsDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, nlabels2);
  for (int j = 0; j < nlabels2; ++j) {
    BOOST_CHECK_EQUAL(labels2_values(0, j), labels2(0, j));
  }

  std::shared_ptr<int[]> labels3_ptr;
  tensorTable1.getAxes().at("3")->getLabelsDataPointer(labels3_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels3_values(labels3_ptr.get(), 1, nlabels3);
  for (int j = 0; j < nlabels3; ++j) {
    BOOST_CHECK_EQUAL(labels3_values(0, j), labels3(0, j));
  }
}

BOOST_AUTO_TEST_CASE(getCsvDataRowDefaultDevice)
{
  // Set up the device
  Eigen::DefaultDevice device;

  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  Eigen::Tensor<char, 2> labels3(1, nlabels);
  labels3.setValues({ {'0', '1', '2'} });
  auto axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<char>>(TensorAxisDefaultDevice<char>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  std::vector<std::string> row_0_test, row_1_test, row_4_test;
  int row_iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        float value = i + j * nlabels + k * nlabels*nlabels;
        tensor_values(i, j, k) = value;
        if (row_iter == 0) {
          row_0_test.push_back(std::to_string(value));
        }
        else if (row_iter == 1) {
          row_1_test.push_back(std::to_string(value));
        }
        else if (row_iter == 4) {
          row_4_test.push_back(std::to_string(value));
        }
      }
      ++row_iter;
    }
  }
  tensorTable.setData(tensor_values);

  // Test getCsvDataRow
  // TODO: also test char and tensorArray types
  std::vector<std::string> row_0 = tensorTable.getCsvDataRow(0);
  std::vector<std::string> row_1 = tensorTable.getCsvDataRow(1);
  std::vector<std::string> row_4 = tensorTable.getCsvDataRow(4);
  BOOST_CHECK_EQUAL(row_0.size(), nlabels);
  BOOST_CHECK_EQUAL(row_1.size(), nlabels);
  BOOST_CHECK_EQUAL(row_4.size(), nlabels);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(row_0.at(i), row_0_test.at(i));
    BOOST_CHECK_EQUAL(row_1.at(i), row_1_test.at(i));
    BOOST_CHECK_EQUAL(row_4.at(i), row_4_test.at(i));
  }

  // Make the expected labels row values
  std::map<std::string, std::vector<std::string>> labels_row_0_test = { {"2", {"0"}}, {"3", {"a"}} };
  std::map<std::string, std::vector<std::string>> labels_row_1_test = { {"2", {"1"}}, {"3", {"a"}} };
  std::map<std::string, std::vector<std::string>> labels_row_4_test = { {"2", {"1"}}, {"3", {"b"}} };

  // Test getCsvAxesLabelsRow
  std::map<std::string, std::vector<std::string>> labels_row_0 = tensorTable.getCsvAxesLabelsRow(0);
  std::map<std::string, std::vector<std::string>> labels_row_1 = tensorTable.getCsvAxesLabelsRow(1);
  std::map<std::string, std::vector<std::string>> labels_row_4 = tensorTable.getCsvAxesLabelsRow(4);
  BOOST_CHECK_EQUAL(labels_row_0.size(), 2);
  BOOST_CHECK_EQUAL(labels_row_1.size(), 2);
  BOOST_CHECK_EQUAL(labels_row_4.size(), 2);
  for (int i = 2; i < 4; ++i) {
    std::string axis_name = std::to_string(i);
    for (int j = 0; j < 1; ++j) {
      BOOST_CHECK_EQUAL(labels_row_0.at(axis_name).at(j), labels_row_0_test.at(axis_name).at(j));
      BOOST_CHECK_EQUAL(labels_row_1.at(axis_name).at(j), labels_row_1_test.at(axis_name).at(j));
      BOOST_CHECK_EQUAL(labels_row_4.at(axis_name).at(j), labels_row_4_test.at(axis_name).at(j));
    }
  }
}

BOOST_AUTO_TEST_CASE(insertIntoTableFromCsvDefaultDevice)
{
  // Set up the device
  Eigen::DefaultDevice device;

  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, 1), labels3(1, 1);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0} });
  labels3.setValues({ {0} });
  auto axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data, the new tensor data from csv, and the new axes labels from csv
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, 1, 1 }));
  Eigen::Tensor<std::string, 2> new_values_str(Eigen::array<Eigen::Index, 2>({ nlabels, 8 }));
  Eigen::Tensor<std::string, 2> labels_2_str(1, 8);
  Eigen::Tensor<std::string, 2> labels_3_str(1, 8);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        if (j == 0 && k == 0) {
          tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
        }
        else {
          int index = j + k * nlabels - 1;
          new_values_str(i, index) = std::to_string(i + j * nlabels + k * nlabels*nlabels);
          labels_2_str(0, index) = std::to_string(j);
          labels_3_str(0, index) = std::to_string(k);
        }
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new axis labels from csv
  std::map<std::string, Eigen::Tensor<std::string, 2>> labels_new_str;
  labels_new_str.emplace("2", labels_2_str);
  labels_new_str.emplace("3", labels_3_str);

  // Setup the device
  tensorTable.insertIntoTableFromCsv(labels_new_str, new_values_str, device);

  // Test for the tensor data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        BOOST_CHECK_EQUAL(tensorTable.getData()(i, j, k), i + j * nlabels + k * nlabels*nlabels);
      }
    }
  }

  // Test for the axis labels
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(axis_1_ptr->getLabels()(0, i), i);
    BOOST_CHECK_EQUAL(axis_2_ptr->getLabels()(0, i), i);
    BOOST_CHECK_EQUAL(axis_3_ptr->getLabels()(0, i), i);
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()