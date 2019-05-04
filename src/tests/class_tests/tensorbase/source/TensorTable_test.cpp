/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorTable test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorTable.h>

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
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorTableDefaultDevice<float, 3> tensorTable;
  // Check defaults
  BOOST_CHECK_EQUAL(tensorTable.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().size(), 0);

  // Check getters/setters
  tensorTable.setId(1);
  tensorTable.setName("1");

  BOOST_CHECK_EQUAL(tensorTable.getId(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "1");

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
  tensorTable.setAxes();

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
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("1")->getData()(0), 1);

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
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("2")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("2")->getData()(0), 0);

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
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("3")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("3")->getData()(0), 0);

  // Test expected axis to dims mapping
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("1"), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("2"), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor dimensions
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 5);

  // Test expected tensor data values
  BOOST_CHECK_EQUAL(tensorTable.getData()->getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getData()->getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getData()->getDimensions().at(2), 5);
  size_t test = 2 * 3 * 5 * sizeof(float);
  BOOST_CHECK_EQUAL(tensorTable.getData()->getTensorSize(), test);

  // Test clear
  tensorTable.clear();
  BOOST_CHECK_EQUAL(tensorTable.getAxes().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 0);
  BOOST_CHECK_EQUAL(tensorTable.getData(), nullptr);
}

BOOST_AUTO_TEST_SUITE_END()