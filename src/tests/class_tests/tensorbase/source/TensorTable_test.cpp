/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorTable test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorTable.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorTable)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorTableDefaultDevice<float, 3>* ptr = nullptr;
  TensorTableDefaultDevice<float, 3>* nullPointer = nullptr;
	ptr = new TensorTableDefaultDevice<float, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorTableDefaultDevice<float, 3>* ptr = nullptr;
	ptr = new TensorTableDefaultDevice<float, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructorNameAndAxes)
{
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant("x-axis");
  labels2.setConstant("y-axis");
  labels3.setConstant("z-axis");

  TensorTableDefaultDevice<float, 3> tensorTable("1", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
    });

  BOOST_CHECK_EQUAL(tensorTable.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "1");

  // Test expected axes values
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getName(), "1");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getLabels()(0, 0), "x-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->operator()(nlabels1 - 1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->operator()(nlabels1 - 1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("1")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("1")->operator()(0), 1);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getName(), "2");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getLabels()(0, 0), "y-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNLabels(), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->operator()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->operator()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("2")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("2")->operator()(0), 0);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getName(), "3");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getLabels()(0, 0), "z-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNLabels(), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->operator()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->operator()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("3")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("3")->operator()(0), 0);

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
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
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
  Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant("x-axis");
  labels2.setConstant("y-axis");
  labels3.setConstant("z-axis");
  tensorTable.setAxes({ TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
   });

  // Test expected axes values
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getName(), "1");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getLabels()(0,0), "x-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->operator()(nlabels1 -1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->operator()(nlabels1 - 1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("1")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("1")->operator()(0), 1);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getName(), "2");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getLabels()(0, 0), "y-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNLabels(), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->operator()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->operator()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("2")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("2")->operator()(0), 0);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getName(), "3");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getLabels()(0, 0), "z-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNLabels(), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->operator()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->operator()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->operator()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("3")->operator()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("3")->operator()(0), 0);

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