/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE HeteroTensor test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/HeteroTensor.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(heteroTensor)

BOOST_AUTO_TEST_CASE(constructor) 
{
  HeteroTensorDefaultDevice<3>* ptr = nullptr;
  HeteroTensorDefaultDevice<3>* nullPointer = nullptr;
	ptr = new HeteroTensorDefaultDevice<3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
  HeteroTensorDefaultDevice<3>* ptr = nullptr;
	ptr = new HeteroTensorDefaultDevice<3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  HeteroTensorDefaultDevice<3, float, int> heterotensor;
  // Check defaults
  BOOST_CHECK_EQUAL(heterotensor.getId(), -1);
  BOOST_CHECK_EQUAL(heterotensor.getName(), "");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().size(), 0);

  // Check getters/setters
  heterotensor.setId(1);
  heterotensor.setName("1");

  BOOST_CHECK_EQUAL(heterotensor.getId(), 1);
  BOOST_CHECK_EQUAL(heterotensor.getName(), "1");

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
  heterotensor.setAxes({ TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
   });

  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("1")->getName(), "1");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("1")->getLabels()(0,0), "x-axis");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("1")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("1")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(heterotensor.getIndices().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(heterotensor.getIndices().at("1")->operator()(nlabels1 -1), nlabels1);
  BOOST_CHECK_EQUAL(heterotensor.getIndicesView().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(heterotensor.getIndicesView().at("1")->operator()(nlabels1 - 1), nlabels1);
  BOOST_CHECK_EQUAL(heterotensor.getIsModified().at("1")->operator()(0), 0);
  BOOST_CHECK_EQUAL(heterotensor.getInMemory().at("1")->operator()(0), 0);
  BOOST_CHECK_EQUAL(heterotensor.getIsShardable().at("1")->operator()(0), 1);

  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("2")->getName(), "2");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("2")->getLabels()(0, 0), "y-axis");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("2")->getNLabels(), nlabels2);
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("2")->getDimensions()(0), "y");
  BOOST_CHECK_EQUAL(heterotensor.getIndices().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(heterotensor.getIndices().at("2")->operator()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(heterotensor.getIndicesView().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(heterotensor.getIndicesView().at("2")->operator()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(heterotensor.getIsModified().at("2")->operator()(0), 0);
  BOOST_CHECK_EQUAL(heterotensor.getInMemory().at("2")->operator()(0), 0);
  BOOST_CHECK_EQUAL(heterotensor.getIsShardable().at("2")->operator()(0), 0);

  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("3")->getName(), "3");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("3")->getLabels()(0, 0), "z-axis");
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("3")->getNLabels(), nlabels3);
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(heterotensor.getAxes().at("3")->getDimensions()(0), "z");
  BOOST_CHECK_EQUAL(heterotensor.getIndices().at("3")->operator()(0), 1);
  BOOST_CHECK_EQUAL(heterotensor.getIndices().at("3")->operator()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(heterotensor.getIndicesView().at("3")->operator()(0), 1);
  BOOST_CHECK_EQUAL(heterotensor.getIndicesView().at("3")->operator()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(heterotensor.getIsModified().at("3")->operator()(0), 0);
  BOOST_CHECK_EQUAL(heterotensor.getInMemory().at("3")->operator()(0), 0);
  BOOST_CHECK_EQUAL(heterotensor.getIsShardable().at("3")->operator()(0), 0);

  // SetTensors associated getters/setters
  heterotensor.setTensors({ std::vector<std::string>({ "x-axis" }), std::vector<std::string>({ "x-axis" }) });
}

BOOST_AUTO_TEST_SUITE_END()