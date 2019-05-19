/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorDimension test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorDimension.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorDimension)

/*TensorDimensionDefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
	TensorDimensionDefaultDevice<std::string>* ptr = nullptr;
	TensorDimensionDefaultDevice<std::string>* nullPointer = nullptr;
	ptr = new TensorDimensionDefaultDevice<std::string>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	TensorDimensionDefaultDevice<std::string>* ptr = nullptr;
	ptr = new TensorDimensionDefaultDevice<std::string>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructorNameDefaultDevice)
{
  TensorDimensionDefaultDevice<std::string> tensordimension("1");
  BOOST_CHECK_EQUAL(tensordimension.getId(), -1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 0);
}

BOOST_AUTO_TEST_CASE(constructorNameAndLabelsDefaultDevice)
{
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  TensorDimensionDefaultDevice<int> tensordimension("1", labels);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), 1);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), 1);
}

// TODO: switch from `std::string` to fixed length string implementation
//BOOST_AUTO_TEST_CASE(constructorNameAndLabelsStringDefaultDevice)
//{
//  Eigen::Tensor<std::string, 1> labels(5);
//  labels.setConstant("Hello!");
//  TensorDimensionDefaultDevice<std::string> tensordimension("1", labels);
//  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
//  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
//  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), "Hello!");
//  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), "Hello!");
//}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorDimensionDefaultDevice<int> tensordimension;
  // Check defaults
  BOOST_CHECK_EQUAL(tensordimension.getId(), -1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 0);

  // Check getters/setters
  tensordimension.setId(1);
  tensordimension.setName("1");
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  tensordimension.setLabels(labels);

  BOOST_CHECK_EQUAL(tensordimension.getId(), 1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), 1);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), 1);
}

// TODO: switch from `std::string` to fixed length string implementation
//BOOST_AUTO_TEST_CASE(gettersAndSettersStringDefaultDevice)
//{
//  TensorDimensionDefaultDevice<std::string> tensordimension;
//  // Check defaults
//  BOOST_CHECK_EQUAL(tensordimension.getId(), -1);
//  BOOST_CHECK_EQUAL(tensordimension.getName(), "");
//  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 0);
//
//  // Check getters/setters
//  tensordimension.setId(1);
//  tensordimension.setName("1");
//  Eigen::Tensor<std::string, 1> labels(5);
//  labels.setConstant("Hello!");
//  tensordimension.setLabels(labels);
//
//  BOOST_CHECK_EQUAL(tensordimension.getId(), 1);
//  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
//  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
//  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), "Hello!");
//  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), "Hello!");
//}

/*TensorDimensionCpu Tests*/
BOOST_AUTO_TEST_CASE(constructorCpu)
{
  TensorDimensionCpu<std::string>* ptr = nullptr;
  TensorDimensionCpu<std::string>* nullPointer = nullptr;
  ptr = new TensorDimensionCpu<std::string>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorCpu)
{
  TensorDimensionCpu<std::string>* ptr = nullptr;
  ptr = new TensorDimensionCpu<std::string>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructorNameCpu)
{
  TensorDimensionCpu<std::string> tensordimension("1");
  BOOST_CHECK_EQUAL(tensordimension.getId(), -1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 0);
}

BOOST_AUTO_TEST_CASE(constructorNameAndLabelsCpu)
{
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  TensorDimensionCpu<int> tensordimension("1", labels);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), 1);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), 1);
}

BOOST_AUTO_TEST_CASE(gettersAndSettersCpu)
{
  TensorDimensionCpu<int> tensordimension;
  // Check defaults
  BOOST_CHECK_EQUAL(tensordimension.getId(), -1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 0);

  // Check getters/setters
  tensordimension.setId(1);
  tensordimension.setName("1");
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  tensordimension.setLabels(labels);

  BOOST_CHECK_EQUAL(tensordimension.getId(), 1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), 1);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), 1);
}

BOOST_AUTO_TEST_SUITE_END()