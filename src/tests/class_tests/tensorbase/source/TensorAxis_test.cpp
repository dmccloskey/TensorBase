/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorAxis test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorAxis.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorAxis)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
	TensorAxisDefaultDevice<int>* ptr = nullptr;
	TensorAxisDefaultDevice<int>* nullPointer = nullptr;
	ptr = new TensorAxisDefaultDevice<int>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	TensorAxisDefaultDevice<int>* ptr = nullptr;
	ptr = new TensorAxisDefaultDevice<int>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor1DefaultDevice)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), -1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), 1);
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorAxisDefaultDevice<int> tensoraxis;
  // Check defaults
  BOOST_CHECK_EQUAL(tensoraxis.getId(), -1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "");
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 0);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 0);

  // Check getters/setters
  tensoraxis.setId(1);
  tensoraxis.setName("1");
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  tensoraxis.setDimensionsAndLabels(dimensions, labels);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), 1);
}

BOOST_AUTO_TEST_SUITE_END()