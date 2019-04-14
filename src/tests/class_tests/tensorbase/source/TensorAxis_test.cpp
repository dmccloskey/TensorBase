/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorAxis test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorAxis.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorAxis)

BOOST_AUTO_TEST_CASE(constructor) 
{
	TensorAxis* ptr = nullptr;
	TensorAxis* nullPointer = nullptr;
	ptr = new TensorAxis();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
	TensorAxis* ptr = nullptr;
	ptr = new TensorAxis();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  TensorAxis tensoraxis;
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
  Eigen::Tensor<std::string, 2> labels(3, 5);
  labels.setConstant("Hello!");
  tensoraxis.setDimensionsAndLabels(dimensions, labels);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), "Hello!");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), "Hello!");
}

BOOST_AUTO_TEST_SUITE_END()