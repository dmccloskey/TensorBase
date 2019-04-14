/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorDimension test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorDimension.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorDimension)

BOOST_AUTO_TEST_CASE(constructor) 
{
	TensorDimension* ptr = nullptr;
	TensorDimension* nullPointer = nullptr;
	ptr = new TensorDimension();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
	TensorDimension* ptr = nullptr;
	ptr = new TensorDimension();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  TensorDimension tensordimension;
  // Check defaults
  BOOST_CHECK_EQUAL(tensordimension.getId(), 0);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 0);

  // Check getters/setters
  tensordimension.setId(1);
  tensordimension.setName("1");
  Eigen::Tensor<std::string, 1> labels(5);
  labels.setConstant("Hello!");
  tensordimension.setLabels(labels);

  BOOST_CHECK_EQUAL(tensordimension.getId(), 1);
  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), "Hello!");
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), "Hello!");
}

BOOST_AUTO_TEST_SUITE_END()