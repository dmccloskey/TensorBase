/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorDimension test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorDimensionDefaultDevice.h>

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

BOOST_AUTO_TEST_CASE(loadAndStoreLabelsDefaultDevice)
{
  // initialize the dimensions
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  TensorDimensionDefaultDevice<int> tensordimension_io("1", labels);

  // write the dimension labels to disk
  Eigen::DefaultDevice device;
  tensordimension_io.storeLabelsBinary("Dimensions_test.bin", device);

  // read the dimension labels from disk
  TensorDimensionDefaultDevice<int> tensordimension("1", 5);
  tensordimension.loadLabelsBinary("Dimensions_test.bin", device);

  BOOST_CHECK_EQUAL(tensordimension.getName(), "1");
  BOOST_CHECK_EQUAL(tensordimension.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(0), 1);
  BOOST_CHECK_EQUAL(tensordimension.getLabels()(4), 1);
}

BOOST_AUTO_TEST_SUITE_END()