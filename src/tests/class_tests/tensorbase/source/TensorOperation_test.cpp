/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorOperation test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorOperation.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorOperation1)

/*TensorAppendToAxis Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorAppendToAxis)
{
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
	ptr = new TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorAppendToAxis)
{
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
	ptr = new TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(redoTensorAppendToAxis)
{
  // TODO
}

BOOST_AUTO_TEST_CASE(undoTensorAppendToAxis)
{
  // TODO
}

/*TensorDeleteFromAxis Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorDeleteFromAxis)
{
  TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
  ptr = new TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorDeleteFromAxis)
{
  TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  ptr = new TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(redoTensorDeleteFromAxis)
{
  // TODO
}

BOOST_AUTO_TEST_CASE(undoTensorDeleteFromAxis)
{
  // TODO
}

/*TensorUpdate Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorUpdate)
{
  TensorUpdate<float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorUpdate<float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
  ptr = new TensorUpdate<float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorUpdate)
{
  TensorUpdate<float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  ptr = new TensorUpdate<float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(redoTensorUpdate)
{
  // TODO
}

BOOST_AUTO_TEST_CASE(undoTensorUpdate)
{
  // TODO
}

BOOST_AUTO_TEST_SUITE_END()