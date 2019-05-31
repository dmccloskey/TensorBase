/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollectionFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/TensorCollectionFile.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorCollectionFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorCollectionFileDefaultDevice* ptr = nullptr;
  TensorCollectionFileDefaultDevice* nullPointer = nullptr;
  ptr = new TensorCollectionFileDefaultDevice();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  TensorCollectionFileDefaultDevice* ptr = nullptr;
	ptr = new TensorCollectionFileDefaultDevice();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary) 
{
  TensorCollectionFileDefaultDevice data;

  std::string filename = "TensorCollectionFileDefaultDeviceTest.dat";
}

BOOST_AUTO_TEST_SUITE_END()