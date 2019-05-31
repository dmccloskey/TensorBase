/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollectionFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/TensorCollectionFile.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorCollectionFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorCollectionFile* ptr = nullptr;
  TensorCollectionFile* nullPointer = nullptr;
  ptr = new TensorCollectionFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  TensorCollectionFile* ptr = nullptr;
	ptr = new TensorCollectionFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary) 
{
  TensorCollectionFile data;

  std::string filename = "TensorCollectionFileTest.dat";
}

BOOST_AUTO_TEST_SUITE_END()