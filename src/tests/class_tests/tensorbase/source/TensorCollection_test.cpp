/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollection test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorCollection.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorCollection)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorCollection* ptr = nullptr;
  TensorCollection* nullPointer = nullptr;
	ptr = new TensorCollection();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorCollection* ptr = nullptr;
	ptr = new TensorCollection();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  TensorCollection tensorCollection;
  // Check defaults
  BOOST_CHECK_EQUAL(tensorCollection.getId(), -1);
  BOOST_CHECK_EQUAL(tensorCollection.getName(), "");

  // Check getters/setters
  tensorCollection.setId(1);
  tensorCollection.setName("1");
  BOOST_CHECK_EQUAL(tensorCollection.getId(), 1);
  BOOST_CHECK_EQUAL(tensorCollection.getName(), "1");
}

BOOST_AUTO_TEST_SUITE_END()