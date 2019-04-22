/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollection test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorCollection.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorCollection)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorCollection<TensorTable<float,Eigen::DefaultDevice,3>>* ptr = nullptr;
  TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>* nullPointer = nullptr;
	ptr = new TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>* ptr = nullptr;
	ptr = new TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  TensorCollection<
    TensorTable<float, Eigen::DefaultDevice, 3>,
    TensorTable<int, Eigen::DefaultDevice, 2>,
    TensorTable<char, Eigen::DefaultDevice, 4>
  > tensorCollection;

  // name setters and getters
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "tfloat", "tint", "tchar" }));

  // table getters
  //BOOST_CHECK_EQUAL(tensorCollection.getTableIndex("tfloat"), 0);
  //BOOST_CHECK_EQUAL(tensorCollection.getTable("tfloat")->getName(), "tfloat");
  //BOOST_CHECK_EQUAL(tensorCollection.getTable("tint")->getName(), "tint");
  //BOOST_CHECK_EQUAL(tensorCollection.getTable("tchar")->getName(), "tchar");
}

BOOST_AUTO_TEST_SUITE_END()