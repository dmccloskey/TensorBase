/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorType test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorType.h>
#include <cassert>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorType)

BOOST_AUTO_TEST_CASE(Int8Type1)
{
  Int8Type tensorType;

  // Check getters
  BOOST_CHECK_EQUAL(tensorType.getId(), Type::INT8);
  BOOST_CHECK_EQUAL(tensorType.getName(), "int8");

  // Inheritance and type checks
  static_assert(sizeof(Int8Type) != sizeof(int8_t), "The wrapper does not have overhead."); // It should have some overhead
  static_assert(std::is_same<Int8Type::c_type, int8_t>::value, "Value type is not exposed.");
  static_assert(std::is_base_of<TensorType, Int8Type>::value, "Int8Type is not the base of TensorType.");
  static_assert(std::is_convertible<Int8Type::c_type*, int8_t*>::value, "Value types are not compatible.");

  // Brief and incomplete set of basic operator checks
  // to ensure that the fundamental type operators have been inherited
  tensorType = 0; BOOST_CHECK(tensorType == 0);
  tensorType += 1; BOOST_CHECK(tensorType == 1);  
  ++tensorType; BOOST_CHECK(tensorType == 2);  
  BOOST_CHECK(tensorType < 3);
}

// TODO implement all other tests

BOOST_AUTO_TEST_CASE(HeteroVector1)
{
  std::vector<std::shared_ptr<TensorType>> heteroVecT;
  heteroVecT.push_back(std::shared_ptr<TensorType>(new BooleanType(true)));
  heteroVecT.push_back(std::shared_ptr<TensorType>(new CharType('a')));
  heteroVecT.push_back(std::shared_ptr<TensorType>(new UInt8Type(1)));

  std::cout << heteroVecT[0]->getName() << heteroVecT[0] << std::endl;
  std::cout << heteroVecT[1]->getName() << heteroVecT[1] << std::endl;
  std::cout << heteroVecT[2]->getName() << heteroVecT[2] << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()