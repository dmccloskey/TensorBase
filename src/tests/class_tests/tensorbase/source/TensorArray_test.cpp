/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorArray test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorArray.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorArray)

/* TensorArray8 Tests
*/
BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
	TensorArray8<float>* ptr = nullptr;
	TensorArray8<float>* nullPointer = nullptr;
	ptr = new TensorArray8<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	TensorArray8<float>* ptr = nullptr;
	ptr = new TensorArray8<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(8);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8 });
  TensorArray8<float> tensorArrayFloat1(same_equal_float_1);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getArraySize(), 8);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(0), 1);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(1), 2);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(2), 3);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(3), 4);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(4), 5);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(5), 6);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(6), 7);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(7), 8);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(0), 1);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(1), 2);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(2), 3);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(3), 4);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(4), 5);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(5), 6);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(6), 7);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(7), 8);

  // Check same and equal length char
  TensorArray8<char> tensorArrayChar1({ '1','2','3','4','5','6','7','8' });
  BOOST_CHECK_EQUAL(tensorArrayChar1.getArraySize(), 8);
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(0), '1');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(1), '2');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(2), '3');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(3), '4');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(4), '5');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(5), '6');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(6), '7');
  BOOST_CHECK_EQUAL(tensorArrayChar1.getTensorArray()(7), '8');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(0), '1');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(1), '2');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(2), '3');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(3), '4');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(4), '5');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(5), '6');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(6), '7');
  BOOST_CHECK_EQUAL(tensorArrayChar1.at(7), '8');

  TensorArray8<char> tensorArrayChar2({ '1','2','3','4','5','6' });
  BOOST_CHECK_EQUAL(tensorArrayChar2.getArraySize(), 8);
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(0), '1');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(1), '2');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(2), '3');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(3), '4');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(4), '5');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(5), '6');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(6), '\0');
  BOOST_CHECK_EQUAL(tensorArrayChar2.getTensorArray()(7), '\0');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(0), '1');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(1), '2');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(2), '3');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(3), '4');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(4), '5');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(5), '6');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(6), '\0');
  BOOST_CHECK_EQUAL(tensorArrayChar2.at(7), '\0');

  // Check same and equal length char
  TensorArray8<char> tensorArrayString1("12345678");
  BOOST_CHECK_EQUAL(tensorArrayString1.getArraySize(), 8);
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(0), '1');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(1), '2');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(2), '3');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(3), '4');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(4), '5');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(5), '6');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(6), '7');
  BOOST_CHECK_EQUAL(tensorArrayString1.getTensorArray()(7), '8');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(0), '1');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(1), '2');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(2), '3');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(3), '4');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(4), '5');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(5), '6');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(6), '7');
  BOOST_CHECK_EQUAL(tensorArrayString1.at(7), '8');
}

BOOST_AUTO_TEST_CASE(comparisonDefaultDevice)
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(8);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8 });
  TensorArray8<float> tensorArrayFloat1(same_equal_float_1);
  Eigen::Tensor<float, 1> same_equal_float_2(8);
  same_equal_float_2.setValues({ 1,2,3,4,5,6,7,8 });
	TensorArray8<float> tensorArrayFloat2(same_equal_float_2);
	BOOST_CHECK(tensorArrayFloat1 == tensorArrayFloat2);
  BOOST_CHECK(!(tensorArrayFloat1 != tensorArrayFloat2));
  BOOST_CHECK(!(tensorArrayFloat1 < tensorArrayFloat2));
  BOOST_CHECK(!(tensorArrayFloat1 > tensorArrayFloat2));
  BOOST_CHECK(tensorArrayFloat1 <= tensorArrayFloat2);
  BOOST_CHECK(tensorArrayFloat1 >= tensorArrayFloat2);

  // Check different and equal length float
  Eigen::Tensor<float, 1> same_equal_float_3(8);
  same_equal_float_3.setValues({ 1,2,0,4,5,6,7,8 });
  TensorArray8<float> tensorArrayFloat3(same_equal_float_3);
  BOOST_CHECK(!(tensorArrayFloat1 == tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 != tensorArrayFloat3);
  BOOST_CHECK(!(tensorArrayFloat1 < tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 > tensorArrayFloat3);
  BOOST_CHECK(!(tensorArrayFloat1 <= tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 >= tensorArrayFloat3);

  // Check same and equal length char
  Eigen::Tensor<char, 1> same_equal_char_1(8);
  same_equal_char_1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar1(same_equal_char_1);
  Eigen::Tensor<char, 1> same_equal_char_2(8);
  same_equal_char_2.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar2(same_equal_char_2);
  BOOST_CHECK(tensorArrayChar1 == tensorArrayChar2);
  BOOST_CHECK(!(tensorArrayChar1 != tensorArrayChar2));
  BOOST_CHECK(!(tensorArrayChar1 < tensorArrayChar2));
  BOOST_CHECK(!(tensorArrayChar1 > tensorArrayChar2));
  BOOST_CHECK(tensorArrayChar1 <= tensorArrayChar2);
  BOOST_CHECK(tensorArrayChar1 >= tensorArrayChar2);

  // Check different and equal length char
  Eigen::Tensor<char, 1> same_equal_char_3(8);
  same_equal_char_3.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar3(same_equal_char_3);
  BOOST_CHECK(!(tensorArrayChar1 == tensorArrayChar3));
  BOOST_CHECK(tensorArrayChar1 != tensorArrayChar3);
  BOOST_CHECK(!(tensorArrayChar1 < tensorArrayChar3));
  BOOST_CHECK(tensorArrayChar1 > tensorArrayChar3);
  BOOST_CHECK(!(tensorArrayChar1 <= tensorArrayChar3));
  BOOST_CHECK(tensorArrayChar1 >= tensorArrayChar3);
}

BOOST_AUTO_TEST_CASE(getTensorArrayAsStringDefaultDevice)
{
  TensorArray8<int> tensorArrayInt1({ 1,2,3,4,5,6,7,8 });
  // Check << operator
  std::ostringstream os;
  os << tensorArrayInt1;
  BOOST_CHECK_EQUAL(std::string(os.str()), "12345678");

  // Check getter
  BOOST_CHECK_EQUAL(tensorArrayInt1.getTensorArrayAsString(), "12345678");
}

BOOST_AUTO_TEST_CASE(selectDefaultDevice)
{
  Eigen::Tensor<char, 1> same_equal_char_1(8);
  same_equal_char_1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar1(same_equal_char_1);
  Eigen::Tensor<char, 1> same_equal_char_2(8);
  same_equal_char_2.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar2(same_equal_char_2);

  Eigen::Tensor<TensorArray8<char>, 1> in1(2);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2 });

  Eigen::Tensor<TensorArray8<char>, 1> selected = (in1 == in1).select(in1, in1);
}

BOOST_AUTO_TEST_SUITE_END()