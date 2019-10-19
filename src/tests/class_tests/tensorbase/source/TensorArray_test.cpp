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
BOOST_AUTO_TEST_CASE(constructorTensorArray8DefaultDevice) 
{
	TensorArray8<float>* ptr = nullptr;
	TensorArray8<float>* nullPointer = nullptr;
	ptr = new TensorArray8<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorTensorArray8DefaultDevice)
{
	TensorArray8<float>* ptr = nullptr;
	ptr = new TensorArray8<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersTensorArray8DefaultDevice)
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

BOOST_AUTO_TEST_CASE(comparisonTensorArray8DefaultDevice)
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

BOOST_AUTO_TEST_CASE(getTensorArrayAsStringTensorArray8DefaultDevice)
{
  TensorArray8<int> tensorArrayInt1({ 1,2,3,4,5,6,7,8 });
  // Check << operator
  std::ostringstream os;
  os << tensorArrayInt1;
  BOOST_CHECK_EQUAL(std::string(os.str()), "12345678");

  // Check getter
  BOOST_CHECK_EQUAL(tensorArrayInt1.getTensorArrayAsString(), "12345678");
}

/* TensorArray32 Tests
*/
BOOST_AUTO_TEST_CASE(constructorTensorArray32DefaultDevice)
{
  TensorArray32<float>* ptr = nullptr;
  TensorArray32<float>* nullPointer = nullptr;
  ptr = new TensorArray32<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorTensorArray32DefaultDevice)
{
  TensorArray32<float>* ptr = nullptr;
  ptr = new TensorArray32<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersTensorArray32DefaultDevice)
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(32);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  TensorArray32<float> tensorArrayFloat1(same_equal_float_1);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getArraySize(), 32);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(0), 1);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(1), 2);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(2), 3);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(3), 4);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(4), 5);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(5), 6);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(6), 7);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(7), 8);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(8), 9);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(9), 10);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(10), 11);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(11), 12);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(12), 13);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(13), 14);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(14), 15);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(15), 16);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(16), 17);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(17), 18);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(18), 19);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(19), 20);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(20), 21);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(21), 22);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(22), 23);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(23), 24);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(24), 25);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(25), 26);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(26), 27);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(27), 28);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(28), 29);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(29), 30);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(30), 31);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.getTensorArray()(31), 32);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(0), 1);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(1), 2);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(2), 3);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(3), 4);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(4), 5);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(5), 6);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(6), 7);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(7), 8);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(8), 9);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(9), 10);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(10), 11);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(11), 12);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(12), 13);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(13), 14);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(14), 15);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(15), 16);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(16), 17);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(17), 18);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(18), 19);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(19), 20);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(20), 21);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(21), 22);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(22), 23);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(23), 24);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(24), 25);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(25), 26);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(26), 27);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(27), 28);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(28), 29);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(29), 30);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(30), 31);
  BOOST_CHECK_EQUAL(tensorArrayFloat1.at(31), 32);

  TensorArray32<float> tensorArrayFloat2({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(0), 1);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(1), 2);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(2), 3);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(3), 4);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(4), 5);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(5), 6);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(6), 7);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(7), 8);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(8), 9);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(9), 10);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(10), 11);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(11), 12);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(12), 13);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(13), 14);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(14), 15);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(15), 16);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(16), 17);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(17), 18);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(18), 19);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(19), 20);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(20), 21);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(21), 22);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(22), 23);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(23), 24);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(24), 25);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(25), 26);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(26), 27);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(27), 28);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(28), 29);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(29), 30);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(30), 31);
  BOOST_CHECK_EQUAL(tensorArrayFloat2.at(31), 32);
}

BOOST_AUTO_TEST_CASE(comparisonTensorArray32DefaultDevice)
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(32);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  TensorArray32<float> tensorArrayFloat1(same_equal_float_1);
  Eigen::Tensor<float, 1> same_equal_float_2(32);
  same_equal_float_2.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  TensorArray32<float> tensorArrayFloat2(same_equal_float_2);
  BOOST_CHECK(tensorArrayFloat1 == tensorArrayFloat2);
  BOOST_CHECK(!(tensorArrayFloat1 != tensorArrayFloat2));
  BOOST_CHECK(!(tensorArrayFloat1 < tensorArrayFloat2));
  BOOST_CHECK(!(tensorArrayFloat1 > tensorArrayFloat2));
  BOOST_CHECK(tensorArrayFloat1 <= tensorArrayFloat2);
  BOOST_CHECK(tensorArrayFloat1 >= tensorArrayFloat2);

  // Check different and equal length float
  Eigen::Tensor<float, 1> same_equal_float_3(32);
  same_equal_float_3.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,0,28,29,30,31,32 });
  TensorArray32<float> tensorArrayFloat3(same_equal_float_3);
  BOOST_CHECK(!(tensorArrayFloat1 == tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 != tensorArrayFloat3);
  BOOST_CHECK(!(tensorArrayFloat1 < tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 > tensorArrayFloat3);
  BOOST_CHECK(!(tensorArrayFloat1 <= tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 >= tensorArrayFloat3);
}

BOOST_AUTO_TEST_CASE(getTensorArrayAsStringTensorArray32DefaultDevice)
{
  TensorArray32<int> tensorArrayInt1({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  // Check << operator
  std::ostringstream os;
  os << tensorArrayInt1;
  BOOST_CHECK_EQUAL(std::string(os.str()), "1234567891011121314151617181920212223242526272829303132");

  // Check getter
  BOOST_CHECK_EQUAL(tensorArrayInt1.getTensorArrayAsString(), "1234567891011121314151617181920212223242526272829303132");
}

BOOST_AUTO_TEST_SUITE_END()