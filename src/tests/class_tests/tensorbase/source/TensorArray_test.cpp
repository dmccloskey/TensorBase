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

BOOST_AUTO_TEST_CASE(assignmentDefaultDevice)
{
  //TensorArray8<float> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  //Eigen::Tensor<float> data(2, 3, 4);
  //data.setConstant(1);
  //tensordata_test.setData(data);

  //// Check copy
  //TensorArray8<float> tensordata(tensordata_test);
  //BOOST_CHECK(tensordata == tensordata_test);
  //BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 1);

  //// Check reference sharing
  //tensordata.getData()(0, 0, 0) = 2;
  //BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(copyDefaultDevice)
{
  //TensorArray8<float> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  //Eigen::Tensor<float> data(2, 3, 4);
  //data.setConstant(1);
  //tensordata_test.setData(data);
  //Eigen::DefaultDevice device;

  //// Check copy
  //std::shared_ptr<TensorArray<float, Eigen::DefaultDevice, 3>> tensordata = tensordata_test.copy(device);
  //BOOST_CHECK(tensordata->getDimensions() == tensordata_test.getDimensions());
  //BOOST_CHECK(tensordata->getTensorBytes() == tensordata_test.getTensorBytes());
  //BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  //BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  //// Check reference change
  //tensordata->getData()(0, 0, 0) = 2;
  //BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  //BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
}

BOOST_AUTO_TEST_CASE(selectDefaultDevice)
{
}

BOOST_AUTO_TEST_SUITE_END()