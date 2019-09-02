/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorArray test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorArray.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorArray)

/* TensorArrayOperatorsCpu Tests
*/
BOOST_AUTO_TEST_CASE(compareDefaultDevice)
{
  // Test equal for same size array
  Eigen::Tensor<float, 1> same_equal_1(5);
  same_equal_1.setValues({ 1,2,3,4,5 });
  Eigen::Tensor<float, 1> same_equal_2(5);
  same_equal_2.setValues({ 1,2,3,4,5 });
  BOOST_CHECK(TensorArrayOperatorsCpu::isEqualTo(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isNotEqualTo(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThan(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isGreaterThan(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isLessThanOrEqualTo(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(same_equal_1.data(), same_equal_2.data(), 5));

  // Test different for same size array
  Eigen::Tensor<float, 1> diff_equal_2(5);
  diff_equal_2.setValues({ 1,2,0,4,5 });
  BOOST_CHECK(!TensorArrayOperatorsCpu::isEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isNotEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThan(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThan(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThanOrEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
}

BOOST_AUTO_TEST_CASE(compareCharDefaultDevice)
{
  // Test equal for same size array
  Eigen::Tensor<char, 1> same_equal_1(5);
  same_equal_1.setValues({ 'a', 'b', 'c', 'd', 'e' });
  Eigen::Tensor<char, 1> same_equal_2(5);
  same_equal_2.setValues({ 'a', 'b', 'c', 'd', 'e' });
  BOOST_CHECK(TensorArrayOperatorsCpu::isEqualTo(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isNotEqualTo(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThan(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isGreaterThan(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isLessThanOrEqualTo(same_equal_1.data(), same_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(same_equal_1.data(), same_equal_2.data(), 5));

  Eigen::Tensor<char, 1> same_equal_3(5);
  same_equal_1.setValues({ 'a', 'b', 'c', 'd', '\0' });
  Eigen::Tensor<char, 1> same_equal_4(5);
  same_equal_2.setValues({ 'a', 'b', 'c', 'd', '\0' });
  BOOST_CHECK(TensorArrayOperatorsCpu::isEqualTo(same_equal_3.data(), same_equal_4.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isNotEqualTo(same_equal_3.data(), same_equal_4.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThan(same_equal_3.data(), same_equal_4.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isGreaterThan(same_equal_3.data(), same_equal_4.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isLessThanOrEqualTo(same_equal_3.data(), same_equal_4.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(same_equal_3.data(), same_equal_4.data(), 5));

  // Test different for same size array
  Eigen::Tensor<char, 1> diff_equal_2(5);
  diff_equal_2.setValues({ 'a', 'b', 'c', 'a', 'd' });
  BOOST_CHECK(!TensorArrayOperatorsCpu::isEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isNotEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThan(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThan(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThanOrEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(same_equal_1.data(), diff_equal_2.data(), 5));

  Eigen::Tensor<char, 1> diff_unequal_2(5);
  diff_unequal_2.setValues({ 'a', 'b', 'c', '\0', '\0' });
  BOOST_CHECK(!TensorArrayOperatorsCpu::isEqualTo(same_equal_1.data(), diff_unequal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isNotEqualTo(same_equal_1.data(), diff_unequal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThan(same_equal_1.data(), diff_unequal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThan(same_equal_1.data(), diff_unequal_2.data(), 5));
  BOOST_CHECK(!TensorArrayOperatorsCpu::isLessThanOrEqualTo(same_equal_1.data(), diff_unequal_2.data(), 5));
  BOOST_CHECK(TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(same_equal_1.data(), diff_unequal_2.data(), 5));
}

/* TensorArrayDefaultDevice Tests
*/
BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
	TensorArrayDefaultDevice<float>* ptr = nullptr;
	TensorArrayDefaultDevice<float>* nullPointer = nullptr;
	ptr = new TensorArrayDefaultDevice<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	TensorArrayDefaultDevice<float>* ptr = nullptr;
	ptr = new TensorArrayDefaultDevice<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparisonDefaultDevice)
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(5);
  same_equal_float_1.setValues({ 1,2,3,4,5 });
  TensorArrayDefaultDevice<float> tensorArrayFloat1(same_equal_float_1);
  Eigen::Tensor<float, 1> same_equal_float_2(5);
  same_equal_float_2.setValues({ 1,2,3,4,5 });
	TensorArrayDefaultDevice<float> tensorArrayFloat2(same_equal_float_2);
	BOOST_CHECK(tensorArrayFloat1 == tensorArrayFloat2);
  BOOST_CHECK(!(tensorArrayFloat1 != tensorArrayFloat2));
  BOOST_CHECK(!(tensorArrayFloat1 < tensorArrayFloat2));
  BOOST_CHECK(!(tensorArrayFloat1 > tensorArrayFloat2));
  BOOST_CHECK(tensorArrayFloat1 <= tensorArrayFloat2);
  BOOST_CHECK(tensorArrayFloat1 >= tensorArrayFloat2);

  // Check different and equal length float
  Eigen::Tensor<float, 1> same_equal_float_3(5);
  same_equal_float_3.setValues({ 1,2,0,4,5 });
  TensorArrayDefaultDevice<float> tensorArrayFloat3(same_equal_float_3);
  BOOST_CHECK(!(tensorArrayFloat1 == tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 != tensorArrayFloat3);
  BOOST_CHECK(!(tensorArrayFloat1 < tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 > tensorArrayFloat3);
  BOOST_CHECK(!(tensorArrayFloat1 <= tensorArrayFloat3));
  BOOST_CHECK(tensorArrayFloat1 >= tensorArrayFloat3);

  // Check same and equal length char
  Eigen::Tensor<char, 1> same_equal_char_1(5);
  same_equal_char_1.setValues({ 'a', 'b', 'c', 'd', 'e' });
  TensorArrayDefaultDevice<char> tensorArrayChar1(same_equal_char_1);
  Eigen::Tensor<char, 1> same_equal_char_2(5);
  same_equal_char_2.setValues({ 'a', 'b', 'c', 'd', 'e' });
  TensorArrayDefaultDevice<char> tensorArrayChar2(same_equal_char_2);
  BOOST_CHECK(tensorArrayChar1 == tensorArrayChar2);
  BOOST_CHECK(!(tensorArrayChar1 != tensorArrayChar2));
  BOOST_CHECK(!(tensorArrayChar1 < tensorArrayChar2));
  BOOST_CHECK(!(tensorArrayChar1 > tensorArrayChar2));
  BOOST_CHECK(tensorArrayChar1 <= tensorArrayChar2);
  BOOST_CHECK(tensorArrayChar1 >= tensorArrayChar2);

  // Check different and unqeual length char
  Eigen::Tensor<char, 1> same_equal_char_3(4);
  same_equal_char_3.setValues({ 'a', 'b', 'a', 'd', 'e' });
  TensorArrayDefaultDevice<char> tensorArrayChar3(same_equal_char_3);
  BOOST_CHECK(!(tensorArrayChar1 == tensorArrayChar3));
  BOOST_CHECK(tensorArrayChar1 != tensorArrayChar3);
  BOOST_CHECK(!(tensorArrayChar1 < tensorArrayChar3));
  BOOST_CHECK(tensorArrayChar1 > tensorArrayChar3);
  BOOST_CHECK(!(tensorArrayChar1 <= tensorArrayChar3));
  BOOST_CHECK(tensorArrayChar1 >= tensorArrayChar3);
}

BOOST_AUTO_TEST_CASE(assignmentDefaultDevice)
{
  //TensorArrayDefaultDevice<float> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  //Eigen::Tensor<float> data(2, 3, 4);
  //data.setConstant(1);
  //tensordata_test.setData(data);

  //// Check copy
  //TensorArrayDefaultDevice<float> tensordata(tensordata_test);
  //BOOST_CHECK(tensordata == tensordata_test);
  //BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 1);

  //// Check reference sharing
  //tensordata.getData()(0, 0, 0) = 2;
  //BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(copyDefaultDevice)
{
  //TensorArrayDefaultDevice<float> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
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

/* TensorArrayCpu Tests
*/
BOOST_AUTO_TEST_CASE(constructorCpu)
{
  TensorArrayCpu<float>* ptr = nullptr;
  TensorArrayCpu<float>* nullPointer = nullptr;
  ptr = new TensorArrayCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorCpu)
{
  TensorArrayCpu<float>* ptr = nullptr;
  ptr = new TensorArrayCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(assignmentCpu)
{
  //TensorArrayCpu<float> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  //Eigen::Tensor<float> data(2, 3, 4);
  //data.setConstant(1);
  //tensordata_test.setData(data);

  //// Check copy
  //TensorArrayCpu<float> tensordata(tensordata_test);
  //BOOST_CHECK(tensordata == tensordata_test);
  //BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 1);

  //// Check reference sharing
  //tensordata.getData()(0, 0, 0) = 2;
  //BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(copyCpu)
{
  //TensorArrayCpu<float> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  //Eigen::Tensor<float> data(2, 3, 4);
  //data.setConstant(1);
  //tensordata_test.setData(data);
  //Eigen::ThreadPool pool(1);
  //Eigen::ThreadPoolDevice device(&pool, 1);

  //// Check copy
  //std::shared_ptr<TensorArray<float, Eigen::ThreadPoolDevice, 3>> tensordata = tensordata_test.copy(device);
  //BOOST_CHECK(tensordata->getDimensions() == tensordata_test.getDimensions());
  //BOOST_CHECK(tensordata->getTensorBytes() == tensordata_test.getTensorBytes());
  //BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  //BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  //// Check reference change
  //tensordata->getData()(0, 0, 0) = 2;
  //BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  //BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
}

BOOST_AUTO_TEST_CASE(selectCpu)
{
}

BOOST_AUTO_TEST_SUITE_END()