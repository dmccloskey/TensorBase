/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorArray test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorArray.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorArray)

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
 // // Check null
	//TensorArrayDefaultDevice<float> tensordata, tensordata_test;
	//BOOST_CHECK(tensordata == tensordata_test);

 // // Check same size
 // BOOST_CHECK(tensordata == tensordata_test);

 // // Check different sizes
 // BOOST_CHECK(tensordata != tensordata_test);
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