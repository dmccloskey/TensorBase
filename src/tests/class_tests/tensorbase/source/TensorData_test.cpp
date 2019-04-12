/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorData test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorData.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorData)

BOOST_AUTO_TEST_CASE(constructor) 
{
	TensorDataCpu<float>* ptr = nullptr;
	TensorDataCpu<float>* nullPointer = nullptr;
	ptr = new TensorDataCpu<float>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor) 
{
	TensorDataCpu<float>* ptr = nullptr;
	ptr = new TensorDataCpu<float>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparison) 
{
	TensorDataCpu<float> tensordata, tensordata_test;
	BOOST_CHECK(tensordata == tensordata_test);
}

#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	TensorDataGpu<float, int, int, int> tensordata(2, 3, 4);

  Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

  tensordata.setData(data);
	BOOST_CHECK_EQUAL(tensordata.getData()(1, 2, 3), 0.5);
	BOOST_CHECK(tensordata.getDataStatus().first);
	BOOST_CHECK(!tensordata.getDataStatus().second);

	// Test mutability
  tensordata.getData()(0, 0, 0) = 5;

	BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 5);
}

BOOST_AUTO_TEST_CASE(syncHAndD2)
{
	TensorDataGpu<float, int, int, int> tensordata(2, 3, 4);

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

  tensordata.setData(data);

	Eigen::GpuStreamDevice stream_device;
	Eigen::GpuDevice device(&stream_device);
  tensordata.syncHAndDData(device);

	BOOST_CHECK(!tensordata.getDataStatus().first);
	BOOST_CHECK(tensordata.getDataStatus().second);

  tensordata.syncHAndDData(device);

	BOOST_CHECK(tensordata.getDataStatus().first);
	BOOST_CHECK(!tensordata.getDataStatus().second);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	TensorDataCpu<float, int, int, int> tensordata(2, 3, 4);
	size_t test = 2 * 3 * 4 * sizeof(float);
	BOOST_CHECK_EQUAL(tensordata.getTensorSize(), test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	TensorDataCpu<float, int, int, int> tensordata(2, 3, 4);

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0);

  tensordata.setData(data);

	BOOST_CHECK_EQUAL(tensordata.getData()(1, 2, 3), 0);
	BOOST_CHECK(tensordata.getDataStatus().first);
	BOOST_CHECK(tensordata.getDataStatus().second);

	// Test mutability
  tensordata.getData()(0, 0, 0) = 5;

	BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 5);
}

BOOST_AUTO_TEST_CASE(syncHAndD)
{
	TensorDataCpu<float, int, int, int> tensordata(2, 3, 4);

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

  tensordata.setData(data);

	Eigen::DefaultDevice device;
  tensordata.syncHAndDData(device);

	BOOST_CHECK(tensordata.getDataStatus().first);
	BOOST_CHECK(tensordata.getDataStatus().second);

  tensordata.syncHAndDData(device);

	BOOST_CHECK(tensordata.getDataStatus().first);
	BOOST_CHECK(tensordata.getDataStatus().second);
}

BOOST_AUTO_TEST_SUITE_END()