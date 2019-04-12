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
	TensorDataCpu<float> data, data_test;
	BOOST_CHECK(data == data_test);
}

#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(gettersAndSetters2)
{
	TensorDataGpu<float> data;

  Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

	data.setData(data);
	BOOST_CHECK_EQUAL(data.getData()(1, 2, 3), 0.5);
	BOOST_CHECK(data.getDataStatus().first);
	BOOST_CHECK(!data.getDataStatus().second);

	// Test mutability
	data.getData()(0, 0, 0) = 5;

	BOOST_CHECK_EQUAL(data.getData()(0, 0, 0), 5);
}

BOOST_AUTO_TEST_CASE(syncHAndD2)
{
	TensorDataGpu<float> data;

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

	data.setData(data);

	Eigen::GpuStreamDevice stream_device;
	Eigen::GpuDevice device(&stream_device);
	data.syncHAndDData(device);

	BOOST_CHECK(!data.getDataStatus().first);
	BOOST_CHECK(data.getDataStatus().second);

	data.syncHAndDData(device);

	BOOST_CHECK(data.getDataStatus().first);
	BOOST_CHECK(!data.getDataStatus().second);
}
#endif

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
	TensorDataCpu<float> data;
	size_t test = 2 * 3 * 4 * sizeof(float);
	BOOST_CHECK_EQUAL(data.getTensorSize(), test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters1)
{
	TensorDataCpu<float> data;

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0);

	data.setData(data);

	BOOST_CHECK_EQUAL(data.getData()(1, 2, 3), 0);
	BOOST_CHECK(data.getDataStatus().first);
	BOOST_CHECK(data.getDataStatus().second);

	// Test mutability
	data.getData()(0, 0, 0) = 5;

	BOOST_CHECK_EQUAL(data.getData()(0, 0, 0), 5);
}

BOOST_AUTO_TEST_CASE(syncHAndD)
{
	TensorDataCpu<float> data;

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

	data.setData(data);

	Eigen::DefaultDevice device;
	data.syncHAndDData(device);

	BOOST_CHECK(data.getDataStatus().first);
	BOOST_CHECK(data.getDataStatus().second);

	data.syncHAndDData(device);

	BOOST_CHECK(data.getDataStatus().first);
	BOOST_CHECK(data.getDataStatus().second);
}

BOOST_AUTO_TEST_SUITE_END()