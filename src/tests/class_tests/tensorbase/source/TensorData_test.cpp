/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorData test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorData.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorData)

/* TensorDataDefaultDevice Tests
*/
BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
	TensorDataDefaultDevice<float, 3>* ptr = nullptr;
	TensorDataDefaultDevice<float, 3>* nullPointer = nullptr;
	ptr = new TensorDataDefaultDevice<float, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	TensorDataDefaultDevice<float, 3>* ptr = nullptr;
	ptr = new TensorDataDefaultDevice<float, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(comparisonDefaultDevice)
{
	TensorDataDefaultDevice<float, 3> tensordata, tensordata_test;
	BOOST_CHECK(tensordata == tensordata_test);
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata;
  // Check defaults
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 0);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 0);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 0);
  BOOST_CHECK_EQUAL(tensordata.getTensorSize(), 0);
  BOOST_CHECK_EQUAL(tensordata.getDims(), 3);
  BOOST_CHECK_EQUAL(tensordata.getDeviceName(), typeid(Eigen::DefaultDevice).name());

  // initialize indices
  tensordata.setDimensions(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 4);
  size_t test = 2 * 3 * 4 * sizeof(float);
  BOOST_CHECK_EQUAL(tensordata.getTensorSize(), test);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters2DefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 4);

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

BOOST_AUTO_TEST_CASE(syncHAndDDefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));

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

/* TensorDataCpu Tests
*/
BOOST_AUTO_TEST_CASE(constructorCpu)
{
  TensorDataCpu<float, 3>* ptr = nullptr;
  TensorDataCpu<float, 3>* nullPointer = nullptr;
  ptr = new TensorDataCpu<float, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorCpu)
{
  TensorDataCpu<float, 3>* ptr = nullptr;
  ptr = new TensorDataCpu<float, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters2Cpu)
{
  TensorDataCpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 4);
  BOOST_CHECK_EQUAL(tensordata.getDims(), 3);
  BOOST_CHECK_EQUAL(tensordata.getDeviceName(), typeid(Eigen::ThreadPoolDevice).name());

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

BOOST_AUTO_TEST_CASE(syncHAndDCpu)
{
  TensorDataCpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));

  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(0.5);

  tensordata.setData(data);

  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensordata.syncHAndDData(device);
  BOOST_CHECK(tensordata.getDataStatus().first);
  BOOST_CHECK(tensordata.getDataStatus().second);

  tensordata.syncHAndDData(device);
  BOOST_CHECK(tensordata.getDataStatus().first);
  BOOST_CHECK(tensordata.getDataStatus().second);
}

/* TensorDataDefaultDevice Tests
*/
#if COMPILE_WITH_CUDA
BOOST_AUTO_TEST_CASE(constructorGpu)
{
  TensorDataGpu<float, 3>* ptr = nullptr;
  TensorDataGpu<float, 3>* nullPointer = nullptr;
  ptr = new TensorDataGpu<float, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorGpu)
{
  TensorDataGpu<float, 3>* ptr = nullptr;
  ptr = new TensorDataGpu<float, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersGpu)
{
  TensorDataGpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 4);
  BOOST_CHECK_EQUAL(tensordata.getDims(), 3);
  BOOST_CHECK_EQUAL(tensordata.getDeviceName(), typeid(Eigen::GpuDevice).name());

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

BOOST_AUTO_TEST_CASE(syncHAndDGpu)
{
	TensorDataGpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));

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

BOOST_AUTO_TEST_SUITE_END()