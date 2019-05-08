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
  // Check null
	TensorDataDefaultDevice<float, 3> tensordata, tensordata_test;
	BOOST_CHECK(tensordata == tensordata_test);

  //// Check different types [TODO: fix]
  //TensorDataDefaultDevice<int, 3> tensordata_int;
  //BOOST_CHECK(tensordata_int != tensordata_test);

  // Check same dimensions
  tensordata.setDimensions(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_test.setDimensions(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  BOOST_CHECK(tensordata == tensordata_test);

  // Check different dimensions
  tensordata.setDimensions(Eigen::array<Eigen::Index, 3>({ 1, 2, 3 }));
  BOOST_CHECK(tensordata != tensordata_test);
}

BOOST_AUTO_TEST_CASE(assignmentDefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check copy
  TensorDataDefaultDevice<float, 3> tensordata(tensordata_test);
  BOOST_CHECK(tensordata == tensordata_test);
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 1);

  // Check reference sharing
  tensordata.getData()(0, 0, 0) = 2;
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(copyDefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);
  Eigen::DefaultDevice device;

  // Check copy
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensordata = tensordata_test.copy(device);
  BOOST_CHECK(tensordata->getDimensions() == tensordata_test.getDimensions());
  BOOST_CHECK(tensordata->getTensorSize() == tensordata_test.getTensorSize());
  BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata;
  static_assert(std::is_same<TensorDataDefaultDevice<float, 3>::tensorT, float>::value, "Value type is not captured.");
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

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map(1, 2, 3), 0);
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

BOOST_AUTO_TEST_CASE(typeCompatibilityDefaultDevice)
{
  { // float
    TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<float, 3> data(2, 3, 4);
    data.setConstant(0.5);
    tensordata.setData(data);
  }
  
  { // int
    TensorDataDefaultDevice<int, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<int, 3> data(2, 3, 4);
    data.setConstant(1);
    tensordata.setData(data);
  }

  { // bool
    TensorDataDefaultDevice<bool, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<bool, 3> data(2, 3, 4);
    data.setConstant(0.5);
    tensordata.setData(data);
  }
  
  { // char
    TensorDataDefaultDevice<char, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<char, 3> data(2, 3, 4);
    data.setConstant('a');
    tensordata.setData(data);
  }

  { // char*
    typedef char* varchar;
    varchar abc = new char[128];
    abc[0] = 'a'; abc[1] = 'b'; abc[2] = 'c';
    TensorDataDefaultDevice<varchar, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<varchar, 3> data(2, 3, 4);
    data.setConstant(abc);
    tensordata.setData(data);
    delete[] abc;
  }
  
  { // struct
    struct estruct { int e; };
    TensorDataDefaultDevice<estruct, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<estruct, 3> data(2, 3, 4);
    estruct e; e.e = 1;
    data.setConstant(e);
    tensordata.setData(data);
  }

  { // struct char[]
    struct charstruct { char c[128]; };
    TensorDataDefaultDevice<charstruct, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
    Eigen::Tensor<charstruct, 3> data(2, 3, 4);
    charstruct abcd; abcd.c[0] = 'a'; abcd.c[1] = 'b'; abcd.c[2] = 'c'; abcd.c[3] = 'd';
    data.setConstant(abcd);
    tensordata.setData(data);
  }

  // Known failures: std::string
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

BOOST_AUTO_TEST_CASE(assignmentCpu)
{
  TensorDataCpu<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check copy
  TensorDataCpu<float, 3> tensordata(tensordata_test);
  BOOST_CHECK(tensordata == tensordata_test);
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 1);

  // Check reference sharing
  tensordata.getData()(0, 0, 0) = 2;
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(copyCpu)
{
  TensorDataCpu<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);

  // Check copy
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 3>> tensordata = tensordata_test.copy(device);
  BOOST_CHECK(tensordata->getDimensions() == tensordata_test.getDimensions());
  BOOST_CHECK(tensordata->getTensorSize() == tensordata_test.getTensorSize());
  BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
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

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map(1, 2, 3), 0);
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

/* TensorDataGpuDevice Tests
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

BOOST_AUTO_TEST_CASE(assignmentGpu)
{
  TensorDataGpu<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check copy
  TensorDataGpu<float, 3> tensordata(tensordata_test);
  BOOST_CHECK(tensordata == tensordata_test);
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 1);

  // Check reference sharing
  tensordata.getData()(0, 0, 0) = 2;
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(copyGpu)
{
  TensorDataGpu<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);
  Eigen::GpuStreamDevice stream_device;
  Eigen::GpuDevice device(&stream_device);

  // Check copy
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 3>> tensordata = tensordata_test.copy(device);
  BOOST_CHECK(tensordata->getDimensions() == tensordata_test.getDimensions());
  BOOST_CHECK(tensordata->getTensorSize() == tensordata_test.getTensorSize());
  BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
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

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map(1, 2, 3), 0);
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