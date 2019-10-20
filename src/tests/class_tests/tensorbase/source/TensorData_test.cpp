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
  BOOST_CHECK(tensordata->getTensorBytes() == tensordata_test.getTensorBytes());
  BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
}

BOOST_AUTO_TEST_CASE(selectDefaultDevice)
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  int iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        tensor_values(i, j, k) = float(iter);
        if (i == dim_sizes - 1 || j == dim_sizes - 1 || k == dim_sizes - 1)
          indices_values(i, j, k) = 0;
        else
          indices_values(i, j, k) = 1;
        ++iter;
      }
    }
  }
  TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataDefaultDevice<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);

  // Make the expected tensor
  int dim_sizes_select = 2;
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        if (i < dim_sizes - 1 && j < dim_sizes - 1 && k < dim_sizes - 1)
          tensor_values_test(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  TensorDataDefaultDevice<float, 3> tensorselect(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  tensorselect.setData();
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensorselect_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(tensorselect);

  // Test
  Eigen::DefaultDevice device;
  tensordata.select(tensorselect_ptr, std::make_shared<TensorDataDefaultDevice<int, 3>>(indices), device);
  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        BOOST_CHECK_CLOSE(tensorselect_ptr->getData()(i, j, k), tensor_values_test(i, j, k), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(sortDefaultDevice)
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    tensor_values.data()[i] = i;
  }
  TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);

  Eigen::DefaultDevice device;
  // Test ASC
  tensordata.sort("ASC", device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    tensor_values_test.data()[i] = tensor_values.size() - i - 1;
  }

  // Test DESC
  tensordata.sort("DESC", device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(sortIndicesDefaultDevice)
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values.data()[i] = i + 1;
    tensor_values.data()[i] = i;
  }
  TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataDefaultDevice<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_ptr = std::make_shared<TensorDataDefaultDevice<int, 3>>(indices);

  Eigen::DefaultDevice device;
  // Test ASC
  tensordata.sortIndices(indices_ptr, "ASC", device);
  tensordata.sort(indices_ptr, device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(indices_ptr->getData()(i, j, k), indices_values(i,j,k));
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<int, 3> indices_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values_test.data()[i] = tensor_values.size() - i;
    tensor_values_test.data()[i] = tensor_values.size() - i - 1;
  }

  // Test DESC
  tensordata.sortIndices(indices_ptr, "DESC", device);
  tensordata.sort(indices_ptr, device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(indices_ptr->getData()(i, j, k), indices_values_test(i, j, k));
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(partitionDefaultDevice)
{
  // Make the tensor data and partition indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    if (i%2==0)
      indices_values.data()[i] = i + 1;
    else
      indices_values.data()[i] = 0;
    tensor_values.data()[i] = i;
  }
  TensorDataDefaultDevice<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataDefaultDevice<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_ptr = std::make_shared<TensorDataDefaultDevice<int, 3>>(indices);

  // Make the expected partitioned data tensor
  Eigen::Tensor<float, 1> expected_values_flat(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes * dim_sizes }));
  expected_values_flat.setValues({0,2,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25});
  Eigen::Tensor<float, 3> expected_values = expected_values_flat.reshape(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));

  Eigen::DefaultDevice device;
  // Test partition
  tensordata.partition(indices_ptr, device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), expected_values(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(runLengthEncodeDefaultDevice)
{
  // Make the tensor data and expected data values
  int dim_sizes = 3;
  Eigen::Tensor<float, 2> tensor_values(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensor_values.setValues({ { 1, 2, 3}, { 1, 2, 3}, { 1, 2, 3} });
  Eigen::Tensor<float, 1> unique_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique_values.setValues({ 1, 2, 3, -1, -1, -1, -1, -1, -1 });
  Eigen::Tensor<int, 1> count_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count_values.setValues({ 3, 3, 3, -1, -1, -1, -1, -1, -1 });
  Eigen::Tensor<int, 1> num_runs_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs_values.setValues({ 3 });

  // Make the tensor data
  TensorDataDefaultDevice<float, 2> tensordata(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);

  // Make the data pointers
  TensorDataDefaultDevice<float, 1> unique(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique.setData();
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> uniquev_ptr = std::make_shared<TensorDataDefaultDevice<float, 1>>(unique);
  TensorDataDefaultDevice<int, 1> count(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count.setData();
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> count_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(count);
  TensorDataDefaultDevice<int, 1> num_runs(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs.setData();
  num_runs.getData()(0) = 0;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> num_runs_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(num_runs);

  Eigen::DefaultDevice device;
  // Test runLengthEncode for the case where the last value is the same as the previous
  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
  BOOST_CHECK_EQUAL(num_runs_ptr->getData()(0), 3);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    BOOST_CHECK_EQUAL(uniquev_ptr->getData()(i), unique_values(i));
    BOOST_CHECK_EQUAL(count_ptr->getData()(i), count_values(i));
  }

  // Test runLengthEncode for the case where the last value is not the same as the previous
  tensor_values.setValues({ { 1, 2, 3}, { 1, 2, 3}, { 1, 2, 4} });
  tensordata.setData(tensor_values);
  unique_values.setValues({ 1, 2, 3, 4, -1, -1, -1, -1, -1 });
  count_values.setValues({ 3, 3, 2, 1, -1, -1, -1, -1, -1 });
  num_runs_values.setValues({ 4 });
  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
  BOOST_CHECK_EQUAL(num_runs_ptr->getData()(0), 4);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    BOOST_CHECK_EQUAL(uniquev_ptr->getData()(i), unique_values(i));
    BOOST_CHECK_EQUAL(count_ptr->getData()(i), count_values(i));
  }
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorDataDefaultDevice<float, 3> tensordata;
  static_assert(std::is_same<TensorDataDefaultDevice<float, 3>::tensorT, float>::value, "Value type is not captured.");
  // Check defaults
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 0);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 0);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 0);
  BOOST_CHECK_EQUAL(tensordata.getTensorBytes(), 0);
  BOOST_CHECK_EQUAL(tensordata.getTensorSize(), 0);
  BOOST_CHECK_EQUAL(tensordata.getDims(), 3);
  BOOST_CHECK_EQUAL(tensordata.getDeviceName(), typeid(Eigen::DefaultDevice).name());

  // initialize indices
  tensordata.setDimensions(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensordata.getDimensions().at(2), 4);
  size_t test = 2 * 3 * 4 * sizeof(float);
  BOOST_CHECK_EQUAL(tensordata.getTensorBytes(), test);
  BOOST_CHECK_EQUAL(tensordata.getTensorSize(), 2 * 3 * 4);
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

  // Test manual data status setters
  tensordata.setDataStatus(false, true);
  assert(!tensordata.getDataStatus().first);
  assert(tensordata.getDataStatus().second);

  // Test mutability
  tensordata.getData()(0, 0, 0) = 5;
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 5);

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map(1, 2, 3), 0);

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map_h(tensordata.getHDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map_h(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map_h(1, 2, 3), 0);
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

BOOST_AUTO_TEST_CASE(convertFromStringToTensorTDefaultDevice)
{
  Eigen::Tensor<std::string, 3> tensor_string(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensor_string.setConstant("1.001");
  Eigen::DefaultDevice device;

  // Test float
  TensorDataDefaultDevice<float, 3> tensordata_f(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_f.setData();
  tensordata_f.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_CLOSE(tensordata_f.getData()(0, 0, 0), 1.001, 1e-3);
  BOOST_CHECK_CLOSE(tensordata_f.getData()(1, 2, 3), 1.001, 1e-3);

  // Test double
  TensorDataDefaultDevice<double, 3> tensordata_d(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_d.setData();
  tensordata_d.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_CLOSE(tensordata_d.getData()(0, 0, 0), 1.001, 1e-3);
  BOOST_CHECK_CLOSE(tensordata_d.getData()(1, 2, 3), 1.001, 1e-3);

  // Test int
  TensorDataDefaultDevice<int, 3> tensordata_i(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_i.setData();
  tensordata_i.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_EQUAL(tensordata_i.getData()(0, 0, 0), 1);
  BOOST_CHECK_EQUAL(tensordata_i.getData()(1, 2, 3), 1);

  // Test int
  TensorDataDefaultDevice<char, 3> tensordata_c(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_c.setData();
  tensordata_c.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_EQUAL(tensordata_c.getData()(0, 0, 0), '1');
  BOOST_CHECK_EQUAL(tensordata_c.getData()(1, 2, 3), '1');

  // Test array
  TensorDataDefaultDevice<TensorArray8<char>, 3> tensordata_a8(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_a8.setData();
  tensordata_a8.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_EQUAL(tensordata_a8.getData()(0, 0, 0), TensorArray8<char>("1.001"));
  BOOST_CHECK_EQUAL(tensordata_a8.getData()(1, 2, 3), TensorArray8<char>("1.001"));
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
  BOOST_CHECK(tensordata->getTensorBytes() == tensordata_test.getTensorBytes());
  BOOST_CHECK(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  BOOST_CHECK_EQUAL(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  BOOST_CHECK_NE(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));
  BOOST_CHECK_EQUAL(tensordata->getData()(1, 0, 0), tensordata_test.getData()(1, 0, 0));
}

BOOST_AUTO_TEST_CASE(selectCpu)
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  int iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        tensor_values(i, j, k) = float(iter);
        if (i == dim_sizes - 1 || j == dim_sizes - 1 || k == dim_sizes - 1)
          indices_values(i, j, k) = 0;
        else
          indices_values(i, j, k) = 1;
        ++iter;
      }
    }
  }
  TensorDataCpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataCpu<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);

  // Make the expected tensor
  int dim_sizes_select = 2;
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        if (i < dim_sizes - 1 && j < dim_sizes - 1 && k < dim_sizes - 1)
          tensor_values_test(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  TensorDataCpu<float, 3> tensorselect(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  tensorselect.setData();
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 3>> tensorselect_ptr = std::make_shared<TensorDataCpu<float, 3>>(tensorselect);

  // Test
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensordata.select(tensorselect_ptr, std::make_shared<TensorDataCpu<int, 3>>(indices), device);
  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        BOOST_CHECK_CLOSE(tensorselect_ptr->getData()(i, j, k), tensor_values_test(i, j, k), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(sortCpu)
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    tensor_values.data()[i] = i;
  }
  TensorDataCpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);

  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  // Test ASC
  tensordata.sort("ASC", device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    tensor_values_test.data()[i] = tensor_values.size() - i - 1;
  }

  // Test DESC
  tensordata.sort("DESC", device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(sortIndicesCpu)
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values.data()[i] = i + 1;
    tensor_values.data()[i] = i;
  }
  TensorDataCpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataCpu<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> indices_ptr = std::make_shared<TensorDataCpu<int, 3>>(indices);

  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  // Test ASC
  tensordata.sortIndices(indices_ptr, "ASC", device);
  tensordata.sort(indices_ptr, device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(indices_ptr->getData()(i, j, k), indices_values(i, j, k));
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<int, 3> indices_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values_test.data()[i] = tensor_values.size() - i;
    tensor_values_test.data()[i] = tensor_values.size() - i - 1;
  }

  // Test DESC
  tensordata.sortIndices(indices_ptr, "DESC", device);
  tensordata.sort(indices_ptr, device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(indices_ptr->getData()(i, j, k), indices_values_test(i, j, k));
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), tensor_values_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(partitionCpu)
{
  // Make the tensor data and partition indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    if (i % 2 == 0)
      indices_values.data()[i] = i + 1;
    else
      indices_values.data()[i] = 0;
    tensor_values.data()[i] = i;
  }
  TensorDataCpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataCpu<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 3>> indices_ptr = std::make_shared<TensorDataCpu<int, 3>>(indices);

  // Make the expected partitioned data tensor
  Eigen::Tensor<float, 1> expected_values_flat(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes * dim_sizes }));
	expected_values_flat.setValues({ 0,2,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25 });
  Eigen::Tensor<float, 3> expected_values = expected_values_flat.reshape(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));

  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  // Test partition
  tensordata.partition(indices_ptr, device);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        BOOST_CHECK_EQUAL(tensordata.getData()(i, j, k), expected_values(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(runLengthEncodeCpu)
{
  // Make the tensor data and expected data values
  int dim_sizes = 3;
  Eigen::Tensor<float, 2> tensor_values(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensor_values.setValues({ { 0, 1, 2}, { 0, 1, 2}, { 0, 1, 2} });
  Eigen::Tensor<float, 1> unique_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique_values.setValues({ 0, 1, 2, -1, -1, -1, -1, -1, -1 });
  Eigen::Tensor<int, 1> count_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count_values.setValues({ 3, 3, 3, -1, -1, -1, -1, -1, -1 });
  Eigen::Tensor<int, 1> num_runs_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs_values.setValues({ 3 });

  // Make the tensor data and test data pointers
  TensorDataCpu<float, 2> tensordata(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataCpu<float, 1> unique(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique.setData();
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 1>> uniquev_ptr = std::make_shared<TensorDataCpu<float, 1>>(unique);
  TensorDataCpu<int, 1> count(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count.setData();
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> count_ptr = std::make_shared<TensorDataCpu<int, 1>>(count);
  TensorDataCpu<int, 1> num_runs(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs.setData();
  num_runs.getData()(0) = 0;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> num_runs_ptr = std::make_shared<TensorDataCpu<int, 1>>(num_runs);

  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  // Test runLengthEncode for the case where the last value is the same as the previous
  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
  BOOST_CHECK_EQUAL(num_runs_ptr->getData()(0), 3);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    BOOST_CHECK_EQUAL(uniquev_ptr->getData()(i), unique_values(i));
    BOOST_CHECK_EQUAL(count_ptr->getData()(i), count_values(i));
  }

  // Test runLengthEncode for the case where the last value is not the same as the previous
  tensor_values.setValues({ { 1, 2, 3}, { 1, 2, 3}, { 1, 2, 4} });
  tensordata.setData(tensor_values);
  unique_values.setValues({ 1, 2, 3, 4, -1, -1, -1, -1, -1 });
  count_values.setValues({ 3, 3, 2, 1, -1, -1, -1, -1, -1 });
  num_runs_values.setValues({ 4 });
  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
  BOOST_CHECK_EQUAL(num_runs_ptr->getData()(0), 4);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    BOOST_CHECK_EQUAL(uniquev_ptr->getData()(i), unique_values(i));
    BOOST_CHECK_EQUAL(count_ptr->getData()(i), count_values(i));
  }
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

  // Test manual data status setters
  tensordata.setDataStatus(false, true);
  assert(!tensordata.getDataStatus().first);
  assert(tensordata.getDataStatus().second);

  // Test mutability
  tensordata.getData()(0, 0, 0) = 5;
  BOOST_CHECK_EQUAL(tensordata.getData()(0, 0, 0), 5);

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map(1, 2, 3), 0);

  // Test getDataPointer
  Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map_h(tensordata.getHDataPointer().get(), 2, 3, 4);
  BOOST_CHECK_EQUAL(data_map_h(0, 0, 0), 5);
  BOOST_CHECK_EQUAL(data_map_h(1, 2, 3), 0);
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

BOOST_AUTO_TEST_CASE(convertFromStringToTensorTCpu)
{
  Eigen::Tensor<std::string, 3> tensor_string(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensor_string.setConstant("1.001");
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);

  // Test float
  TensorDataCpu<float, 3> tensordata_f(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_f.setData();
  tensordata_f.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_CLOSE(tensordata_f.getData()(0, 0, 0), 1.001, 1e-3);
  BOOST_CHECK_CLOSE(tensordata_f.getData()(1, 2, 3), 1.001, 1e-3);

  // Test double
  TensorDataCpu<double, 3> tensordata_d(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_d.setData();
  tensordata_d.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_CLOSE(tensordata_d.getData()(0, 0, 0), 1.001, 1e-3);
  BOOST_CHECK_CLOSE(tensordata_d.getData()(1, 2, 3), 1.001, 1e-3);

  // Test int
  TensorDataCpu<int, 3> tensordata_i(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_i.setData();
  tensordata_i.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_EQUAL(tensordata_i.getData()(0, 0, 0), 1);
  BOOST_CHECK_EQUAL(tensordata_i.getData()(1, 2, 3), 1);

  // Test int
  TensorDataCpu<char, 3> tensordata_c(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_c.setData();
  tensordata_c.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_EQUAL(tensordata_c.getData()(0, 0, 0), '1');
  BOOST_CHECK_EQUAL(tensordata_c.getData()(1, 2, 3), '1');

  // Test array
  TensorDataCpu<TensorArray8<char>, 3> tensordata_a8(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_a8.setData();
  tensordata_a8.convertFromStringToTensorT(tensor_string, device);
  BOOST_CHECK_EQUAL(tensordata_a8.getData()(0, 0, 0), TensorArray8<char>("1.001"));
  BOOST_CHECK_EQUAL(tensordata_a8.getData()(1, 2, 3), TensorArray8<char>("1.001"));
}

BOOST_AUTO_TEST_SUITE_END()