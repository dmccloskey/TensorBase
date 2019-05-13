/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorData.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

/* TensorDataGpuDevice Tests
*/
void test_constructorGpu()
{
  TensorDataGpu<float, 3>* ptr = nullptr;
  TensorDataGpu<float, 3>* nullPointer = nullptr;
  ptr = new TensorDataGpu<float, 3>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorDataGpu<float, 3>* ptr = nullptr;
  ptr = new TensorDataGpu<float, 3>();
  delete ptr;
}

void test_assignmentGpu()
{
  TensorDataGpu<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check copy
  TensorDataGpu<float, 3> tensordata(tensordata_test);
  assert(tensordata == tensordata_test);
  assert(tensordata.getData()(0, 0, 0) == 1);

  // Check reference sharing
  tensordata.getData()(0, 0, 0) = 2;
  assert(tensordata.getData()(0, 0, 0) == tensordata_test.getData()(0, 0, 0));
}

void test_copyGpu()
{
  TensorDataGpu<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);
  Eigen::GpuStreamDevice stream_device;
  Eigen::GpuDevice device(&stream_device);

  // Check copy
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensordata = tensordata_test.copy(device);
  assert(tensordata->getDimensions() == tensordata_test.getDimensions());
  assert(tensordata->getTensorSize() == tensordata_test.getTensorSize());
  assert(tensordata->getDeviceName() == tensordata_test.getDeviceName());
  assert(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  assert(tensordata->getData()(0, 0, 0) != tensordata_test.getData()(0, 0, 0));
  assert(tensordata->getData()(1, 0, 0) == tensordata_test.getData()(1, 0, 0));
}

void test_selectGpu()
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
  TensorDataGpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpu<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
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
  TensorDataGpu<float, 3> tensorselect(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  tensorselect.setData();
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensorselect_ptr = std::make_shared<TensorDataGpu<float, 3>>(tensorselect);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  tensordata.select(tensorselect_ptr, std::make_shared<TensorDataGpu<int, 3>>(indices), device);
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        assert(tensorselect_ptr->getData()(i, j, k) == tensor_values_test(i, j, k));
      }
    }
  }
}

void test_sortIndicesGpu()
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values.data()[i] = i + 1;
    tensor_values.data()[i] = i;
  }
  TensorDataGpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpu<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpu<int, 3>>(indices);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test ASC
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  tensordata.sortIndices(indices_ptr, "ASC", device);
  tensordata.sort(indices_ptr, device);
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        assert(indices_ptr->getData()(i, j, k) == indices_values(i, j, k));
        assert(tensordata.getData()(i, j, k) == tensor_values(i, j, k));
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
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        assert(indices_ptr->getData()(i, j, k) == indices_values_test(i, j, k));
        assert(tensordata.getData()(i, j, k) == tensor_values_test(i, j, k));
      }
    }
  }
}

void test_gettersAndSettersGpu()
{
  TensorDataGpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  assert(tensordata.getDimensions().at(0) == 2);
  assert(tensordata.getDimensions().at(1) == 3);
  assert(tensordata.getDimensions().at(2) == 4);
  assert(tensordata.getDims() == 3);
  assert(tensordata.getDeviceName() == typeid(Eigen::GpuDevice).name());

  Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

  tensordata.setData(data);
	assert(tensordata.getData()(1, 2, 3), 0.5);
	assert(tensordata.getDataStatus().first);
	assert(!tensordata.getDataStatus().second);

	// Test mutability
  tensordata.getData()(0, 0, 0) = 5;
	assert(tensordata.getData()(0, 0, 0) == 5);

  // NOTE: not testable on the Gpu
  //// Test getDataPointer
  //Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  //assert(data_map(0, 0, 0) == 5);
  //assert(data_map(1, 2, 3) == 0);
}

void test_syncHAndDGpu()
{
	TensorDataGpu<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));

	Eigen::Tensor<float, 3> data(2, 3, 4);
	data.setConstant(0.5);

  tensordata.setData(data);

	Eigen::GpuStreamDevice stream_device;
	Eigen::GpuDevice device(&stream_device);
  tensordata.syncHAndDData(device);

	assert(!tensordata.getDataStatus().first);
	assert(tensordata.getDataStatus().second);

  tensordata.syncHAndDData(device);

	assert(tensordata.getDataStatus().first);
	assert(!tensordata.getDataStatus().second);
}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_assignmentGpu();
  test_copyGpu();
  test_selectGpu();
  test_sortIndicesGpu();
  test_gettersAndSettersGpu();
  test_syncHAndDGpu();
  return 0;
}
#endif