/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorDataGpu.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

/* TensorDataGpuPrimitiveT Tests
*/
void test_constructorGpu()
{
  TensorDataGpuPrimitiveT<float, 3>* ptr = nullptr;
  TensorDataGpuPrimitiveT<float, 3>* nullPointer = nullptr;
  ptr = new TensorDataGpuPrimitiveT<float, 3>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorDataGpuPrimitiveT<float, 3>* ptr = nullptr;
  ptr = new TensorDataGpuPrimitiveT<float, 3>();
  delete ptr;
}

void test_assignmentGpu()
{
  TensorDataGpuPrimitiveT<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check copy
  TensorDataGpuPrimitiveT<float, 3> tensordata(tensordata_test);
  assert(tensordata == tensordata_test);
  assert(tensordata.getData()(0, 0, 0) == 1);

  // Check reference sharing
  tensordata.getData()(0, 0, 0) = 2;
  assert(tensordata.getData()(0, 0, 0) == tensordata_test.getData()(0, 0, 0));
}

void test_copyGpu()
{
  TensorDataGpuPrimitiveT<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
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
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 3>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices);

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
  TensorDataGpuPrimitiveT<float, 3> tensorselect(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  tensorselect.setData();
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensorselect_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(tensorselect);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  tensordata.select(tensorselect_ptr, indices_ptr, device);
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        //std::cout << "Test Select i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensorselect_ptr->getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        assert(tensorselect_ptr->getData()(i, j, k) == tensor_values_test(i, j, k));
      }
    }
  }
}

void test_sortGpu()
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    tensor_values.data()[i] = i;
  }
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test ASC
  tensordata.syncHAndDData(device);
  tensordata.sort("ASC", device);
  tensordata.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        assert(tensordata.getData()(i, j, k) == tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<float, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    tensor_values_test.data()[i] = tensor_values.size() - i - 1;
  }

  // Test DESC
  tensordata.setDataStatus(false, true);
  tensordata.sort("DESC", device);
  tensordata.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor_sorted_values2(tensordata.getDataPointer().get(), tensordata.getDimensions());
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        assert(tensor_sorted_values2(i, j, k) == tensor_values_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
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
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices);

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
        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values(i, j, k) << std::endl;
        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values(i, j, k) << std::endl;
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
  tensordata.setDataStatus(false, true);
  indices_ptr->setDataStatus(false, true);
  tensordata.sortIndices(indices_ptr, "DESC", device);
  tensordata.sort(indices_ptr, device);
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values_test(i, j, k) << std::endl;
        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        assert(indices_ptr->getData()(i, j, k) == indices_values_test(i, j, k));
        assert(tensordata.getData()(i, j, k) == tensor_values_test(i, j, k)); //FIXME
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_partitionGpu()
{
  // Make the tensor data and partition indices
  int dim_sizes = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    if (i % 2 == 0)
      indices_values.data()[i] = 1;
    else
      indices_values.data()[i] = 0;
    tensor_values.data()[i] = i;
  }
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices);

  // Make the expected partitioned data tensor
  Eigen::Tensor<float, 1> expected_values_flat(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes * dim_sizes }));
  expected_values_flat.setValues({ 0,2,4,6,8,10,12,14,16,18,20,22,24,26,25,23,21,19,17,15,13,11,9,7,5,3,1 });
  Eigen::Tensor<float, 3> expected_values = expected_values_flat.reshape(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test partition
  indices_ptr->syncHAndDData(device);
  tensordata.syncHAndDData(device);
  tensordata.partition(indices_ptr, device);
  tensordata.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        assert(tensordata.getData()(i, j, k) == expected_values(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}


void test_runLengthEncodeGpu()
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
  TensorDataGpuPrimitiveT<float, 2> tensordata(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<float, 1> unique(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique.setData();
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> uniquev_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(unique);
  TensorDataGpuPrimitiveT<int, 1> count(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count.setData();
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> count_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(count);
  TensorDataGpuPrimitiveT<int, 1> num_runs(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs.setData();
  num_runs.getData()(0) = 0;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> num_runs_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(num_runs);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test runLengthEncode for the case where the last value is the same as the previous
  tensordata.syncHAndDData(device);
  uniquev_ptr->syncHAndDData(device);
  count_ptr->syncHAndDData(device);
  num_runs_ptr->syncHAndDData(device);
  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
  tensordata.syncHAndDData(device);
  uniquev_ptr->syncHAndDData(device);
  count_ptr->syncHAndDData(device);
  num_runs_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(num_runs_ptr->getData()(0) == 3);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    assert(uniquev_ptr->getData()(i) == unique_values(i));
    assert(count_ptr->getData()(i) == count_values(i));
  }

  // Test runLengthEncode for the case where the last value is not the same as the previous
  tensor_values.setValues({ { 1, 2, 3}, { 1, 2, 3}, { 1, 2, 4} });
  tensordata.setData(tensor_values);
  unique_values.setValues({ 1, 2, 3, 4, -1, -1, -1, -1, -1 });
  count_values.setValues({ 3, 3, 2, 1, -1, -1, -1, -1, -1 });
  num_runs_values.setValues({ 4 });  
  uniquev_ptr->syncHAndDData(device);
  count_ptr->syncHAndDData(device);
  num_runs_ptr->syncHAndDData(device);
  tensordata.syncHAndDData(device);
  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
  uniquev_ptr->syncHAndDData(device);
  count_ptr->syncHAndDData(device);
  num_runs_ptr->syncHAndDData(device);
  tensordata.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(num_runs_ptr->getData()(0) == 4);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    assert(uniquev_ptr->getData()(i) == unique_values(i));
    assert(count_ptr->getData()(i) == count_values(i));
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_gettersAndSettersGpu()
{
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
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

  // Test manual data status setters
  tensordata.setDataStatus(false, true);
  assert(!tensordata.getDataStatus().first);
  assert(tensordata.getDataStatus().second);

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
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));

  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(0.5);
  tensordata.setData(data);

  Eigen::GpuStreamDevice stream_device;
  Eigen::GpuDevice device(&stream_device);

  // Test sync data status
  tensordata.syncHAndDData(device);
  assert(!tensordata.getDataStatus().first);
  assert(tensordata.getDataStatus().second);

  tensordata.syncHAndDData(device);
  assert(tensordata.getDataStatus().first);
  assert(!tensordata.getDataStatus().second);

  // Test that data was copied correctly between device and host
  assert(tensordata.getData()(0, 0, 0) == 0.5);
  assert(tensordata.getData()(1, 0, 0) == 0.5);
  assert(tensordata.getData()(0, 2, 0) == 0.5);
  assert(tensordata.getData()(0, 0, 3) == 0.5);
  assert(tensordata.getData()(1, 2, 3) == 0.5);
}

void test_selectStringGpu()
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<TensorArray8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  std::cout << "tensor_values(i, j, k)" << std::endl;
  int iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        Eigen::Tensor<char, 1> tmp(8); 
        sprintf(tmp.data(), "%d", i);
        tensor_values(i, j, k) = tmp;

        for (int z = 0; z < 8; ++z) {
          std::cout << "i(" << i << ")j(" << j << ")k(" << k << "): " << tensor_values(i, j, k).at(z) << std::endl;
        }
        std::cout << std::endl;

        if (i == dim_sizes - 1 || j == dim_sizes - 1 || k == dim_sizes - 1)
          indices_values(i, j, k) = 0;
        else
          indices_values(i, j, k) = 1;
        ++iter;

      }
    }
  }
  TensorDataGpuClassT<TensorArray8<char>, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuClassT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorDataGpuClassT<int, 3>> indices_ptr = std::make_shared<TensorDataGpuClassT<int, 3>>(indices);

  // Make the expected tensor
  int dim_sizes_select = 2;
  Eigen::Tensor<TensorArray8<char>, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        if (i < dim_sizes - 1 && j < dim_sizes - 1 && k < dim_sizes - 1) {
          Eigen::Tensor<char, 1> tmp(8);
          sprintf(tmp.data(), "%d", iter);
          tensor_values_test(i, j, k) = tmp;
        }
        ++iter;
      }
    }
  }
  TensorDataGpuClassT<TensorArray8<char>, 3> tensorselect(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  tensorselect.setData();
  std::shared_ptr<TensorData<TensorArray8<char>, Eigen::GpuDevice, 3>> tensorselect_ptr = std::make_shared<TensorDataGpuClassT<TensorArray8<char>, 3>>(tensorselect);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  tensordata.select(tensorselect_ptr, indices_ptr, device);
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        //std::cout << "Test Select i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensorselect_ptr->getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        assert(tensorselect_ptr->getData()(i, j, k) == tensor_values_test(i, j, k));
      }
    }
  }
}

//void test_sortStringGpu()
//{
//  // Make the tensor data and select indices
//  int dim_sizes = 3;
//  Eigen::Tensor<TensorArray8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  for (int i = 0; i < tensor_values.size(); ++i) {
//    Eigen::Tensor<char, 1> tmp(8);
//    sprintf(tmp.data(), "%d", i);
//    tensor_values.data()[i] = tmp;
//  }
//  TensorDataGpuClassT<TensorArray8<char>, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  tensordata.setData(tensor_values);
//
//  // Initialize the device
//  cudaStream_t stream;
//  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
//  Eigen::GpuStreamDevice stream_device(&stream, 0);
//  Eigen::GpuDevice device(&stream_device);
//
//  // Test ASC
//  tensordata.syncHAndDData(device);
//  tensordata.sort("ASC", device);
//  tensordata.syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  for (int i = 0; i < dim_sizes; ++i) {
//    for (int j = 0; j < dim_sizes; ++j) {
//      for (int k = 0; k < dim_sizes; ++k) {
//        assert(tensordata.getData()(i, j, k) == tensor_values(i, j, k));
//      }
//    }
//  }
//
//  // Make the expected indices and values
//  Eigen::Tensor<TensorArray8<char>, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  for (int i = 0; i < tensor_values.size(); ++i) {
//    Eigen::Tensor<char, 1> tmp(8);
//    sprintf(tmp.data(), "%d", tensor_values.size() - i - 1);
//    tensor_values_test.data()[i] = tmp;
//  }
//
//  // Test DESC
//  tensordata.setDataStatus(false, true);
//  tensordata.sort("DESC", device);
//  tensordata.syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 3>> tensor_sorted_values2(tensordata.getDataPointer().get(), tensordata.getDimensions());
//  for (int i = 0; i < dim_sizes; ++i) {
//    for (int j = 0; j < dim_sizes; ++j) {
//      for (int k = 0; k < dim_sizes; ++k) {
//        assert(tensor_sorted_values2(i, j, k) == tensor_values_test(i, j, k));
//      }
//    }
//  }
//  assert(cudaStreamDestroy(stream) == cudaSuccess);
//}
//
//void test_sortIndicesStringGpu()
//{
//  // Make the tensor data and select indices
//  int dim_sizes = 3;
//  Eigen::Tensor<TensorArray8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  for (int i = 0; i < tensor_values.size(); ++i) {
//    indices_values.data()[i] = i + 1;
//    Eigen::Tensor<char, 1> tmp(8);
//    sprintf(tmp.data(), "%d", i);
//    tensor_values.data()[i] = tmp;
//  }
//  TensorDataGpuClassT<TensorArray8<char>, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  tensordata.setData(tensor_values);
//  TensorDataGpuClassT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  indices.setData(indices_values);
//  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpuClassT<int, 3>>(indices);
//
//  // Initialize the device
//  cudaStream_t stream;
//  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
//  Eigen::GpuStreamDevice stream_device(&stream, 0);
//  Eigen::GpuDevice device(&stream_device);
//
//  // Test ASC
//  tensordata.syncHAndDData(device);
//  indices_ptr->syncHAndDData(device);
//  tensordata.sortIndices(indices_ptr, "ASC", device);
//  tensordata.sort(indices_ptr, device);
//  tensordata.syncHAndDData(device);
//  indices_ptr->syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  for (int i = 0; i < dim_sizes; ++i) {
//    for (int j = 0; j < dim_sizes; ++j) {
//      for (int k = 0; k < dim_sizes; ++k) {
//        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values(i, j, k) << std::endl;
//        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values(i, j, k) << std::endl;
//        assert(indices_ptr->getData()(i, j, k) == indices_values(i, j, k));
//        assert(tensordata.getData()(i, j, k) == tensor_values(i, j, k));
//      }
//    }
//  }
//
//  // Make the expected indices and values
//  Eigen::Tensor<int, 3> indices_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  Eigen::Tensor<TensorArray8<char>, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  for (int i = 0; i < tensor_values.size(); ++i) {
//    indices_values_test.data()[i] = tensor_values.size() - i;
//
//    Eigen::Tensor<char, 1> tmp(8);
//    sprintf(tmp.data(), "%d", tensor_values.size() - i - 1);
//    tensor_values_test.data()[i] = tmp;
//  }
//
//  // Test DESC
//  tensordata.setDataStatus(false, true);
//  indices_ptr->setDataStatus(false, true);
//  tensordata.sortIndices(indices_ptr, "DESC", device);
//  tensordata.sort(indices_ptr, device);
//  tensordata.syncHAndDData(device);
//  indices_ptr->syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  for (int i = 0; i < dim_sizes; ++i) {
//    for (int j = 0; j < dim_sizes; ++j) {
//      for (int k = 0; k < dim_sizes; ++k) {
//        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values_test(i, j, k) << std::endl;
//        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
//        assert(indices_ptr->getData()(i, j, k) == indices_values_test(i, j, k));
//        assert(tensordata.getData()(i, j, k) == tensor_values_test(i, j, k)); //FIXME
//      }
//    }
//  }
//  assert(cudaStreamDestroy(stream) == cudaSuccess);
//}
//
//void test_partitionStringGpu()
//{
//  // Make the tensor data and partition indices
//  int dim_sizes = 3;
//  Eigen::Tensor<TensorArray8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  for (int i = 0; i < tensor_values.size(); ++i) {
//    if (i % 2 == 0)
//      indices_values.data()[i] = 1;
//    else
//      indices_values.data()[i] = 0;
//    Eigen::Tensor<char, 1> tmp(8);
//    sprintf(tmp.data(), "%d", i);
//    tensor_values.data()[i] = tmp;
//  }
//  TensorDataGpuClassT<TensorArray8<char>, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  tensordata.setData(tensor_values);
//  TensorDataGpuClassT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//  indices.setData(indices_values);
//  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpuClassT<int, 3>>(indices);
//
//  // Make the expected partitioned data tensor
//  Eigen::Tensor<TensorArray8<char>, 1> expected_values_flat(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes * dim_sizes }));
//  expected_values_flat.setValues({ 0,2,4,6,8,10,12,14,16,18,20,22,24,26,25,23,21,19,17,15,13,11,9,7,5,3,1 });
//  Eigen::Tensor<TensorArray8<char>, 3> expected_values = expected_values_flat.reshape(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
//
//  // Initialize the device
//  cudaStream_t stream;
//  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
//  Eigen::GpuStreamDevice stream_device(&stream, 0);
//  Eigen::GpuDevice device(&stream_device);
//
//  // Test partition
//  indices_ptr->syncHAndDData(device);
//  tensordata.syncHAndDData(device);
//  tensordata.partition(indices_ptr, device);
//  tensordata.syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  for (int i = 0; i < dim_sizes; ++i) {
//    for (int j = 0; j < dim_sizes; ++j) {
//      for (int k = 0; k < dim_sizes; ++k) {
//        assert(tensordata.getData()(i, j, k) == expected_values(i, j, k));
//      }
//    }
//  }
//  assert(cudaStreamDestroy(stream) == cudaSuccess);
//}
//
//
//void test_runLengthEncodeStringGpu()
//{
//  // Make the tensor data and expected data values
//  int dim_sizes = 3;
//  Eigen::Tensor<TensorArray8<char>, 2> tensor_values(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
//  tensor_values.setValues({ { '0', '1', '2'}, { '0', '1', '2'}, { '0', '1', '2'} });
//  Eigen::Tensor<TensorArray8<char>, 1> unique_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
//  unique_values.setValues({ '0', '1', '2', '\0', '\0', '\0', '\0', '\0', '\0' });
//  Eigen::Tensor<int, 1> count_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
//  count_values.setValues({ 3, 3, 3, -1, -1, -1, -1, -1, -1 });
//  Eigen::Tensor<int, 1> num_runs_values(Eigen::array<Eigen::Index, 1>({ 1 }));
//  num_runs_values.setValues({ 3 });
//
//  // Make the tensor data and test data pointers
//  TensorDataGpuClassT<TensorArray8<char>, 2> tensordata(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
//  tensordata.setData(tensor_values);
//  TensorDataGpuClassT<TensorArray8<char>, 1> unique(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
//  unique.setData();
//  std::shared_ptr<TensorData<TensorArray8<char>, Eigen::GpuDevice, 1>> uniquev_ptr = std::make_shared<TensorDataGpuClassT<TensorArray8<char>, 1>>(unique);
//  TensorDataGpuClassT<int, 1> count(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
//  count.setData();
//  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> count_ptr = std::make_shared<TensorDataGpuClassT<int, 1>>(count);
//  TensorDataGpuClassT<int, 1> num_runs(Eigen::array<Eigen::Index, 1>({ 1 }));
//  num_runs.setData();
//  num_runs.getData()(0) = 0;
//  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> num_runs_ptr = std::make_shared<TensorDataGpuClassT<int, 1>>(num_runs);
//
//  // Initialize the device
//  cudaStream_t stream;
//  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
//  Eigen::GpuStreamDevice stream_device(&stream, 0);
//  Eigen::GpuDevice device(&stream_device);
//
//  // Test runLengthEncode for the case where the last value is the same as the previous
//  tensordata.syncHAndDData(device);
//  uniquev_ptr->syncHAndDData(device);
//  count_ptr->syncHAndDData(device);
//  num_runs_ptr->syncHAndDData(device);
//  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
//  tensordata.syncHAndDData(device);
//  uniquev_ptr->syncHAndDData(device);
//  count_ptr->syncHAndDData(device);
//  num_runs_ptr->syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  assert(num_runs_ptr->getData()(0) == 3);
//  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
//    assert(uniquev_ptr->getData()(i) == unique_values(i));
//    assert(count_ptr->getData()(i) == count_values(i));
//  }
//
//  // Test runLengthEncode for the case where the last value is not the same as the previous
//  tensor_values.setValues({ { '1', '2', '3'}, { '1', '2', '3'}, { '1', '2', '4'} });
//  tensordata.setData(tensor_values);
//  unique_values.setValues({ '1', '2', '3', '4', '\0', '\0', '\0', '\0', '\0' });
//  count_values.setValues({ 3, 3, 2, 1, -1, -1, -1, -1, -1 });
//  num_runs_values.setValues({ 4 });
//  uniquev_ptr->syncHAndDData(device);
//  count_ptr->syncHAndDData(device);
//  num_runs_ptr->syncHAndDData(device);
//  tensordata.syncHAndDData(device);
//  tensordata.runLengthEncode(uniquev_ptr, count_ptr, num_runs_ptr, device);
//  uniquev_ptr->syncHAndDData(device);
//  count_ptr->syncHAndDData(device);
//  num_runs_ptr->syncHAndDData(device);
//  tensordata.syncHAndDData(device);
//  assert(cudaStreamSynchronize(stream) == cudaSuccess);
//  assert(num_runs_ptr->getData()(0) == 4);
//  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
//    assert(uniquev_ptr->getData()(i) == unique_values(i));
//    assert(count_ptr->getData()(i) == count_values(i));
//  }
//  assert(cudaStreamDestroy(stream) == cudaSuccess);
//}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_gettersAndSettersGpu();
  test_syncHAndDGpu();
  test_assignmentGpu();
  test_copyGpu();
  test_selectGpu();
  test_sortIndicesGpu();
  test_partitionGpu();
  test_runLengthEncodeGpu();
  test_selectStringGpu();
  //test_sortIndicesStringGpu();
  //test_partitionStringGpu();
  //test_runLengthEncodeStringGpu();
  return 0;
}
#endif