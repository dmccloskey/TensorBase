/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorDataGpu.h>

#include <iostream>

using namespace TensorBase;
using namespace std;

/* TensorDataGpuPrimitiveT Tests
*/
void test_constructorGpuPrimitiveT()
{
  TensorDataGpuPrimitiveT<float, 3>* ptr = nullptr;
  TensorDataGpuPrimitiveT<float, 3>* nullPointer = nullptr;
  ptr = new TensorDataGpuPrimitiveT<float, 3>();
  gpuCheckNotEqual(ptr, nullPointer);
  delete ptr;
}

void test_destructorGpuPrimitiveT()
{
  TensorDataGpuPrimitiveT<float, 3>* ptr = nullptr;
  ptr = new TensorDataGpuPrimitiveT<float, 3>();
  delete ptr;
}

void test_assignmentGpuPrimitiveT()
{
  TensorDataGpuPrimitiveT<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check copyToHost
  TensorDataGpuPrimitiveT<float, 3> tensordata(tensordata_test);
  gpuCheckEqualNoLhsRhsPrint(tensordata, tensordata_test);
  gpuCheckEqual(tensordata.getData()(0, 0, 0),1);

  // Check reference sharing
  tensordata.getData()(0, 0, 0) = 2;
  gpuCheckEqual(tensordata.getData()(0, 0, 0),tensordata_test.getData()(0, 0, 0));
}

void test_syncDataGpuPrimitiveT()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the dummy data
  TensorDataGpuPrimitiveT<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);

  // Check syncHData (no transfer)
  gpuCheck(tensordata_test.syncHData(device));
  gpuCheck(tensordata_test.getDataStatus().first);
  gpuCheck(!tensordata_test.getDataStatus().second);

  // Check syncDData (transfer)
  gpuCheck(tensordata_test.syncDData(device));
  gpuCheck(!tensordata_test.getDataStatus().first);
  gpuCheck(tensordata_test.getDataStatus().second);

  // Check syncDData (no transfer)
  gpuCheck(tensordata_test.syncDData(device));
  gpuCheck(!tensordata_test.getDataStatus().first);
  gpuCheck(tensordata_test.getDataStatus().second);

  // Check syncHData (transfer)
  gpuCheck(tensordata_test.syncHData(device));
  gpuCheck(tensordata_test.getDataStatus().first);
  gpuCheck(!tensordata_test.getDataStatus().second);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_copyGpuPrimitiveT()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Make the test data
  TensorDataGpuPrimitiveT<float, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }), false, TensorDataGpuPinnedFlags::HostAllocPortable);
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1);
  tensordata_test.setData(data);
  tensordata_test.syncDData(device);

  // Check copyToHost
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensordata = tensordata_test.copyToHost(device);
  gpuCheckEqualNoLhsRhsPrint(tensordata->getDimensions(),tensordata_test.getDimensions());
  gpuCheckEqual(tensordata->getTensorSize(),tensordata_test.getTensorSize());
  gpuCheckEqual(tensordata->getDeviceName(),tensordata_test.getDeviceName());
  gpuCheckEqualNoLhsRhsPrint(tensordata->getPinnedMemory(),tensordata_test.getPinnedMemory());
  gpuCheckEqualNoLhsRhsPrint(tensordata->getPinnedFlag(),tensordata_test.getPinnedFlag());
  gpuCheckEqual(tensordata->getData()(0, 0, 0),1);
  tensordata->setDataStatus(false, true);
  tensordata->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckNotEqual(tensordata->getData()(0, 0, 0), 1);

  // Check reference change
  tensordata->getData()(0, 0, 0) = 2;
  gpuCheckNotEqual(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));

  // Check copyToDevice
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensordata2 = tensordata_test.copyToDevice(device);
  gpuCheckEqualNoLhsRhsPrint(tensordata2->getDimensions(),tensordata_test.getDimensions());
  gpuCheckEqual(tensordata2->getTensorSize(),tensordata_test.getTensorSize());
  gpuCheckEqual(tensordata2->getDeviceName(),tensordata_test.getDeviceName());
  gpuCheckEqualNoLhsRhsPrint(tensordata2->getPinnedMemory(),tensordata_test.getPinnedMemory());
  gpuCheckEqualNoLhsRhsPrint(tensordata2->getPinnedFlag(),tensordata_test.getPinnedFlag());
  gpuCheckNotEqual(tensordata2->getData()(0, 0, 0), 1);
  tensordata2->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensordata2->getData()(0, 0, 0),1);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_selectGpuPrimitiveT()
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
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  tensordata.syncHAndDData(device);
  tensorselect_ptr->setDataStatus(false, true);
  indices_ptr->syncHAndDData(device);
  tensordata.select(tensorselect_ptr, indices_ptr, device);
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuErrchk(cudaStreamDestroy(stream));

  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        //std::cout << "Test Select i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensorselect_ptr->getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        gpuCheckEqual(tensorselect_ptr->getData()(i, j, k),tensor_values_test(i, j, k));
      }
    }
  }
}

void test_sortGpuPrimitiveT()
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
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test ASC
  tensordata.syncHAndDData(device);
  tensordata.sort("ASC", device);
  tensordata.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values(i, j, k));
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
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_sortIndicesGpuPrimitiveT()
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
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test ASC
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  tensordata.sortIndices(indices_ptr, "ASC", device);
  tensordata.sort(indices_ptr, device);
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values(i, j, k) << std::endl;
        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values(i, j, k) << std::endl;
        gpuCheckEqual(indices_ptr->getData()(i, j, k),indices_values(i, j, k)); 
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values(i, j, k));
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
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values_test(i, j, k) << std::endl;
        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        gpuCheckEqual(indices_ptr->getData()(i, j, k),indices_values_test(i, j, k));
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values_test(i, j, k)); //FIXME
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_partitionGpuPrimitiveT()
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
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test partition
  indices_ptr->syncHAndDData(device);
  tensordata.syncHAndDData(device);
  tensordata.partition(indices_ptr, device);
  tensordata.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        gpuCheckEqual(tensordata.getData()(i, j, k),expected_values(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_runLengthEncodeGpuPrimitiveT()
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
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
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
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(num_runs_ptr->getData()(0),3);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    gpuCheckEqual(uniquev_ptr->getData()(i),unique_values(i));
    gpuCheckEqual(count_ptr->getData()(i),count_values(i));
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
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(num_runs_ptr->getData()(0),4);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    gpuCheckEqual(uniquev_ptr->getData()(i),unique_values(i));
    gpuCheckEqual(count_ptr->getData()(i),count_values(i));
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_histogramGpuPrimitiveT()
{
	// Initialize the device
	cudaStream_t stream;
	gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	// Make the tensor data and select indices
	int dim_sizes = 3;
	Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
	for (int i = 0; i < dim_sizes; ++i) {
		for (int j = 0; j < dim_sizes; ++j) {
			for (int k = 0; k < dim_sizes; ++k) {
				float value = i + j * dim_sizes + k * dim_sizes * dim_sizes;
				tensor_values(i, j, k) = dim_sizes * dim_sizes * dim_sizes - value;
			}
		}
	}
	TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
	tensordata.setData(tensor_values);

	// Make the expected tensor
	const int bin_width = 3;
	const int n_bins = dim_sizes * dim_sizes * dim_sizes / bin_width;
	const int n_levels = n_bins + 1;
	const float lower_level = 0.0;
	const float upper_level = bin_width * n_bins;
	TensorDataGpuPrimitiveT<float, 1> histogram_bins(Eigen::array<Eigen::Index, 1>({ n_bins }));
	histogram_bins.setData();
	std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> histogram_bins_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(histogram_bins);

	// Test
	histogram_bins_ptr->syncHAndDData(device);
	tensordata.syncHAndDData(device);
	tensordata.histogram(n_levels, lower_level, upper_level, histogram_bins_ptr, device);
	histogram_bins_ptr->syncHAndDData(device);
	gpuErrchk(cudaStreamSynchronize(stream));
	for (int i = 0; i < n_bins; ++i) {
		gpuCheckEqual(histogram_bins_ptr->getData()(i),3.0);
	}
	gpuErrchk(cudaStreamDestroy(stream));
}

void test_gettersAndSettersGpuPrimitiveT()
{
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  gpuCheckEqual(tensordata.getDimensions().at(0),2);
  gpuCheckEqual(tensordata.getDimensions().at(1),3);
  gpuCheckEqual(tensordata.getDimensions().at(2),4);
  gpuCheckEqual(tensordata.getDims(),3);
  gpuCheckEqual(tensordata.getDeviceName(),typeid(Eigen::GpuDevice).name());
  gpuCheck(!tensordata.getPinnedMemory()); // NOTE: changed default to False
  gpuCheckEqualNoLhsRhsPrint(tensordata.getPinnedFlag(),TensorDataGpuPinnedFlags::HostAllocDefault);

  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(0.5);

  tensordata.setData(data);
  gpuCheckEqual(tensordata.getData()(1, 2, 3),0.5);
  gpuCheck(tensordata.getDataStatus().first);
  gpuCheck(!tensordata.getDataStatus().second);

  // Test manual data status setters
  tensordata.setDataStatus(false, true);
  gpuCheck(!tensordata.getDataStatus().first);
  gpuCheck(tensordata.getDataStatus().second);

  // Test mutability
  tensordata.getData()(0, 0, 0) = 5;
  gpuCheckEqual(tensordata.getData()(0, 0, 0),5);

  // NOTE: not testable on the Gpu
  //// Test getDataPointer
  //Eigen::TensorMap<Eigen::Tensor<float, 3>> data_map(tensordata.getDataPointer().get(), 2, 3, 4);
  //gpuCheckEqual(data_map(0, 0, 0),5);
  //gpuCheckEqual(data_map(1, 2, 3),0);
}

void test_syncHAndDGpuPrimitiveT()
{
  TensorDataGpuPrimitiveT<float, 3> tensordata(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));

  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(0.5);
  tensordata.setData(data);

  Eigen::GpuStreamDevice stream_device;
  Eigen::GpuDevice device(&stream_device);

  // Test sync data status
  tensordata.syncHAndDData(device);
  gpuCheck(!tensordata.getDataStatus().first);
  gpuCheck(tensordata.getDataStatus().second);

  tensordata.syncHAndDData(device);
  gpuCheck(tensordata.getDataStatus().first);
  gpuCheck(!tensordata.getDataStatus().second);

  // Test that data was copied correctly between device and host
  gpuCheckEqual(tensordata.getData()(0, 0, 0),0.5);
  gpuCheckEqual(tensordata.getData()(1, 0, 0),0.5);
  gpuCheckEqual(tensordata.getData()(0, 2, 0),0.5);
  gpuCheckEqual(tensordata.getData()(0, 0, 3),0.5);
  gpuCheckEqual(tensordata.getData()(1, 2, 3),0.5);
}

void test_memoryModelsGpuPrimitiveT()
{
  // Make the dummy data
  Eigen::Tensor<float, 3> data(2, 3, 4);
  data.setConstant(1.0);

  // Test pinned default
  TensorDataGpuPrimitiveT<float, 3> tensordata_0(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }), 
    true, TensorDataGpuPinnedFlags::HostAllocDefault);
  tensordata_0.setData(data);
  gpuCheckEqual(tensordata_0.getData()(0, 0, 0),1.0);
  gpuCheckEqual(tensordata_0.getData()(1, 2, 3),1.0);

  // Test pinned HostAllocPortable
  TensorDataGpuPrimitiveT<float, 3> tensordata_1(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }), 
    true, TensorDataGpuPinnedFlags::HostAllocPortable);
  tensordata_1.setData(data);
  gpuCheckEqual(tensordata_1.getData()(0, 0, 0),1.0);
  gpuCheckEqual(tensordata_1.getData()(1, 2, 3),1.0);

  //// Test pinned HostAllocMapped [TODO: bug]
  //TensorDataGpuPrimitiveT<float, 3> tensordata_2(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }), 
  //  true, TensorDataGpuPinnedFlags::HostAllocMapped);
  //tensordata_2.setData(data);
  //gpuCheckEqual(tensordata_2.getData()(0, 0, 0),1.0);
  //gpuCheckEqual(tensordata_2.getData()(1, 2, 3),1.0);

  // Test pinned HostAllocWriteCombined
  TensorDataGpuPrimitiveT<float, 3> tensordata_3(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }), 
    true, TensorDataGpuPinnedFlags::HostAllocWriteCombined);
  tensordata_3.setData(data);
  gpuCheckEqual(tensordata_3.getData()(0, 0, 0),1.0);
  gpuCheckEqual(tensordata_3.getData()(1, 2, 3),1.0);

  // Test pageable
  TensorDataGpuPrimitiveT<float, 3> tensordata_4(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }),
    false, TensorDataGpuPinnedFlags::HostAllocDefault);
  tensordata_4.setData(data);
  gpuCheckEqual(tensordata_4.getData()(0, 0, 0),1.0);
  gpuCheckEqual(tensordata_4.getData()(1, 2, 3),1.0);
}

void test_convertFromStringToTensorTGpuPrimitiveT()
{
  Eigen::Tensor<std::string, 3> tensor_string(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensor_string.setConstant("1.001");
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device;
  Eigen::GpuDevice device(&stream_device);

  // Test float
  TensorDataGpuPrimitiveT<float, 3> tensordata_f(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_f.setData();
  tensordata_f.syncHAndDData(device);
  tensordata_f.convertFromStringToTensorT(tensor_string, device);
  tensordata_f.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckLessThan(tensordata_f.getData()(0, 0, 0), 1.001 + 1e-3);
  gpuCheckGreaterThan(tensordata_f.getData()(0, 0, 0), 1.001 - 1e-3);
  gpuCheckLessThan(tensordata_f.getData()(1, 2, 3), 1.001 + 1e-3);
  gpuCheckGreaterThan(tensordata_f.getData()(1, 2, 3),1.001 - 1e-3);

  // Test double
  TensorDataGpuPrimitiveT<double, 3> tensordata_d(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_d.setData();
  tensordata_d.syncHAndDData(device);
  tensordata_d.convertFromStringToTensorT(tensor_string, device);
  tensordata_d.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckLessThan(tensordata_d.getData()(0, 0, 0), 1.001 + 1e-3);
  gpuCheckGreaterThan(tensordata_d.getData()(0, 0, 0), 1.001 - 1e-3);
  gpuCheckLessThan(tensordata_d.getData()(1, 2, 3), 1.001 + 1e-3);
  gpuCheckGreaterThan(tensordata_d.getData()(1, 2, 3), 1.001 - 1e-3);

  // Test int
  TensorDataGpuPrimitiveT<int, 3> tensordata_i(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_i.setData();
  tensordata_i.syncHAndDData(device);
  tensordata_i.convertFromStringToTensorT(tensor_string, device);
  tensordata_i.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensordata_i.getData()(0, 0, 0),1);
  gpuCheckEqual(tensordata_i.getData()(1, 2, 3),1);

  // Test int
  TensorDataGpuPrimitiveT<char, 3> tensordata_c(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_c.setData();
  tensordata_c.syncHAndDData(device);
  tensordata_c.convertFromStringToTensorT(tensor_string, device);
  tensordata_c.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensordata_c.getData()(0, 0, 0),'1');
  gpuCheckEqual(tensordata_c.getData()(1, 2, 3),'1');
}

/* TensorDataGpuClassT Tests
*/
void test_constructorGpuClassT()
{
  TensorDataGpuClassT<TensorArrayGpu8, float, 3>* ptr = nullptr;
  TensorDataGpuClassT<TensorArrayGpu8, float, 3>* nullPointer = nullptr;
  ptr = new TensorDataGpuClassT<TensorArrayGpu8, float, 3>();
  gpuCheckNotEqual(ptr, nullPointer);
  delete ptr;
}

void test_destructorGpuClassT()
{
  TensorDataGpuClassT<TensorArrayGpu8, float, 3>* ptr = nullptr;
  ptr = new TensorDataGpuClassT<TensorArrayGpu8, float, 3>();
  delete ptr;
}
void test_syncDataGpuClassT()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the dummy data
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  Eigen::Tensor<TensorArrayGpu8<char>, 3> data(2, 3, 4);
  data.setConstant(TensorArrayGpu8<char>("hello"));
  tensordata_test.setData(data);

  // Check syncHData (no transfer)
  gpuCheck(tensordata_test.syncHData(device));
  gpuCheck(tensordata_test.getDataStatus().first);
  gpuCheck(!tensordata_test.getDataStatus().second);

  // Check syncDData (transfer)
  gpuCheck(tensordata_test.syncDData(device));
  gpuCheck(!tensordata_test.getDataStatus().first);
  gpuCheck(tensordata_test.getDataStatus().second);

  // Check syncDData (no transfer)
  gpuCheck(tensordata_test.syncDData(device));
  gpuCheck(!tensordata_test.getDataStatus().first);
  gpuCheck(tensordata_test.getDataStatus().second);

  // Check syncHData (transfer)
  gpuCheck(tensordata_test.syncHData(device));
  gpuCheck(tensordata_test.getDataStatus().first);
  gpuCheck(!tensordata_test.getDataStatus().second);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_copyGpuClassT()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // make the char array
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(2, 3, 4);
  int iter = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
        sprintf(tmp.data(), "%d", iter);
        tensor_values(i, j, k) = tmp;
        ++iter;
      }
    }
  }

  // make the tensor data class
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata_test(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_test.setData(tensor_values);
  tensordata_test.syncDData(device);

  // Check copyToHost
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> tensordata = tensordata_test.copyToHost(device);
  gpuCheckEqualNoLhsRhsPrint(tensordata->getDimensions(),tensordata_test.getDimensions());
  gpuCheckEqual(tensordata->getTensorSize(),tensordata_test.getTensorSize());
  gpuCheckEqual(tensordata->getDeviceName(),tensordata_test.getDeviceName());
  gpuCheckEqual(tensordata->getData()(0, 0, 0).at(0),'0');
  tensordata->setDataStatus(false, true);
  tensordata->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckNotEqual(tensordata->getData()(0, 0, 0).at(0),'0');

  // Check reference change
  tensordata->getData()(0, 0, 0).setTensorArray({ '2' });
  gpuCheckNotEqual(tensordata->getData()(0, 0, 0), tensordata_test.getData()(0, 0, 0));

  // Check copyToDevice
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> tensordata2 = tensordata_test.copyToDevice(device);
  gpuCheckEqualNoLhsRhsPrint(tensordata2->getDimensions(),tensordata_test.getDimensions());
  gpuCheckEqual(tensordata2->getTensorSize(),tensordata_test.getTensorSize());
  gpuCheckEqual(tensordata2->getDeviceName(),tensordata_test.getDeviceName());
  gpuCheckNotEqual(tensordata2->getData()(0, 0, 0).at(0), '0');
  tensordata2->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensordata2->getData()(0, 0, 0).at(0),'0');

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_selectGpuClassT()
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  //std::cout << "tensor_values(i, j, k)" << std::endl; //DEBUG
  int iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        Eigen::Tensor<char, 1> tmp(8); tmp.setZero(); 
        sprintf(tmp.data(), "%d", iter);
        tensor_values(i, j, k) = tmp;
        //std::cout << "i(" << i << ")j(" << j << ")k(" << k << "): "  << tensor_values(i, j, k)  << std::endl; //DEBUG
        if (i == dim_sizes - 1 || j == dim_sizes - 1 || k == dim_sizes - 1)
          indices_values(i, j, k) = 0;
        else
          indices_values(i, j, k) = 1;
        ++iter;

      }
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 3>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices);

  // Make the expected tensor
  int dim_sizes_select = 2;
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  iter = 0;
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        if (i < dim_sizes - 1 && j < dim_sizes - 1 && k < dim_sizes - 1) {
          Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
          sprintf(tmp.data(), "%d", iter);
          tensor_values_test(i, j, k) = tmp;
        }
        ++iter;
      }
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensorselect(Eigen::array<Eigen::Index, 3>({ dim_sizes_select, dim_sizes_select, dim_sizes_select }));
  tensorselect.setData();
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> tensorselect_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(tensorselect);

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  tensordata.select(tensorselect_ptr, indices_ptr, device);
  tensordata.syncHAndDData(device);
  tensorselect_ptr->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuErrchk(cudaStreamDestroy(stream));

  for (int i = 0; i < dim_sizes_select; ++i) {
    for (int j = 0; j < dim_sizes_select; ++j) {
      for (int k = 0; k < dim_sizes_select; ++k) {
        //std::cout << "Test Select i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensorselect_ptr->getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        gpuCheckEqual(tensorselect_ptr->getData()(i, j, k),tensor_values_test(i, j, k));
      }
    }
  }
}

void test_sortGpuClassT()
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
    tensor_values.data()[i] = TensorArrayGpu8<char>(std::to_string(i));
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test ASC
  tensordata.syncHAndDData(device);
  tensordata.sort("ASC", device);
  tensordata.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
    sprintf(tmp.data(), "%d", tensor_values.size() - i - 1);
    tensor_values_test.data()[i] = tmp;
  }

  // Test DESC
  tensordata.setDataStatus(false, true);
  tensordata.sort("DESC", device);
  tensordata.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_sortIndicesGpuClassT()
{
  // Make the tensor data and select indices
  int dim_sizes = 3;
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values.data()[i] = i + 1;
    Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
    sprintf(tmp.data(), "%d", i);
    tensor_values.data()[i] = tmp;
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices);

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test ASC
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  tensordata.sortIndices(indices_ptr, "ASC", device);
  tensordata.sort(indices_ptr, device);
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values(i, j, k) << std::endl;
        //std::cout << "Test ASC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values(i, j, k) << std::endl;
        gpuCheckEqual(indices_ptr->getData()(i, j, k),indices_values(i, j, k));
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values(i, j, k));
      }
    }
  }

  // Make the expected indices and values
  Eigen::Tensor<int, 3> indices_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values_test(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    indices_values_test.data()[i] = tensor_values.size() - i;

    Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
    sprintf(tmp.data(), "%d", tensor_values.size() - i - 1);
    tensor_values_test.data()[i] = tmp;
  }

  // Test DESC
  tensordata.setDataStatus(false, true);
  indices_ptr->setDataStatus(false, true);
  tensordata.sortIndices(indices_ptr, "DESC", device);
  tensordata.sort(indices_ptr, device);
  tensordata.syncHAndDData(device);
  indices_ptr->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Indices sorted: " << indices_ptr->getData()(i, j, k) << "; Expected: " << indices_values_test(i, j, k) << std::endl;
        //std::cout << "Test DESC i,j,k :" << i << "," << j << "," << k << "; Tensor select: " << tensordata.getData()(i, j, k) << "; Expected: " << tensor_values_test(i, j, k) << std::endl;
        gpuCheckEqual(indices_ptr->getData()(i, j, k),indices_values_test(i, j, k));
        gpuCheckEqual(tensordata.getData()(i, j, k),tensor_values_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_partitionGpuClassT()
{
  // Make the tensor data and partition indices
  int dim_sizes = 3;
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  for (int i = 0; i < tensor_values.size(); ++i) {
    if (i % 2 == 0)
      indices_values.data()[i] = 1;
    else
      indices_values.data()[i] = 0;
    Eigen::Tensor<char, 1> tmp(8); tmp.setZero();
    sprintf(tmp.data(), "%d", i);
    tensor_values.data()[i] = tmp;
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));
  indices.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices);

  // Make the expected partitioned data tensor
  Eigen::Tensor<TensorArrayGpu8<char>, 1> expected_values_flat(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes * dim_sizes }));
  expected_values_flat.setValues({ 
    TensorArrayGpu8<char>({'0'}),
    TensorArrayGpu8<char>({'2'}),
    TensorArrayGpu8<char>({'4'}),
    TensorArrayGpu8<char>({'6'}),
    TensorArrayGpu8<char>({'8'}),
    TensorArrayGpu8<char>({'1','0'}),
    TensorArrayGpu8<char>({'1','2'}),
    TensorArrayGpu8<char>({'1','4'}),
    TensorArrayGpu8<char>({'1','6'}),
    TensorArrayGpu8<char>({'1','8'}),
    TensorArrayGpu8<char>({'2','0'}),
    TensorArrayGpu8<char>({'2','2'}),
    TensorArrayGpu8<char>({'2','4'}),
    TensorArrayGpu8<char>({'2','6'}),
    //TensorArrayGpu8<char>({'2','3'}),
    //TensorArrayGpu8<char>({'2','1'}),
    //TensorArrayGpu8<char>({'1','9'}),
    //TensorArrayGpu8<char>({'1','7'}),
    //TensorArrayGpu8<char>({'1','5'}),
    //TensorArrayGpu8<char>({'1','3'}),
    //TensorArrayGpu8<char>({'1','1'}),
    //TensorArrayGpu8<char>({'9'}),
    //TensorArrayGpu8<char>({'7'}),
    //TensorArrayGpu8<char>({'5'}),
    //TensorArrayGpu8<char>({'3'}),
    //TensorArrayGpu8<char>({'1'})
    TensorArrayGpu8<char>({'1'}),
    TensorArrayGpu8<char>({'3'}),
    TensorArrayGpu8<char>({'5'}),
    TensorArrayGpu8<char>({'7'}),
    TensorArrayGpu8<char>({'9'}),
    TensorArrayGpu8<char>({'1','1'}),
    TensorArrayGpu8<char>({'1','3'}),
    TensorArrayGpu8<char>({'1','5'}),
    TensorArrayGpu8<char>({'1','7'}),
    TensorArrayGpu8<char>({'1','9'}),
    TensorArrayGpu8<char>({'2','1'}),
    TensorArrayGpu8<char>({'2','3'}),
    TensorArrayGpu8<char>({'2','5'})
    });
  Eigen::Tensor<TensorArrayGpu8<char>, 3> expected_values = expected_values_flat.reshape(Eigen::array<Eigen::Index, 3>({ dim_sizes, dim_sizes, dim_sizes }));

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test partition
  indices_ptr->syncHAndDData(device);
  tensordata.syncHAndDData(device);
  tensordata.partition(indices_ptr, device);
  tensordata.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < dim_sizes; ++i) {
    for (int j = 0; j < dim_sizes; ++j) {
      for (int k = 0; k < dim_sizes; ++k) {
        //std::cout << "Test Partition i,j,k :" << i << "," << j << "," << k << "; Tensor partition: " << tensordata.getData()(i, j, k) << "; Expected: " << expected_values(i, j, k) << std::endl;
        gpuCheckEqual(tensordata.getData()(i, j, k),expected_values(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}


void test_runLengthEncodeGpuClassT()
{
  // Make the tensor data and expected data values
  int dim_sizes = 3;
  Eigen::Tensor<TensorArrayGpu8<char>, 2> tensor_values(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensor_values.setValues({ 
    {TensorArrayGpu8<char>({'0'}), TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'})}, 
    {TensorArrayGpu8<char>({'0'}), TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'})}, 
    {TensorArrayGpu8<char>({'0'}), TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'})}
    });
  Eigen::Tensor<TensorArrayGpu8<char>, 1> unique_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique_values.setValues({
    TensorArrayGpu8<char>({'0'}), TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'}), 
    TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}), 
    TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}) 
    });
  Eigen::Tensor<int, 1> count_values(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count_values.setValues({ 3, 3, 3, -1, -1, -1, -1, -1, -1 });
  Eigen::Tensor<int, 1> num_runs_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs_values.setValues({ 3 });

  // Make the tensor data and test data pointers
  TensorDataGpuClassT<TensorArrayGpu8, char, 2> tensordata(Eigen::array<Eigen::Index, 2>({ dim_sizes, dim_sizes }));
  tensordata.setData(tensor_values);
  TensorDataGpuClassT<TensorArrayGpu8, char, 1> unique(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  unique.setData();
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 1>> uniquev_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(unique);
  TensorDataGpuPrimitiveT<int, 1> count(Eigen::array<Eigen::Index, 1>({ dim_sizes * dim_sizes }));
  count.setData();
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> count_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(count);
  TensorDataGpuPrimitiveT<int, 1> num_runs(Eigen::array<Eigen::Index, 1>({ 1 }));
  num_runs.setData();
  num_runs.getData()(0) = 0;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> num_runs_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(num_runs);

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
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
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(num_runs_ptr->getData()(0),3);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    gpuCheckEqual(uniquev_ptr->getData()(i),unique_values(i));
    gpuCheckEqual(count_ptr->getData()(i),count_values(i));
  }

  // Test runLengthEncode for the case where the last value is not the same as the previous
  tensor_values.setValues({ 
    { TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'}), TensorArrayGpu8<char>({'3'})},
    { TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'}), TensorArrayGpu8<char>({'3'})},
    { TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'}), TensorArrayGpu8<char>({'4'})}
    });
  tensordata.setData(tensor_values);
  unique_values.setValues({ 
    TensorArrayGpu8<char>({'1'}), TensorArrayGpu8<char>({'2'}), TensorArrayGpu8<char>({'3'}),
    TensorArrayGpu8<char>({'4'}), TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}),
    TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}), TensorArrayGpu8<char>({'\0'}) 
    });
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
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(num_runs_ptr->getData()(0),4);
  for (int i = 0; i < num_runs_ptr->getData()(0); ++i) {
    gpuCheckEqual(uniquev_ptr->getData()(i),unique_values(i));
    gpuCheckEqual(count_ptr->getData()(i),count_values(i));
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_convertFromStringToTensorTGpuClassT()
{
  Eigen::Tensor<std::string, 3> tensor_string(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensor_string.setConstant("1.001");
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
  Eigen::GpuStreamDevice stream_device;
  Eigen::GpuDevice device(&stream_device);

  // Test array  // FIXME: causing a memory leak on deallocation
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensordata_a8(Eigen::array<Eigen::Index, 3>({ 2, 3, 4 }));
  tensordata_a8.setData();
  tensordata_a8.syncHAndDData(device);
  tensordata_a8.convertFromStringToTensorT(tensor_string, device);
  tensordata_a8.syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensordata_a8.getData()(0, 0, 0),TensorArrayGpu8<char>("1.001"));
  gpuCheckEqual(tensordata_a8.getData()(1, 2, 3),TensorArrayGpu8<char>("1.001"));
}

int main(int argc, char** argv)
{
  // PrimitiveT
  gpuCheckEqual(cudaDeviceReset(),cudaSuccess);
  test_constructorGpuPrimitiveT();
  test_destructorGpuPrimitiveT();
  test_gettersAndSettersGpuPrimitiveT();
  test_syncHAndDGpuPrimitiveT();
  test_memoryModelsGpuPrimitiveT();
  test_assignmentGpuPrimitiveT();
  test_syncDataGpuPrimitiveT();
  test_copyGpuPrimitiveT();
  test_selectGpuPrimitiveT();
  test_sortGpuPrimitiveT();
  test_sortIndicesGpuPrimitiveT();
  test_partitionGpuPrimitiveT();
  test_runLengthEncodeGpuPrimitiveT();
	test_histogramGpuPrimitiveT();
  test_convertFromStringToTensorTGpuPrimitiveT();

  // ClassT
  gpuCheckEqual(cudaDeviceReset(),cudaSuccess);
  test_constructorGpuClassT();
  test_destructorGpuClassT();
  test_syncDataGpuClassT();
  test_copyGpuClassT();
  test_selectGpuClassT();
  //test_sortGpuClassT(); // run time device synchronization error during sort (2nd pass in Thrust merge_sort)
  //test_sortIndicesGpuClassT(); // run time device synchronization error during sort (2nd pass in Thrust merge_sort)
  test_partitionGpuClassT();
  test_runLengthEncodeGpuClassT();
  test_convertFromStringToTensorTGpuClassT();
  return 0;
}
#endif