/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorAxisGpu.h>

using namespace TensorBase;
using namespace std;

void test_constructorGpuPrimitiveT() 
{
	TensorAxisGpuPrimitiveT<int>* ptr = nullptr;
	TensorAxisGpuPrimitiveT<int>* nullPointer = nullptr;
	ptr = new TensorAxisGpuPrimitiveT<int>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpuPrimitiveT()
{
	TensorAxisGpuPrimitiveT<int>* ptr = nullptr;
	ptr = new TensorAxisGpuPrimitiveT<int>();
  delete ptr;
}

void test_constructor1GpuPrimitiveT()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  assert(tensoraxis.getDimensions()(2) == "TensorDimension3");
  assert(tensoraxis.getLabels()(0, 0) == 1);
  assert(tensoraxis.getLabels()(2, 4) == 1);
}

void test_constructor2GpuPrimitiveT()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", 1, 1);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 1);
  assert(tensoraxis.getNLabels() == 1);

  tensoraxis.setDimensions(dimensions);
  tensoraxis.setLabels(labels);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  assert(tensoraxis.getDimensions()(2) == "TensorDimension3");
  assert(tensoraxis.getLabels()(0, 0) == 1);
  assert(tensoraxis.getLabels()(2, 4) == 1);
}

void test_gettersAndSettersGpuPrimitiveT()
{
  TensorAxisGpuPrimitiveT<int> tensoraxis;
  // Check defaults
  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "");
  assert(tensoraxis.getNLabels() == 0);
  assert(tensoraxis.getNDimensions() == 0);

  // Check getters/setters
  tensoraxis.setId(1);
  tensoraxis.setName("1");
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  tensoraxis.setDimensionsAndLabels(dimensions, labels);

  assert(tensoraxis.getId() == 1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  assert(tensoraxis.getDimensions()(2) == "TensorDimension3");
  assert(tensoraxis.getLabels()(0, 0) == 1);
  assert(tensoraxis.getLabels()(2, 4) == 1);
}

void test_copyGpuPrimitiveT()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisGpuPrimitiveT<int> tensoraxis1("1", dimensions, labels);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test expected
  tensoraxis1.syncDData(device);
  auto tensoraxis_copy = tensoraxis1.copyToHost(device);
  tensoraxis1.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(*(tensoraxis_copy.get()) == tensoraxis1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      assert(tensoraxis_copy->getLabels()(i, j) == labels(i, j));
    }
  }
  auto tensoraxis2_copy = tensoraxis1.copyToDevice(device);
  tensoraxis2_copy->syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  assert(*(tensoraxis2_copy.get()) == tensoraxis1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      assert(tensoraxis2_copy->getLabels()(i, j) == labels(i, j));
    }
  }
}

void test_deleteFromAxisGpuPrimitiveT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  // Setup the selection indices and the expected labels
  int n_select_labels = 3;
  Eigen::Tensor<int, 1> indices_values(n_labels);
  Eigen::Tensor<int, 2> labels_test(n_dimensions, n_select_labels);
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      if (j % 2 == 0) {
        indices_values(j) = j + 1;
        labels_test(i, j/2) = iter;
      }
      else {
        indices_values(j) = 0;
      }
      ++iter;
    }
  }
  TensorDataGpuPrimitiveT<int, 1> indices(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices.setData(indices_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  indices_ptr->syncDData(device);
  tensoraxis.syncDData(device);
  tensoraxis.deleteFromAxis(indices_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_select_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_select_labels; ++j) {
      std::cout << "Test Labels i,j :" << i << "," << j << "; Reduced labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels_test(i, j) << std::endl;
      //assert(tensoraxis.getLabels()(i, j) == labels_test(i, j)); // FIXME
    }
  }
}

void test_appendLabelsToAxis1GpuPrimitiveT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  // Setup the new labels
  int n_new_labels = 2;
  Eigen::Tensor<int, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_new_labels; ++j) {
      labels_values(i, j) = iter;
      ++iter;
    }
  }
  TensorDataGpuPrimitiveT<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> labels_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_new);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  labels_new_ptr->syncDData(device);
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxis(labels_new_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels + n_new_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; Original labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels(i, j) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels(i, j));
    }
  }
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = n_labels; j < tensoraxis.getNLabels(); ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; New labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels_values(i, j - n_labels) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels_values(i, j - n_labels));
    }
  }
}

void test_appendLabelsToAxis2GpuPrimitiveT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorDataGpuPrimitiveT<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  labels_new.setData(labels);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> labels_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_new);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  labels_new_ptr->syncDData(device);
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxis(labels_new_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; Original labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels(i, j) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels(i, j));
    }
  }
}

void test_makeSortIndicesGpuPrimitiveT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = j;
    }
  }
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataGpuPrimitiveT<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_view_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_view);

  indices_view_ptr->syncDData(device);
  tensoraxis.syncDData(device);

  // make the expected indices
  Eigen::Tensor<int, 2> indices_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      indices_sort_test(i, j) = i + j * n_dimensions + 1;
    }
  }

  // test making the sort indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_sort;
  tensoraxis.makeSortIndices(indices_view_ptr, indices_sort, device);
  indices_sort->syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(indices_sort->getData()(i, j) == indices_sort_test(i, j));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortLabelsGpuPrimitiveT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = j;
    }
  }
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataGpuPrimitiveT<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_view_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_view);

  indices_view_ptr->syncDData(device);
  tensoraxis.syncDData(device);

  // test sorting ASC
  tensoraxis.sortLabels(indices_view_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == labels(i, j));
    }
  }

  // make the expected labels
  Eigen::Tensor<int, 2> labels_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_sort_test(i, j) = n_labels - j - 1;
    }
  }
  indices_view_ptr->setDataStatus(true, false);
  for (int i = 0; i < n_labels; ++i)
    indices_view_ptr->getData()(i) = n_labels - i;
  indices_view_ptr->syncDData(device);

  // test sorting DESC
  tensoraxis.setDataStatus(false, true);
  tensoraxis.sortLabels(indices_view_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == labels_sort_test(i, j));
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_storeAndLoadLabelsGpuPrimitiveT()
{
  // Setup the axis
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisGpuPrimitiveT<int> tensoraxis_io("1", dimensions, labels);

  // Store the axis data
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);
  tensoraxis_io.storeLabelsBinary("axis", device);

  // Load the axis data
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", 3, 5);
  tensoraxis.loadLabelsBinary("axis", device);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  //assert(tensoraxis.getDimensions()(0) == "TensorDimension1"); // Not loaded
  //assert(tensoraxis.getDimensions()(1) == "TensorDimension2"); // Not loaded
  //assert(tensoraxis.getDimensions()(2) == "TensorDimension3"); // Not loaded
  assert(tensoraxis.getLabels()(0, 0) == 1);
  assert(tensoraxis.getLabels()(2, 4) == 1);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_getLabelsAsStringsGpuPrimitiveT()
{
  // Setup the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the axis
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  // Test getLabelsAsString
  tensoraxis.syncDData(device);
  std::vector<std::string> labels_str = tensoraxis.getLabelsAsStrings(device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  int iter = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      assert(labels_str.at(iter) == std::to_string(tensoraxis.getLabels()(i, j)));
    }
    ++iter;
  }

  // Using Char
  Eigen::Tensor<char, 2> labels_char(3, 5);
  labels_char.setConstant('a');
  TensorAxisGpuPrimitiveT<char> tensoraxis_char("1", dimensions, labels_char);

  // Test getLabelsAsString
  tensoraxis_char.syncDData(device);
  std::vector<std::string> labels_char_str = tensoraxis_char.getLabelsAsStrings(device);
  tensoraxis_char.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      assert(labels_char_str.at(iter) == std::to_string(tensoraxis_char.getLabels()(i, j)));
    }
    ++iter;
  }

  // Use TensorArray8
  Eigen::Tensor<TensorArrayGpu8<int>, 2> labels_array(3, 5);
  labels_array.setConstant(TensorArrayGpu8<int>({ 1,2,3,4,5,6,7,8 }));
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis_array("1", dimensions, labels_array);

  // Test getLabelsAsString
  tensoraxis_array.syncDData(device);
  std::vector<std::string> labels_array_str = tensoraxis_array.getLabelsAsStrings(device);
  tensoraxis_array.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      assert(labels_array_str.at(iter) == labels_array(i, j).getTensorArrayAsString());
    }
    ++iter;
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_appendLabelsToAxisFromCsv1GpuPrimitiveT()
{
  // Setup the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", dimensions, labels);

  // Setup the new labels
  int n_new_labels = 4;
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_new_labels; ++j) {
      if (j < 2)
        labels_values(i, j) = std::to_string(i + j * n_dimensions + iter);
      else
        labels_values(i, j) = std::to_string(i + (j - 2) * n_dimensions + iter); // duplicates
    }
  }

  // Test
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxisFromCsv(labels_values, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels + 2);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == labels(i, j));
    }
  }
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = n_labels; j < tensoraxis.getNLabels(); ++j) {
      assert(tensoraxis.getLabels()(i, j) == std::stoi(labels_values(i, j - n_labels)));
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_appendLabelsToAxisFromCsv2GpuPrimitiveT()
{
  // Setup the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisGpuPrimitiveT<int> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_values(i, j) = std::to_string(i + j * n_dimensions);
    }
  }

  // Test
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxisFromCsv(labels_values, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == i + j * n_dimensions);
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_constructorGpuClassT()
{
  TensorAxisGpuClassT<TensorArrayGpu8, int>* ptr = nullptr;
  TensorAxisGpuClassT<TensorArrayGpu8, int>* nullPointer = nullptr;
  ptr = new TensorAxisGpuClassT<TensorArrayGpu8, int>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpuClassT()
{
  TensorAxisGpuClassT<TensorArrayGpu8, int>* ptr = nullptr;
  ptr = new TensorAxisGpuClassT<TensorArrayGpu8, int>();
  delete ptr;
}

void test_constructor1GpuClassT()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<TensorArrayGpu8<int>, 2> labels(3, 5);
  labels.setConstant(TensorArrayGpu8<int>({1, 1, 1, 1, 1, 1, 1, 1 }));
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis("1", dimensions, labels);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  assert(tensoraxis.getDimensions()(2) == "TensorDimension3");
  assert(tensoraxis.getLabels()(0, 0).getTensorArray()(0) == 1);
  assert(tensoraxis.getLabels()(2, 4).getTensorArray()(0) == 1);
}

void test_constructor2GpuClassT()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<TensorArrayGpu8<int>, 2> labels(3, 5);
  labels.setConstant(TensorArrayGpu8<int>({ 1, 1, 1, 1, 1, 1, 1, 1 }));
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis("1", 1, 1);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 1);
  assert(tensoraxis.getNLabels() == 1);

  tensoraxis.setDimensions(dimensions);
  tensoraxis.setLabels(labels);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  assert(tensoraxis.getDimensions()(2) == "TensorDimension3");
  assert(tensoraxis.getLabels()(0, 0).getTensorArray()(0) == 1);
  assert(tensoraxis.getLabels()(2, 4).getTensorArray()(0) == 1);
}

void test_gettersAndSettersGpuClassT()
{
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis;
  // Check defaults
  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "");
  assert(tensoraxis.getNLabels() == 0);
  assert(tensoraxis.getNDimensions() == 0);

  // Check getters/setters
  tensoraxis.setId(1);
  tensoraxis.setName("1");
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<TensorArrayGpu8<int>, 2> labels(3, 5);
  labels.setConstant(TensorArrayGpu8<int>({ 1, 1, 1, 1, 1, 1, 1, 1 }));
  tensoraxis.setDimensionsAndLabels(dimensions, labels);

  assert(tensoraxis.getId() == 1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  assert(tensoraxis.getDimensions()(2) == "TensorDimension3");
  assert(tensoraxis.getLabels()(0, 0).getTensorArray()(0) == 1);
  assert(tensoraxis.getLabels()(2, 4).getTensorArray()(0) == 1);
}

void test_copyGpuClassT()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<TensorArrayGpu8<int>, 2> labels(3, 5);
  labels.setConstant(TensorArrayGpu8<int>({ 1, 1, 1, 1, 1, 1, 1, 1 }));
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis1("1", dimensions, labels);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test expected
  tensoraxis1.syncDData(device);
  auto tensoraxis_copy = tensoraxis1.copyToHost(device);
  tensoraxis1.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(*(tensoraxis_copy.get()) == tensoraxis1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      assert(tensoraxis_copy->getLabels()(i, j) == labels(i, j));
    }
  }
  auto tensoraxis2_copy = tensoraxis1.copyToDevice(device);
  tensoraxis2_copy->syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  assert(*(tensoraxis2_copy.get()) == tensoraxis1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      assert(tensoraxis2_copy->getLabels()(i, j) == labels(i, j));
    }
  }
}

void test_deleteFromAxisGpuClassT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j).setTensorArray(std::to_string(iter));
      ++iter;
    }
  }
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", dimensions, labels);

  // Setup the selection indices and the expected labels
  int n_select_labels = 3;
  Eigen::Tensor<int, 1> indices_values(n_labels);
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_test(n_dimensions, n_select_labels);
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      if (j % 2 == 0) {
        indices_values(j) = j + 1;
        labels_test(i, j / 2).setTensorArray(std::to_string(iter));
      }
      else {
        indices_values(j) = 0;
      }
      ++iter;
    }
  }
  TensorDataGpuPrimitiveT<int, 1> indices(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices.setData(indices_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> indices_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  indices_ptr->syncDData(device);
  tensoraxis.syncDData(device);
  tensoraxis.deleteFromAxis(indices_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_select_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_select_labels; ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; Reduced labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels_test(i, j) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels_test(i, j));
    }
  }
}

void test_appendLabelsToAxis1GpuClassT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j).setTensorArray(std::to_string(iter));
      ++iter;
    }
  }
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", dimensions, labels);

  // Setup the new labels
  int n_new_labels = 2;
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_new_labels; ++j) {
      labels_values(i, j).setTensorArray(std::to_string(iter));
      ++iter;
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorDataGpuClassT<TensorArrayGpu8, char, 2>> labels_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(labels_new);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  labels_new_ptr->syncDData(device);
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxis(labels_new_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels + n_new_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; Original labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels(i, j) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels(i, j));
    }
  }
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = n_labels; j < tensoraxis.getNLabels(); ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; New labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels_values(i, j - n_labels) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels_values(i, j - n_labels));
    }
  }
}

void test_appendLabelsToAxis2GpuClassT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_values(i, j).setTensorArray(std::to_string(iter));
      ++iter;
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorDataGpuClassT<TensorArrayGpu8, char, 2>> labels_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(labels_new);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  labels_new_ptr->syncDData(device);
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxis(labels_new_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      //std::cout << "Test Labels i,j :" << i << "," << j << "; Original labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels_values(i, j) << std::endl;
      assert(tensoraxis.getLabels()(i, j) == labels_values(i, j));
    }
  }
}

void test_makeSortIndicesGpuClassT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j).setTensorArray(std::to_string(j));
    }
  }
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", dimensions, labels);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataGpuPrimitiveT<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_view_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_view);

  indices_view_ptr->syncDData(device);
  tensoraxis.syncDData(device);

  // make the expected indices
  Eigen::Tensor<int, 2> indices_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      indices_sort_test(i, j) = i + j * n_dimensions + 1;
    }
  }

  // test making the sort indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_sort;
  tensoraxis.makeSortIndices(indices_view_ptr, indices_sort, device);
  indices_sort->syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(indices_sort->getData()(i, j) == indices_sort_test(i, j));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortLabelsGpuClassT()
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j).setTensorArray(std::to_string(j));
    }
  }
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", dimensions, labels);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataGpuPrimitiveT<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_view_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_view);

  indices_view_ptr->syncDData(device);
  tensoraxis.syncDData(device);

  // test sorting ASC
  tensoraxis.sortLabels(indices_view_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == labels(i, j));
    }
  }

  // make the expected labels
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_sort_test(i, j).setTensorArray(std::to_string(n_labels - j - 1));
    }
  }
  indices_view_ptr->setDataStatus(true, false);
  for (int i = 0; i < n_labels; ++i)
    indices_view_ptr->getData()(i) = n_labels - i;
  indices_view_ptr->syncDData(device);

  // test sorting DESC
  tensoraxis.setDataStatus(false, true);
  tensoraxis.sortLabels(indices_view_ptr, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == labels_sort_test(i, j));
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_storeAndLoadLabelsGpuClassT()
{
  // Setup the axis
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<TensorArrayGpu8<int>, 2> labels(3, 5);
  labels.setConstant(TensorArrayGpu8<int>({ 1, 1, 1, 1, 1, 1, 1, 1 }));
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis_io("1", dimensions, labels);

  // Store the axis data
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);
  tensoraxis_io.storeLabelsBinary("axis", device);

  // Load the axis data
  TensorAxisGpuClassT<TensorArrayGpu8, int> tensoraxis("1", 3, 5);
  tensoraxis.loadLabelsBinary("axis", device);

  assert(tensoraxis.getId() == -1);
  assert(tensoraxis.getName() == "1");
  assert(tensoraxis.getNDimensions() == 3);
  assert(tensoraxis.getNLabels() == 5);
  //assert(tensoraxis.getDimensions()(0) == "TensorDimension1"); // Not loaded
  //assert(tensoraxis.getDimensions()(1) == "TensorDimension2"); // Not loaded
  //assert(tensoraxis.getDimensions()(2) == "TensorDimension3"); // Not loaded
  assert(tensoraxis.getLabels()(0, 0).getTensorArray()(0) == 1);
  assert(tensoraxis.getLabels()(2, 4).getTensorArray()(0) == 1);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_appendLabelsToAxisFromCsv1GpuClassT()
{
  // Setup the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<TensorArrayGpu8<char>, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = TensorArrayGpu8<char>(std::to_string(iter));
      ++iter;
    }
  }
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", dimensions, labels);

  // Setup the new labels
  int n_new_labels = 4;
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_new_labels; ++j) {
      if (j < 2)
        labels_values(i, j) = std::to_string(i + j * n_dimensions + iter);
      else
        labels_values(i, j) = std::to_string(i + (j - 2) * n_dimensions + iter); // duplicates
    }
  }

  // Test
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxisFromCsv(labels_values, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels + 2);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == TensorArrayGpu8<char>(labels(i, j)));
    }
  }
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = n_labels; j < tensoraxis.getNLabels(); ++j) {
      assert(tensoraxis.getLabels()(i, j) == TensorArrayGpu8<char>(labels_values(i, j - n_labels)));
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_appendLabelsToAxisFromCsv2GpuClassT()
{
  // Setup the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisGpuClassT<TensorArrayGpu8, char> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_values(i, j) = std::to_string(i + j * n_dimensions);
    }
  }

  // Test
  tensoraxis.syncDData(device);
  tensoraxis.appendLabelsToAxisFromCsv(labels_values, device);
  tensoraxis.syncHData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      assert(tensoraxis.getLabels()(i, j) == TensorArrayGpu8<char>(std::to_string(i + j * n_dimensions)));
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  assert(cudaDeviceReset() == cudaSuccess);
  test_constructorGpuPrimitiveT();
  test_destructorGpuPrimitiveT();
  test_constructor1GpuPrimitiveT();
  test_constructor2GpuPrimitiveT();
  test_gettersAndSettersGpuPrimitiveT();
  test_copyGpuPrimitiveT();
  test_deleteFromAxisGpuPrimitiveT();
  test_appendLabelsToAxis1GpuPrimitiveT();
  test_appendLabelsToAxis2GpuPrimitiveT();
  test_makeSortIndicesGpuPrimitiveT();
  test_sortLabelsGpuPrimitiveT();
  test_storeAndLoadLabelsGpuPrimitiveT();
  test_appendLabelsToAxisFromCsv1GpuPrimitiveT();
  test_appendLabelsToAxisFromCsv2GpuPrimitiveT();

  assert(cudaDeviceReset() == cudaSuccess);
  test_constructorGpuClassT();
  test_destructorGpuClassT();
  test_constructor1GpuClassT(); 
  test_constructor2GpuClassT();
  test_gettersAndSettersGpuClassT();
  test_copyGpuClassT();
  test_deleteFromAxisGpuClassT();
  test_appendLabelsToAxis1GpuClassT();
  test_appendLabelsToAxis2GpuClassT();
  test_makeSortIndicesGpuClassT();
  test_sortLabelsGpuClassT();
  test_storeAndLoadLabelsGpuClassT();
  test_appendLabelsToAxisFromCsv1GpuClassT();
  test_appendLabelsToAxisFromCsv2GpuClassT();
  return 0;
}
#endif