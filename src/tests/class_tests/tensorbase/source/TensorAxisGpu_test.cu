/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorAxisGpu.h>

using namespace TensorBase;
using namespace std;

void test_constructorGpu() 
{
	TensorAxisGpu<int>* ptr = nullptr;
	TensorAxisGpu<int>* nullPointer = nullptr;
	ptr = new TensorAxisGpu<int>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
	TensorAxisGpu<int>* ptr = nullptr;
	ptr = new TensorAxisGpu<int>();
  delete ptr;
}

void test_constructor1Gpu()
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisGpu<int> tensoraxis("1", dimensions, labels);

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

void test_gettersAndSettersGpu()
{
  TensorAxisGpu<int> tensoraxis;
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

void test_deleteFromAxisGpu()
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
  TensorAxisGpu<int> tensoraxis("1", dimensions, labels);

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
  TensorDataGpu<int, 1> indices(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices.setData(indices_values);
  std::shared_ptr<TensorDataGpu<int, 1>> indices_ptr = std::make_shared<TensorDataGpu<int, 1>>(indices);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  indices_ptr->syncHAndDData(device);
  tensoraxis.syncHAndDData(device);
  tensoraxis.deleteFromAxis(indices_ptr, device);
  indices_ptr->syncHAndDData(device);
  tensoraxis.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);

  assert(tensoraxis.getNDimensions() == n_dimensions);
  assert(tensoraxis.getNLabels() == n_select_labels);
  assert(tensoraxis.getDimensions()(0) == "TensorDimension1");
  assert(tensoraxis.getDimensions()(1) == "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_select_labels; ++j) {
      std::cout << "Test Labels i,j :" << i << "," << j << "; Reduced labels: " << tensoraxis.getLabels()(i, j) << "; Expected: " << labels_test(i, j) << std::endl;
      //assert(tensoraxis.getLabels()(i, j) == labels_test(i, j));
    }
  }
}

void test_appendLabelsToAxisGpu()
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
  TensorAxisGpu<int> tensoraxis("1", dimensions, labels);

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
  TensorDataGpu<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorDataGpu<int, 2>> labels_new_ptr = std::make_shared<TensorDataGpu<int, 2>>(labels_new);

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Test
  labels_new_ptr->syncHAndDData(device);
  tensoraxis.syncHAndDData(device);
  tensoraxis.appendLabelsToAxis(labels_new_ptr, device);
  tensoraxis.syncHAndDData(device);
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

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_constructor1Gpu();
  test_gettersAndSettersGpu();
  //test_getLabelsDataPointerGpu(); // Not needed?
  test_deleteFromAxisGpu();
  test_appendLabelsToAxisGpu();
  return 0;
}
#endif