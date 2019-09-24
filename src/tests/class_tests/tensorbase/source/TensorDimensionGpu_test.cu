/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorDimensionGpu.h>
#include <TensorBase/ml/TensorArrayGpu.h>

using namespace TensorBase;
using namespace std;

/*TensorDimensionGpuPrimitiveT Tests*/
void test_constructorGpuPrimitiveT()
{
  TensorDimensionGpuPrimitiveT<int>* ptr = nullptr;
  TensorDimensionGpuPrimitiveT<int>* nullPointer = nullptr;
  ptr = new TensorDimensionGpuPrimitiveT<int>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpuPrimitiveT()
{
  TensorDimensionGpuPrimitiveT<int>* ptr = nullptr;
  ptr = new TensorDimensionGpuPrimitiveT<int>();
  delete ptr;
}

void test_constructorNameGpuPrimitiveT()
{
  TensorDimensionGpuPrimitiveT<int> tensordimension("1", "dir");
  assert(tensordimension.getId() == -1);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getDir() == "dir");
  assert(tensordimension.getNLabels() == 0);
}

void test_constructorNameAndLabelsGpuPrimitiveT()
{
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  TensorDimensionGpuPrimitiveT<int> tensordimension("1", "dir", labels);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getDir() == "dir");
  assert(tensordimension.getNLabels() == 5);
  assert(tensordimension.getLabels()(0) == 1);
  assert(tensordimension.getLabels()(4) == 1);
}

void test_gettersAndSettersGpuPrimitiveT()
{
  TensorDimensionGpuPrimitiveT<int> tensordimension;
  // Check defaults
  assert(tensordimension.getId() == -1);
  assert(tensordimension.getName() == "");
  assert(tensordimension.getDir() == "");
  assert(tensordimension.getNLabels() == 0);

  // Check getters/setters
  tensordimension.setId(1);
  tensordimension.setName("1");
  tensordimension.setDir("dir");
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  tensordimension.setLabels(labels);

  assert(tensordimension.getId() == 1);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getDir() == "dir");
  assert(tensordimension.getNLabels() == 5);
  assert(tensordimension.getLabels()(0) == 1);
  assert(tensordimension.getLabels()(4) == 1);
}

/*TensorDimensionGpuClassT Tests*/
void test_constructorGpuClassT()
{
  TensorDimensionGpuClassT<TensorArrayGpu8, int>* ptr = nullptr;
  TensorDimensionGpuClassT<TensorArrayGpu8, int>* nullPointer = nullptr;
  ptr = new TensorDimensionGpuClassT<TensorArrayGpu8, int>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpuClassT()
{
  TensorDimensionGpuClassT<TensorArrayGpu8, int>* ptr = nullptr;
  ptr = new TensorDimensionGpuClassT<TensorArrayGpu8, int>();
  delete ptr;
}

void test_constructorNameGpuClassT()
{
  TensorDimensionGpuClassT<TensorArrayGpu8, int> tensordimension("1", "dir");
  assert(tensordimension.getId() == -1);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getDir() == "dir");
  assert(tensordimension.getNLabels() == 0);
}

void test_constructorNameAndLabelsGpuClassT()
{
  Eigen::Tensor<TensorArrayGpu8<int>, 1> labels(5);
  labels.setConstant(TensorArrayGpu8<int>({1, 1, 1, 1, 1, 1, 1, 1}));
  TensorDimensionGpuClassT<TensorArrayGpu8, int> tensordimension("1", "dir", labels);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getDir() == "dir");
  assert(tensordimension.getNLabels() == 5);
  assert(tensordimension.getLabels()(0).getTensorArray()(0) == 1);
  assert(tensordimension.getLabels()(4).getTensorArray()(0) == 1);
}

void test_gettersAndSettersGpuClassT()
{
  TensorDimensionGpuClassT<TensorArrayGpu8, int> tensordimension;
  // Check defaults
  assert(tensordimension.getId() == -1);
  assert(tensordimension.getName() == "");
  assert(tensordimension.getDir() == "");
  assert(tensordimension.getNLabels() == 0);

  // Check getters/setters
  tensordimension.setId(1);
  tensordimension.setName("1");
  tensordimension.setDir("dir");
  Eigen::Tensor<TensorArrayGpu8<int>, 1> labels(5);
  labels.setConstant(TensorArrayGpu8<int>({ 1, 1, 1, 1, 1, 1, 1, 1 }));
  tensordimension.setLabels(labels);

  assert(tensordimension.getId() == 1);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getDir() == "dir");
  assert(tensordimension.getNLabels() == 5);
  assert(tensordimension.getLabels()(0).getTensorArray()(0) == 1);
  assert(tensordimension.getLabels()(4).getTensorArray()(0) == 1);
}

int main(int argc, char** argv)
{
  test_constructorGpuPrimitiveT();
  test_destructorGpuPrimitiveT();
  test_constructorNameGpuPrimitiveT();
  test_constructorNameAndLabelsGpuPrimitiveT();
  test_gettersAndSettersGpuPrimitiveT();

  test_constructorGpuClassT();
  test_destructorGpuClassT();
  test_constructorNameGpuClassT();
  test_constructorNameAndLabelsGpuClassT();
  test_gettersAndSettersGpuClassT();
  return 0;
}
#endif