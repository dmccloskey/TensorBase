/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorDimensionGpu.h>

using namespace TensorBase;
using namespace std;

/*TensorDimensionGpu Tests*/
void test_constructorGpu()
{
  TensorDimensionGpu<int>* ptr = nullptr;
  TensorDimensionGpu<int>* nullPointer = nullptr;
  ptr = new TensorDimensionGpu<int>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorDimensionGpu<int>* ptr = nullptr;
  ptr = new TensorDimensionGpu<int>();
  delete ptr;
}

void test_constructorNameGpu()
{
  TensorDimensionGpu<int> tensordimension("1");
  assert(tensordimension.getId() == -1);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getNLabels() == 0);
}

void test_constructorNameAndLabelsGpu()
{
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  TensorDimensionGpu<int> tensordimension("1", labels);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getNLabels() == 5);
  assert(tensordimension.getLabels()(0) == 1);
  assert(tensordimension.getLabels()(4) == 1);
}

void test_gettersAndSettersGpu()
{
  TensorDimensionGpu<int> tensordimension;
  // Check defaults
  assert(tensordimension.getId() == -1);
  assert(tensordimension.getName() == "");
  assert(tensordimension.getNLabels() == 0);

  // Check getters/setters
  tensordimension.setId(1);
  tensordimension.setName("1");
  Eigen::Tensor<int, 1> labels(5);
  labels.setConstant(1);
  tensordimension.setLabels(labels);

  assert(tensordimension.getId() == 1);
  assert(tensordimension.getName() == "1");
  assert(tensordimension.getNLabels() == 5);
  assert(tensordimension.getLabels()(0) == 1);
  assert(tensordimension.getLabels()(4) == 1);
}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_constructorNameGpu();
  test_constructorNameAndLabelsGpu();
  test_gettersAndSettersGpu();
  return 0;
}
#endif