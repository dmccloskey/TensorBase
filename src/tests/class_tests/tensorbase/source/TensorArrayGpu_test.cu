/**TODO:  Add copyright*/


#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorArrayGpu.h>

using namespace TensorBase;
using namespace std;

void test_constructorGpu() 
{
	TensorArrayGpu<float>* ptr = nullptr;
	TensorArrayGpu<float>* nullPointer = nullptr;
	ptr = new TensorArrayGpu<float>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
	TensorArrayGpu<float>* ptr = nullptr;
	ptr = new TensorArrayGpu<float>();
  delete ptr;
}

void test_comparisonGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(5);
  same_equal_float_1.setValues({ 1,2,3,4,5 });
  TensorArrayGpu<float> tensorArrayFloat1(same_equal_float_1);
  tensorArrayFloat1.syncHAndDData(device);
  Eigen::Tensor<float, 1> same_equal_float_2(5);
  same_equal_float_2.setValues({ 1,2,3,4,5 });
	TensorArrayGpu<float> tensorArrayFloat2(same_equal_float_2);
  tensorArrayFloat2.syncHAndDData(device);
	assert(tensorArrayFloat1 == tensorArrayFloat2);
  assert(!(tensorArrayFloat1 != tensorArrayFloat2));
  assert(!(tensorArrayFloat1 < tensorArrayFloat2));
  assert(!(tensorArrayFloat1 > tensorArrayFloat2));
  assert(tensorArrayFloat1 <= tensorArrayFloat2);
  assert(tensorArrayFloat1 >= tensorArrayFloat2);

  // Check different and equal length float
  Eigen::Tensor<float, 1> same_equal_float_3(5);
  same_equal_float_3.setValues({ 1,2,0,4,5 });
  TensorArrayGpu<float> tensorArrayFloat3(same_equal_float_3);
  tensorArrayFloat3.syncHAndDData(device);
  assert(!(tensorArrayFloat1 == tensorArrayFloat3));
  assert(tensorArrayFloat1 != tensorArrayFloat3);
  assert(!(tensorArrayFloat1 < tensorArrayFloat3));
  assert(tensorArrayFloat1 > tensorArrayFloat3);
  assert(!(tensorArrayFloat1 <= tensorArrayFloat3));
  assert(tensorArrayFloat1 >= tensorArrayFloat3);

  // Check same and equal length char
  Eigen::Tensor<char, 1> same_equal_char_1(5);
  same_equal_char_1.setValues({ 'a', 'b', 'c', 'd', 'e' });
  TensorArrayGpu<char> tensorArrayChar1(same_equal_char_1);
  tensorArrayChar1.syncHAndDData(device);
  Eigen::Tensor<char, 1> same_equal_char_2(5);
  same_equal_char_2.setValues({ 'a', 'b', 'c', 'd', 'e' });
  TensorArrayGpu<char> tensorArrayChar2(same_equal_char_2);
  tensorArrayChar2.syncHAndDData(device);
  assert(tensorArrayChar1 == tensorArrayChar2);
  assert(!(tensorArrayChar1 != tensorArrayChar2));
  assert(!(tensorArrayChar1 < tensorArrayChar2));
  assert(!(tensorArrayChar1 > tensorArrayChar2));
  assert(tensorArrayChar1 <= tensorArrayChar2);
  assert(tensorArrayChar1 >= tensorArrayChar2);

  // Check different and unqeual length char
  Eigen::Tensor<char, 1> same_equal_char_3(4);
  same_equal_char_3.setValues({ 'a', 'b', 'a', 'd', 'e' });
  TensorArrayGpu<char> tensorArrayChar3(same_equal_char_3);
  tensorArrayChar3.syncHAndDData(device);
  assert(!(tensorArrayChar1 == tensorArrayChar3));
  assert(tensorArrayChar1 != tensorArrayChar3);
  assert(!(tensorArrayChar1 < tensorArrayChar3));
  assert(tensorArrayChar1 > tensorArrayChar3);
  assert(!(tensorArrayChar1 <= tensorArrayChar3));
  assert(tensorArrayChar1 >= tensorArrayChar3);

  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_comparisonGpu();
}
#endif