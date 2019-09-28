/**TODO:  Add copyright*/


#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorArrayGpu.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

using namespace TensorBase;
using namespace std;

void test_constructorGpu() 
{
	TensorArrayGpu8<float>* ptr = nullptr;
	TensorArrayGpu8<float>* nullPointer = nullptr;
	ptr = new TensorArrayGpu8<float>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorArrayGpu8<float>* ptr = nullptr;
	ptr = new TensorArrayGpu8<float>();
  delete ptr;
}

void test_gettersAndSettersGpu()
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(8);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8 });
  TensorArrayGpu8<float> tensorArrayFloat1(same_equal_float_1);
  assert(tensorArrayFloat1.getArraySize() == 8);
  assert(tensorArrayFloat1.getTensorArray()(0) == 1);
  assert(tensorArrayFloat1.getTensorArray()(1) == 2);
  assert(tensorArrayFloat1.getTensorArray()(2) == 3);
  assert(tensorArrayFloat1.getTensorArray()(3) == 4);
  assert(tensorArrayFloat1.getTensorArray()(4) == 5);
  assert(tensorArrayFloat1.getTensorArray()(5) == 6);
  assert(tensorArrayFloat1.getTensorArray()(6) == 7);
  assert(tensorArrayFloat1.getTensorArray()(7) == 8);
  assert(tensorArrayFloat1.at(0) == 1);
  assert(tensorArrayFloat1.at(1) == 2);
  assert(tensorArrayFloat1.at(2) == 3);
  assert(tensorArrayFloat1.at(3) == 4);
  assert(tensorArrayFloat1.at(4) == 5);
  assert(tensorArrayFloat1.at(5) == 6);
  assert(tensorArrayFloat1.at(6) == 7);
  assert(tensorArrayFloat1.at(7) == 8);

  // Check same and equal length char
  TensorArrayGpu8<char> tensorArrayChar1({ '1','2','3','4','5','6','7','8' });
  assert(tensorArrayChar1.getArraySize() == 8);
  assert(tensorArrayChar1.getTensorArray()(0) == '1');
  assert(tensorArrayChar1.getTensorArray()(1) == '2');
  assert(tensorArrayChar1.getTensorArray()(2) == '3');
  assert(tensorArrayChar1.getTensorArray()(3) == '4');
  assert(tensorArrayChar1.getTensorArray()(4) == '5');
  assert(tensorArrayChar1.getTensorArray()(5) == '6');
  assert(tensorArrayChar1.getTensorArray()(6) == '7');
  assert(tensorArrayChar1.getTensorArray()(7) == '8');
  assert(tensorArrayChar1.at(0) == '1');
  assert(tensorArrayChar1.at(1) == '2');
  assert(tensorArrayChar1.at(2) == '3');
  assert(tensorArrayChar1.at(3) == '4');
  assert(tensorArrayChar1.at(4) == '5');
  assert(tensorArrayChar1.at(5) == '6');
  assert(tensorArrayChar1.at(6) == '7');
  assert(tensorArrayChar1.at(7) == '8');

  TensorArrayGpu8<char> tensorArrayChar2({ '1','2','3','4','5','6' });
  assert(tensorArrayChar2.getArraySize() == 8);
  assert(tensorArrayChar2.getTensorArray()(0) == '1');
  assert(tensorArrayChar2.getTensorArray()(1) == '2');
  assert(tensorArrayChar2.getTensorArray()(2) == '3');
  assert(tensorArrayChar2.getTensorArray()(3) == '4');
  assert(tensorArrayChar2.getTensorArray()(4) == '5');
  assert(tensorArrayChar2.getTensorArray()(5) == '6');
  assert(tensorArrayChar2.getTensorArray()(6) == '\0');
  assert(tensorArrayChar2.getTensorArray()(7) == '\0');
  assert(tensorArrayChar2.at(0) == '1');
  assert(tensorArrayChar2.at(1) == '2');
  assert(tensorArrayChar2.at(2) == '3');
  assert(tensorArrayChar2.at(3) == '4');
  assert(tensorArrayChar2.at(4) == '5');
  assert(tensorArrayChar2.at(5) == '6');
  assert(tensorArrayChar2.at(6) == '\0');
  assert(tensorArrayChar2.at(7) == '\0');

  // Check same and equal length char
  TensorArrayGpu8<char> tensorArrayString1("12345678");
  assert(tensorArrayString1.getArraySize() == 8);
  assert(tensorArrayString1.getTensorArray()(0) == '1');
  assert(tensorArrayString1.getTensorArray()(1) == '2');
  assert(tensorArrayString1.getTensorArray()(2) == '3');
  assert(tensorArrayString1.getTensorArray()(3) == '4');
  assert(tensorArrayString1.getTensorArray()(4) == '5');
  assert(tensorArrayString1.getTensorArray()(5) == '6');
  assert(tensorArrayString1.getTensorArray()(6) == '7');
  assert(tensorArrayString1.getTensorArray()(7) == '8');
  assert(tensorArrayString1.at(0) == '1');
  assert(tensorArrayString1.at(1) == '2');
  assert(tensorArrayString1.at(2) == '3');
  assert(tensorArrayString1.at(3) == '4');
  assert(tensorArrayString1.at(4) == '5');
  assert(tensorArrayString1.at(5) == '6');
  assert(tensorArrayString1.at(6) == '7');
  assert(tensorArrayString1.at(7) == '8');
}

void test_getTensorArrayAsStringGpu()
{
  TensorArrayGpu8<int> tensorArrayInt1({ 1,2,3,4,5,6,7,8 });
  // Check << operator
  std::ostringstream os;
  os << tensorArrayInt1;
  assert(std::string(os.str()) == "12345678");

  // Check getter
  assert(tensorArrayInt1.getTensorArrayAsString() == "12345678");
}

void test_comparisonGpu()
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(8);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8 });
  TensorArrayGpu8<float> tensorArrayFloat1(same_equal_float_1);
  Eigen::Tensor<float, 1> same_equal_float_2(8);
  same_equal_float_2.setValues({ 1,2,3,4,5,6,7,8 });
  TensorArrayGpu8<float> tensorArrayFloat2(same_equal_float_2);
  assert(tensorArrayFloat1 == tensorArrayFloat2);
  assert(!(tensorArrayFloat1 != tensorArrayFloat2));
  assert(!(tensorArrayFloat1 < tensorArrayFloat2));
  assert(!(tensorArrayFloat1 > tensorArrayFloat2));
  assert(tensorArrayFloat1 <= tensorArrayFloat2);
  assert(tensorArrayFloat1 >= tensorArrayFloat2);

  // Check different and equal length float
  Eigen::Tensor<float, 1> same_equal_float_3(8);
  same_equal_float_3.setValues({ 1,2,0,4,5,6,7,8 });
  TensorArrayGpu8<float> tensorArrayFloat3(same_equal_float_3);
  assert(!(tensorArrayFloat1 == tensorArrayFloat3));
  assert(tensorArrayFloat1 != tensorArrayFloat3);
  assert(!(tensorArrayFloat1 < tensorArrayFloat3));
  assert(tensorArrayFloat1 > tensorArrayFloat3);
  assert(!(tensorArrayFloat1 <= tensorArrayFloat3));
  assert(tensorArrayFloat1 >= tensorArrayFloat3);

  // Check same and equal length char
  Eigen::Tensor<char, 1> same_equal_char_1(8);
  same_equal_char_1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar1(same_equal_char_1);
  Eigen::Tensor<char, 1> same_equal_char_2(8);
  same_equal_char_2.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar2(same_equal_char_2);
  assert(tensorArrayChar1 == tensorArrayChar2);
  assert(!(tensorArrayChar1 != tensorArrayChar2));
  assert(!(tensorArrayChar1 < tensorArrayChar2));
  assert(!(tensorArrayChar1 > tensorArrayChar2));
  assert(tensorArrayChar1 <= tensorArrayChar2);
  assert(tensorArrayChar1 >= tensorArrayChar2);

  // Check different and equal length char
  Eigen::Tensor<char, 1> same_equal_char_3(8);
  same_equal_char_3.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar3(same_equal_char_3);
  assert(!(tensorArrayChar1 == tensorArrayChar3));
  assert(tensorArrayChar1 != tensorArrayChar3);
  assert(!(tensorArrayChar1 < tensorArrayChar3));
  assert(tensorArrayChar1 > tensorArrayChar3);
  assert(!(tensorArrayChar1 <= tensorArrayChar3));
  assert(tensorArrayChar1 >= tensorArrayChar3);
}

void test_tensorAssignmentGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Create the string arrays
  Eigen::Tensor<char, 1> char1(8);
  char1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(8);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(8);
  char3.setValues({ 'x', 'y', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  size_t bytes = 3 * sizeof(TensorArrayGpu8<char>);
  TensorArrayGpu8<char>* h_in1;
  TensorArrayGpu8<char>* h_out1;
  TensorArrayGpu8<char>* d_in1;
  TensorArrayGpu8<char>* d_out1;
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_out1), bytes) == cudaSuccess);

  // Copy from the Cpu to the Gpu
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> gpu_in1(d_in1, 3);

  // Tensor copy
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> gpu_out1(d_out1, 3);
  gpu_out1.device(device) = gpu_in1;

  // Tensor compare
  device.memcpyDeviceToHost(h_out1, d_out1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> out1(h_out1, 3);
  assert(out1(0) == tensorArrayChar1);
  assert(out1(1) == tensorArrayChar2);
  assert(out1(2) == tensorArrayChar3);

  // Cleanup
  assert(cudaFree(d_in1) == cudaSuccess);
  assert(cudaFree(d_out1) == cudaSuccess);
  assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_out1) == cudaSuccess);
}

void test_tensorComparisonGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Create the string arrays
  Eigen::Tensor<char, 1> char1(8);
  char1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(8);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(8);
  char3.setValues({ 'x', 'y', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  size_t bytes = 3 * sizeof(TensorArrayGpu8<char>);
  TensorArrayGpu8<char>* h_in1;
  TensorArrayGpu8<char>* h_in2;
  TensorArrayGpu8<char>* d_in1;
  TensorArrayGpu8<char>* d_in2;
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_in2), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in2), bytes) == cudaSuccess);

  // Create the selection indices
  int* h_index_1;
  int* d_index_1;
  assert(cudaHostAlloc((void**)(&h_index_1), 3 * sizeof(int), cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_index_1), 3 * sizeof(int)) == cudaSuccess);

  // Copy form Cpu to Gpu
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> gpu_in1(d_in1, 3);

  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> in2(h_in2, 3);
  in2.setValues({ tensorArrayChar2 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in2, h_in2, bytes);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> gpu_in2(d_in2, 3);

  Eigen::TensorMap<Eigen::Tensor<int, 1>> index1(h_index_1, 3);
  index1.setZero();
  device.memcpyHostToDevice(d_index_1, h_index_1, 3 * sizeof(int));
  Eigen::TensorMap<Eigen::Tensor<int, 1>> gpu_index1(d_index_1, 3);

  // Compare
  gpu_index1.device(device) = (gpu_in1 == gpu_in2).select(gpu_index1.constant(1), gpu_index1.constant(0));

  // Tensor compare
  device.memcpyDeviceToHost(h_index_1, d_index_1, 3 * sizeof(int));
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<int, 1>> out1(h_index_1, 3);
  assert(out1(0) == 0);
  assert(out1(1) == 1);
  assert(out1(1) == 1);

  // Cleanup
  assert(cudaFree(d_in1) == cudaSuccess);
  assert(cudaFree(d_in2) == cudaSuccess);
  assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_in2) == cudaSuccess);

  assert(cudaFree(d_index_1) == cudaSuccess);
  assert(cudaFreeHost(h_index_1) == cudaSuccess);
}

void test_tensorSortGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Create the string arrays
  Eigen::Tensor<char, 1> char1(8);
  char1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(8);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(8);
  char3.setValues({ 'x', 'y', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu8<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  size_t bytes = 3 * sizeof(TensorArrayGpu8<char>);
  TensorArrayGpu8<char>* h_in1;
  TensorArrayGpu8<char>* h_out1;
  TensorArrayGpu8<char>* d_in1;
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);

  // Copy from the Cpu to the Gpu
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> gpu_in1(d_in1, 3);

  // Thrust sort
  isLessThanGpu sortOp(8);
  thrust::cuda::par.on(device.stream());
  thrust::device_ptr<TensorArrayGpu8<char>> d_ptr(gpu_in1.data());
  thrust::sort(d_ptr, d_ptr + 3, sortOp);

  // Tensor compare
  device.memcpyDeviceToHost(h_out1, d_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 1>> out1(h_out1, 3);
  assert(out1(0) == tensorArrayChar2);
  assert(out1(1) == tensorArrayChar1);
  assert(out1(2) == tensorArrayChar3);

  // Cleanup
  assert(cudaFree(d_in1) == cudaSuccess);
  assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_out1) == cudaSuccess);
}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_gettersAndSettersGpu();
  test_getTensorArrayAsStringGpu();
  test_comparisonGpu();
  test_tensorAssignmentGpu();
  test_tensorComparisonGpu();
  test_tensorSortGpu();
}
#endif