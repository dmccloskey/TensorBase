/**TODO:  Add copyright*/


#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorArrayGpu.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

using namespace TensorBase;
using namespace std;

/* TensorArray8 Tests
*/
void test_constructorTensorArray8Gpu() 
{
	TensorArrayGpu8<float>* ptr = nullptr;
	TensorArrayGpu8<float>* nullPointer = nullptr;
	ptr = new TensorArrayGpu8<float>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorTensorArray8Gpu()
{
  TensorArrayGpu8<float>* ptr = nullptr;
	ptr = new TensorArrayGpu8<float>();
  delete ptr;
}

void test_gettersAndSettersTensorArray8Gpu()
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

void test_getTensorArrayAsStringTensorArray8Gpu()
{
  TensorArrayGpu8<int> tensorArrayInt1({ 1,2,3,4,5,6,7,8 });
  // Check << operator
  std::ostringstream os;
  os << tensorArrayInt1;
  assert(std::string(os.str()) == "12345678");

  // Check getter
  assert(tensorArrayInt1.getTensorArrayAsString() == "12345678");
}

void test_comparisonTensorArray8Gpu()
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

void test_tensorAssignmentTensorArray8Gpu()
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

void test_tensorComparisonTensorArray8Gpu()
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

void test_tensorSortTensorArray8Gpu()
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
  isLessThanGpu8 sortOp(8);
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

/* TensorArrayGpu32 Tests
*/
void test_constructorTensorArray32Gpu()
{
  TensorArrayGpu32<float>* ptr = nullptr;
  TensorArrayGpu32<float>* nullPointer = nullptr;
  ptr = new TensorArrayGpu32<float>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorTensorArray32Gpu()
{
  TensorArrayGpu32<float>* ptr = nullptr;
  ptr = new TensorArrayGpu32<float>();
  delete ptr;
}

void test_gettersAndSettersTensorArray32Gpu()
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(32);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  TensorArrayGpu32<float> tensorArrayFloat1(same_equal_float_1);
  assert(tensorArrayFloat1.getArraySize() == 32);
  assert(tensorArrayFloat1.getTensorArray()(0) == 1);
  assert(tensorArrayFloat1.getTensorArray()(1) == 2);
  assert(tensorArrayFloat1.getTensorArray()(2) == 3);
  assert(tensorArrayFloat1.getTensorArray()(3) == 4);
  assert(tensorArrayFloat1.getTensorArray()(4) == 5);
  assert(tensorArrayFloat1.getTensorArray()(5) == 6);
  assert(tensorArrayFloat1.getTensorArray()(6) == 7);
  assert(tensorArrayFloat1.getTensorArray()(7) == 8);
  assert(tensorArrayFloat1.getTensorArray()(8) == 9);
  assert(tensorArrayFloat1.getTensorArray()(9) == 10);
  assert(tensorArrayFloat1.getTensorArray()(10) == 11);
  assert(tensorArrayFloat1.getTensorArray()(11) == 12);
  assert(tensorArrayFloat1.getTensorArray()(12) == 13);
  assert(tensorArrayFloat1.getTensorArray()(13) == 14);
  assert(tensorArrayFloat1.getTensorArray()(14) == 15);
  assert(tensorArrayFloat1.getTensorArray()(15) == 16);
  assert(tensorArrayFloat1.getTensorArray()(16) == 17);
  assert(tensorArrayFloat1.getTensorArray()(17) == 18);
  assert(tensorArrayFloat1.getTensorArray()(18) == 19);
  assert(tensorArrayFloat1.getTensorArray()(19) == 20);
  assert(tensorArrayFloat1.getTensorArray()(20) == 21);
  assert(tensorArrayFloat1.getTensorArray()(21) == 22);
  assert(tensorArrayFloat1.getTensorArray()(22) == 23);
  assert(tensorArrayFloat1.getTensorArray()(23) == 24);
  assert(tensorArrayFloat1.getTensorArray()(24) == 25);
  assert(tensorArrayFloat1.getTensorArray()(25) == 26);
  assert(tensorArrayFloat1.getTensorArray()(26) == 27);
  assert(tensorArrayFloat1.getTensorArray()(27) == 28);
  assert(tensorArrayFloat1.getTensorArray()(28) == 29);
  assert(tensorArrayFloat1.getTensorArray()(29) == 30);
  assert(tensorArrayFloat1.getTensorArray()(30) == 31);
  assert(tensorArrayFloat1.getTensorArray()(31) == 32);
  assert(tensorArrayFloat1.at(0) == 1);
  assert(tensorArrayFloat1.at(1) == 2);
  assert(tensorArrayFloat1.at(2) == 3);
  assert(tensorArrayFloat1.at(3) == 4);
  assert(tensorArrayFloat1.at(4) == 5);
  assert(tensorArrayFloat1.at(5) == 6);
  assert(tensorArrayFloat1.at(6) == 7);
  assert(tensorArrayFloat1.at(7) == 8);
  assert(tensorArrayFloat1.at(8) == 9);
  assert(tensorArrayFloat1.at(9) == 10);
  assert(tensorArrayFloat1.at(10) == 11);
  assert(tensorArrayFloat1.at(11) == 12);
  assert(tensorArrayFloat1.at(12) == 13);
  assert(tensorArrayFloat1.at(13) == 14);
  assert(tensorArrayFloat1.at(14) == 15);
  assert(tensorArrayFloat1.at(15) == 16);
  assert(tensorArrayFloat1.at(16) == 17);
  assert(tensorArrayFloat1.at(17) == 18);
  assert(tensorArrayFloat1.at(18) == 19);
  assert(tensorArrayFloat1.at(19) == 20);
  assert(tensorArrayFloat1.at(20) == 21);
  assert(tensorArrayFloat1.at(21) == 22);
  assert(tensorArrayFloat1.at(22) == 23);
  assert(tensorArrayFloat1.at(23) == 24);
  assert(tensorArrayFloat1.at(24) == 25);
  assert(tensorArrayFloat1.at(25) == 26);
  assert(tensorArrayFloat1.at(26) == 27);
  assert(tensorArrayFloat1.at(27) == 28);
  assert(tensorArrayFloat1.at(28) == 29);
  assert(tensorArrayFloat1.at(29) == 30);
  assert(tensorArrayFloat1.at(30) == 31);
  assert(tensorArrayFloat1.at(31) == 32);

  TensorArrayGpu32<float> tensorArrayFloat2({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  assert(tensorArrayFloat2.at(0) == 1);
  assert(tensorArrayFloat2.at(1) == 2);
  assert(tensorArrayFloat2.at(2) == 3);
  assert(tensorArrayFloat2.at(3) == 4);
  assert(tensorArrayFloat2.at(4) == 5);
  assert(tensorArrayFloat2.at(5) == 6);
  assert(tensorArrayFloat2.at(6) == 7);
  assert(tensorArrayFloat2.at(7) == 8);
  assert(tensorArrayFloat2.at(8) == 9);
  assert(tensorArrayFloat2.at(9) == 10);
  assert(tensorArrayFloat2.at(10) == 11);
  assert(tensorArrayFloat2.at(11) == 12);
  assert(tensorArrayFloat2.at(12) == 13);
  assert(tensorArrayFloat2.at(13) == 14);
  assert(tensorArrayFloat2.at(14) == 15);
  assert(tensorArrayFloat2.at(15) == 16);
  assert(tensorArrayFloat2.at(16) == 17);
  assert(tensorArrayFloat2.at(17) == 18);
  assert(tensorArrayFloat2.at(18) == 19);
  assert(tensorArrayFloat2.at(19) == 20);
  assert(tensorArrayFloat2.at(20) == 21);
  assert(tensorArrayFloat2.at(21) == 22);
  assert(tensorArrayFloat2.at(22) == 23);
  assert(tensorArrayFloat2.at(23) == 24);
  assert(tensorArrayFloat2.at(24) == 25);
  assert(tensorArrayFloat2.at(25) == 26);
  assert(tensorArrayFloat2.at(26) == 27);
  assert(tensorArrayFloat2.at(27) == 28);
  assert(tensorArrayFloat2.at(28) == 29);
  assert(tensorArrayFloat2.at(29) == 30);
  assert(tensorArrayFloat2.at(30) == 31);
  assert(tensorArrayFloat2.at(31) == 32);
}

void test_comparisonTensorArray32Gpu()
{
  // Check same and equal length float
  Eigen::Tensor<float, 1> same_equal_float_1(32);
  same_equal_float_1.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  TensorArrayGpu32<float> tensorArrayFloat1(same_equal_float_1);
  Eigen::Tensor<float, 1> same_equal_float_2(32);
  same_equal_float_2.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  TensorArrayGpu32<float> tensorArrayFloat2(same_equal_float_2);
  assert(tensorArrayFloat1 == tensorArrayFloat2);
  assert(!(tensorArrayFloat1 != tensorArrayFloat2));
  assert(!(tensorArrayFloat1 < tensorArrayFloat2));
  assert(!(tensorArrayFloat1 > tensorArrayFloat2));
  assert(tensorArrayFloat1 <= tensorArrayFloat2);
  assert(tensorArrayFloat1 >= tensorArrayFloat2);

  // Check different and equal length float
  Eigen::Tensor<float, 1> same_equal_float_3(32);
  same_equal_float_3.setValues({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,0,28,29,30,31,32 });
  TensorArrayGpu32<float> tensorArrayFloat3(same_equal_float_3);
  assert(!(tensorArrayFloat1 == tensorArrayFloat3));
  assert(tensorArrayFloat1 != tensorArrayFloat3);
  assert(!(tensorArrayFloat1 < tensorArrayFloat3));
  assert(tensorArrayFloat1 > tensorArrayFloat3);
  assert(!(tensorArrayFloat1 <= tensorArrayFloat3));
  assert(tensorArrayFloat1 >= tensorArrayFloat3);
}

void test_getTensorArrayAsStringTensorArray32Gpu()
{
  TensorArrayGpu32<int> tensorArrayInt1({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 });
  // Check << operator
  std::ostringstream os;
  os << tensorArrayInt1;
  assert(std::string(os.str()) == "1234567891011121314151617181920212223242526272829303132");

  // Check getter
  assert(tensorArrayInt1.getTensorArrayAsString() == "1234567891011121314151617181920212223242526272829303132");
}
void test_tensorAssignmentTensorArray32Gpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Create the string arrays
  Eigen::Tensor<char, 1> char1(32);
  char1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(32);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(32);
  char3.setValues({ 'x', 'y', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  size_t bytes = 3 * sizeof(TensorArrayGpu32<char>);
  TensorArrayGpu32<char>* h_in1;
  TensorArrayGpu32<char>* h_out1;
  TensorArrayGpu32<char>* d_in1;
  TensorArrayGpu32<char>* d_out1;
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_out1), bytes) == cudaSuccess);

  // Copy from the Cpu to the Gpu
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> gpu_in1(d_in1, 3);

  // Tensor copy
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> gpu_out1(d_out1, 3);
  gpu_out1.device(device) = gpu_in1;

  // Tensor compare
  device.memcpyDeviceToHost(h_out1, d_out1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> out1(h_out1, 3);
  assert(out1(0) == tensorArrayChar1);
  assert(out1(1) == tensorArrayChar2);
  assert(out1(2) == tensorArrayChar3);

  // Cleanup
  assert(cudaFree(d_in1) == cudaSuccess);
  assert(cudaFree(d_out1) == cudaSuccess);
  assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_out1) == cudaSuccess);
}

void test_tensorComparisonTensorArray32Gpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Create the string arrays
  Eigen::Tensor<char, 1> char1(32);
  char1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(32);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(32);
  char3.setValues({ 'x', 'y', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  size_t bytes = 3 * sizeof(TensorArrayGpu32<char>);
  TensorArrayGpu32<char>* h_in1;
  TensorArrayGpu32<char>* h_in2;
  TensorArrayGpu32<char>* d_in1;
  TensorArrayGpu32<char>* d_in2;
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
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> gpu_in1(d_in1, 3);

  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> in2(h_in2, 3);
  in2.setValues({ tensorArrayChar2 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in2, h_in2, bytes);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> gpu_in2(d_in2, 3);

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

void test_tensorSortTensorArray32Gpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Create the string arrays
  Eigen::Tensor<char, 1> char1(32);
  char1.setValues({ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(32);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(32);
  char3.setValues({ 'x', 'y', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'e', 'a', 'b', 'c', 'd', 'e', 'f', 'g', '\0' });
  TensorArrayGpu32<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  size_t bytes = 3 * sizeof(TensorArrayGpu32<char>);
  TensorArrayGpu32<char>* h_in1;
  TensorArrayGpu32<char>* h_out1;
  TensorArrayGpu32<char>* d_in1;
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);

  // Copy from the Cpu to the Gpu
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> gpu_in1(d_in1, 3);

  // Thrust sort
  isLessThanGpu32 sortOp(32);
  thrust::cuda::par.on(device.stream());
  thrust::device_ptr<TensorArrayGpu32<char>> d_ptr(gpu_in1.data());
  thrust::sort(d_ptr, d_ptr + 3, sortOp);

  // Tensor compare
  device.memcpyDeviceToHost(h_out1, d_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu32<char>, 1>> out1(h_out1, 3);
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
  test_constructorTensorArray8Gpu();
  test_destructorTensorArray8Gpu();
  test_gettersAndSettersTensorArray8Gpu();
  test_getTensorArrayAsStringTensorArray8Gpu();
  test_comparisonTensorArray8Gpu();
  test_tensorAssignmentTensorArray8Gpu();
  test_tensorComparisonTensorArray8Gpu();
  test_tensorSortTensorArray8Gpu();

  test_constructorTensorArray32Gpu();
  test_destructorTensorArray32Gpu();
  test_gettersAndSettersTensorArray32Gpu();
  test_comparisonTensorArray32Gpu();
  test_getTensorArrayAsStringTensorArray32Gpu();
  test_tensorAssignmentTensorArray32Gpu();
  test_tensorComparisonTensorArray32Gpu();
  test_tensorSortTensorArray32Gpu();
}
#endif