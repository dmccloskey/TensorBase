/**TODO:  Add copyright*/


#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorArrayGpu.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using namespace TensorBase;
using namespace std;

void test_constructorGpu() 
{
	TensorArray8<float>* ptr = nullptr;
	TensorArray8<float>* nullPointer = nullptr;
	ptr = new TensorArray8<float>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
	TensorArray8<float>* ptr = nullptr;
	ptr = new TensorArray8<float>();
  delete ptr;
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
  TensorArray8<char> tensorArrayChar1(char1);
  Eigen::Tensor<char, 1> char2(8);
  char2.setValues({ 'a', 'b', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar2(char2);
  Eigen::Tensor<char, 1> char3(8);
  char3.setValues({ 'x', 'y', 'a', 'd', 'e', 'f', 'g', '\0' });
  TensorArray8<char> tensorArrayChar3(char3);

  // Create the Tensor array of strings
  std::cout << "sizeof(TensorArray8<char>) " << sizeof(TensorArray8<char>) << std::endl;
  std::cout << "sizeof(char1) " << sizeof(tensorArrayChar1) << std::endl;

  size_t bytes = 3 * sizeof(TensorArray8<char>);
  TensorArray8<char>* h_in1;
  TensorArray8<char>* h_out1;
  TensorArray8<char>* d_in1;
  TensorArray8<char>* d_out1;
  int* h_index_1;
  int* d_index_1;
  assert(cudaHostAlloc((void**)(&h_in1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_out1), bytes, cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_in1), bytes) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_out1), bytes) == cudaSuccess);
  assert(cudaHostAlloc((void**)(&h_index_1), 3 * sizeof(int), cudaHostAllocDefault) == cudaSuccess);
  assert(cudaMalloc((void**)(&d_index_1), 3 * sizeof(int)) == cudaSuccess);

  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 1>> in1(h_in1, 3);
  in1.setValues({ tensorArrayChar1 , tensorArrayChar2, tensorArrayChar3 });
  device.memcpyHostToDevice(d_in1, h_in1, bytes);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 1>> gpu_in1(d_in1, 3);

  Eigen::TensorMap<Eigen::Tensor<int, 1>> index1(h_index_1, 3);
  index1.setZero();
  device.memcpyHostToDevice(d_index_1, h_index_1, 3 * sizeof(int));
  Eigen::TensorMap<Eigen::Tensor<int, 1>> gpu_index1(d_index_1, 3);

  index1 = (in1 == in1).select(index1.constant(1), index1.constant(0));
  //gpu_index1.device(device) = (gpu_in1 == gpu_in1).select(gpu_index1.constant(1), gpu_index1.constant(0));

  // Thrust sort
  isLessThanGpu sortOp(8);
  thrust::cuda::par.on(device.stream());
  thrust::sort(gpu_in1.data(), gpu_in1.data() + gpu_in1.size(), sortOp);
  //thrust::host_vector<TensorArray8<char>> h_vec;
  //h_vec.push_back(tensorArrayChar1);
  //h_vec.push_back(tensorArrayChar2); 
  //h_vec.push_back(tensorArrayChar3);
  //thrust::device_vector<TensorArray8<char>> d_vec = h_vec;
  //thrust::sort(d_vec.begin(), d_vec.end(), sortOp);

  // Tensor copy
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 1>> gpu_out1(d_out1, 3);
  gpu_out1.device(device) = gpu_in1;

  // Tensor compare
  device.memcpyDeviceToHost(h_out1, d_out1, bytes);
  device.memcpyDeviceToHost(h_index_1, d_index_1, 3 * sizeof(int));
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(cudaStreamDestroy(stream) == cudaSuccess);
  Eigen::TensorMap<Eigen::Tensor<TensorArray8<char>, 1>> out1(h_out1, 3);
  std::cout << "array0" << out1(0).getTensorArray() << std::endl;
  std::cout << "array1" << out1(1).getTensorArray() << std::endl;
  std::cout << "array2" << out1(2).getTensorArray() << std::endl;
  std::cout << index1 << std::endl;

  // Cleanup
  assert(cudaFree(d_in1) == cudaSuccess);
  assert(cudaFree(d_out1) == cudaSuccess);
  assert(cudaFreeHost(h_in1) == cudaSuccess);
  assert(cudaFreeHost(h_out1) == cudaSuccess);
}

void test_comparisonGpu()
{
 // // Initialize the device
 // cudaStream_t stream;
 // assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
 // Eigen::GpuStreamDevice stream_device(&stream, 0);
 // Eigen::GpuDevice device(&stream_device);

 // // Check same and equal length float
 // Eigen::Tensor<float, 1> same_equal_float_1(5);
 // same_equal_float_1.setValues({ 1,2,3,4,5 });
 // TensorArray8<float> tensorArrayFloat1(same_equal_float_1);
 // tensorArrayFloat1.syncHAndDData(device);
 // Eigen::Tensor<float, 1> same_equal_float_2(5);
 // same_equal_float_2.setValues({ 1,2,3,4,5 });
	//TensorArray8<float> tensorArrayFloat2(same_equal_float_2);
 // tensorArrayFloat2.syncHAndDData(device);
	//assert(tensorArrayFloat1 == tensorArrayFloat2);
 // assert(!(tensorArrayFloat1 != tensorArrayFloat2));
 // assert(!(tensorArrayFloat1 < tensorArrayFloat2));
 // assert(!(tensorArrayFloat1 > tensorArrayFloat2));
 // assert(tensorArrayFloat1 <= tensorArrayFloat2);
 // assert(tensorArrayFloat1 >= tensorArrayFloat2);

 // // Check different and equal length float
 // Eigen::Tensor<float, 1> same_equal_float_3(5);
 // same_equal_float_3.setValues({ 1,2,0,4,5 });
 // TensorArray8<float> tensorArrayFloat3(same_equal_float_3);
 // tensorArrayFloat3.syncHAndDData(device);
 // assert(!(tensorArrayFloat1 == tensorArrayFloat3));
 // assert(tensorArrayFloat1 != tensorArrayFloat3);
 // assert(!(tensorArrayFloat1 < tensorArrayFloat3));
 // assert(tensorArrayFloat1 > tensorArrayFloat3);
 // assert(!(tensorArrayFloat1 <= tensorArrayFloat3));
 // assert(tensorArrayFloat1 >= tensorArrayFloat3);

 // // Check same and equal length char
 // Eigen::Tensor<char, 1> same_equal_char_1(5);
 // same_equal_char_1.setValues({ 'a', 'b', 'c', 'd', 'e' });
 // TensorArray8<char> tensorArrayChar1(same_equal_char_1);
 // tensorArrayChar1.syncHAndDData(device);
 // Eigen::Tensor<char, 1> same_equal_char_2(5);
 // same_equal_char_2.setValues({ 'a', 'b', 'c', 'd', 'e' });
 // TensorArray8<char> tensorArrayChar2(same_equal_char_2);
 // tensorArrayChar2.syncHAndDData(device);
 // assert(tensorArrayChar1 == tensorArrayChar2);
 // assert(!(tensorArrayChar1 != tensorArrayChar2));
 // assert(!(tensorArrayChar1 < tensorArrayChar2));
 // assert(!(tensorArrayChar1 > tensorArrayChar2));
 // assert(tensorArrayChar1 <= tensorArrayChar2);
 // assert(tensorArrayChar1 >= tensorArrayChar2);

 // // Check different and unqeual length char
 // Eigen::Tensor<char, 1> same_equal_char_3(4);
 // same_equal_char_3.setValues({ 'a', 'b', 'a', 'd', 'e' });
 // TensorArray8<char> tensorArrayChar3(same_equal_char_3);
 // tensorArrayChar3.syncHAndDData(device);
 // assert(!(tensorArrayChar1 == tensorArrayChar3));
 // assert(tensorArrayChar1 != tensorArrayChar3);
 // assert(!(tensorArrayChar1 < tensorArrayChar3));
 // assert(tensorArrayChar1 > tensorArrayChar3);
 // assert(!(tensorArrayChar1 <= tensorArrayChar3));
 // assert(tensorArrayChar1 >= tensorArrayChar3);

 // assert(cudaStreamSynchronize(stream) == cudaSuccess);
 // assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_tensorComparisonGpu();
  test_comparisonGpu();
}
#endif