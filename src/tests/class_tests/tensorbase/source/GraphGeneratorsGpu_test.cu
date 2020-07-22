/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/core/GraphGeneratorsGpu.h>

using namespace TensorBase;
using namespace std;

void test_kroneckerGraphGeneratorMakeKroneckerGraphGpu()
{
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);
  KroneckerGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(4, 8, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { int(std::pow(2, 4) * 8), 2 };
  assert(indices->getDimensions() == indices_dims);
  Eigen::array<Eigen::Index, 2> weights_dims = { int(std::pow(2, 4) * 8), 1 };
  assert(weights->getDimensions() == weights_dims);
}

int main(int argc, char** argv)
{
  assert(cudaDeviceReset() == cudaSuccess);
  test_kroneckerGraphGeneratorMakeKroneckerGraphGpu();
  return 0;
}
#endif