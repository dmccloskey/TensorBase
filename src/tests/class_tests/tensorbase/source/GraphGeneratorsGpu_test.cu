/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/core/GraphGeneratorsGpu.h>

using namespace TensorBase;
using namespace std;

void test_kroneckerGraphGeneratorMakeKroneckerGraphGpu()
{
  // init the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // test making the kronecker graph
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  KroneckerGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(4, 8, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { int(std::pow(2, 4) * 8), 2 };
  assert(indices->getDimensions() == indices_dims);
  Eigen::array<Eigen::Index, 2> weights_dims = { int(std::pow(2, 4) * 8), 1 };
  assert(weights->getDimensions() == weights_dims);

  // test getting the node/link ids for the entire graph
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> link_ids;
  graph_generator.getNodeAndLinkIds(0, std::pow(2, 4) * 8, indices, node_ids, link_ids, device);
  link_ids->syncHAndDData(device);
  node_ids->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::array<Eigen::Index, 1> link_ids_dims = { int(std::pow(2, 4) * 8) };
  assert(link_ids->getDimensions() == link_ids_dims);
  for (int i = 0; i < link_ids_dims.at(0); ++i) {
    assert(link_ids->getData()(i) == i);
  }
  Eigen::array<Eigen::Index, 1> node_ids_dims = { int(std::pow(2, 4) * 8) };
  assert(node_ids->getDimensions().at(0) <= std::pow(2, 4));

  // test getting the node/link ids
  node_ids.reset();
  link_ids.reset();
  graph_generator.getNodeAndLinkIds(8, 16, indices, node_ids, link_ids, device);
  link_ids->syncHAndDData(device);
  node_ids->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  link_ids_dims = Eigen::array<Eigen::Index, 1>({ 16 });
  assert(link_ids->getDimensions() == link_ids_dims);
  for (int i = 0; i < link_ids_dims.at(0); ++i) {
    assert(link_ids->getData()(i) == 8 + i);
  }
  assert(node_ids->getDimensions().at(0) <= 16);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  assert(cudaDeviceReset() == cudaSuccess);
  test_kroneckerGraphGeneratorMakeKroneckerGraphGpu();
  return 0;
}
#endif