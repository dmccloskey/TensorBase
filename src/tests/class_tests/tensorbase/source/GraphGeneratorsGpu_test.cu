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

void test_BinaryTreeGraphGeneratorMakeBinaryTree()
{
  // init the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;

  // test making the Binary Tree graph
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  BinaryTreeGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  indices->syncHAndDData(device);
  weights->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  Eigen::array<Eigen::Index, 2> indices_dims = { n_links, 2 };
  assert(indices->getDimensions() == indices_dims);
  std::vector<int> expected_in_nodes = { 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 };
  for (int i = 0; i < n_links; ++i) {
    assert(indices->getData()(i, 0) == expected_in_nodes.at(i));
    if (i % 2 == 0) assert(indices->getData()(i, 1) == expected_in_nodes.at(i) * 2 + 1);
    else assert(indices->getData()(i, 1) == expected_in_nodes.at(i) * 2 + 2);
  }
  Eigen::array<Eigen::Index, 2> weights_dims = { n_links, 1 };
  assert(weights->getDimensions() == weights_dims);

  // test getting the node/link ids for the entire graph
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> link_ids;
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);
  node_ids->syncHAndDData(device);
  link_ids->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(link_ids->getDimensions().at(0) == n_links);
  for (int i = 0; i < n_links; ++i) {
    assert(link_ids->getData()(i) == i);
  }
  assert(node_ids->getDimensions().at(0) == n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    assert(node_ids->getData()(i) == i);
  }

  // test getting the node/link ids for a subset
  node_ids.reset();
  link_ids.reset();
  graph_generator.getNodeAndLinkIds(2, 4, indices, node_ids, link_ids, device);
  node_ids->syncHAndDData(device);
  link_ids->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(link_ids->getDimensions().at(0) == 4);
  for (int i = 0; i < 4; ++i) {
    assert(link_ids->getData()(i) == 2 + i);
  }
  assert(node_ids->getDimensions().at(0) == 6);
  for (int i = 0; i < 6; ++i) {
    assert(node_ids->getData()(i) == i + 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  assert(cudaDeviceReset() == cudaSuccess);
  test_kroneckerGraphGeneratorMakeKroneckerGraphGpu();
  test_BinaryTreeGraphGeneratorMakeBinaryTree();
  return 0;
}
#endif