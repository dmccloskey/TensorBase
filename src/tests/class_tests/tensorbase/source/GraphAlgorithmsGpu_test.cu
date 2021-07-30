/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/core/GraphAlgorithmsGpu.h>
#include <TensorBase/core/GraphGeneratorsGpu.h>

using namespace TensorBase;
using namespace std;

void test_indicesAndWeightsToAdjacencyMatrixGpu()
{
  // init the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // make the toy graph
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  BinaryTreeGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // test making the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixGpu<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);
  adjacency->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  Eigen::array<Eigen::Index, 2> indices_dims = { int(node_ids->getTensorSize()), int(node_ids->getTensorSize()) };
  gpuCheckEqualNoLhsRhsPrint(adjacency->getDimensions(), indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    for (int j = 0; j < node_ids->getTensorSize(); ++j) {
      if ((j == 0 && i == 1) || (j == 0 && i == 2) || (j == 1 && i == 3) || (j == 1 && i == 4) ||
        (j == 2 && i == 5) || (j == 2 && i == 6)) {
        gpuCheckGreaterThan(adjacency->getData()(i, j), 0);
      }
      else {
        gpuCheckEqual(adjacency->getData()(i, j), 0);
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_breadthFirstSearchGpu()
{
  // init the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // make the toy graph
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  BinaryTreeGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // make the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixGpu<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);

  // test BFS
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> tree;
  BreadthFirstSearchGpu<int, float> breadth_first_search;
  breadth_first_search(0, node_ids, adjacency, tree, device);
  tree->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  Eigen::array<Eigen::Index, 2> indices_dims = { int(node_ids->getTensorSize()), int(node_ids->getTensorSize()) + 1 };
  gpuCheckEqualNoLhsRhsPrint(tree->getDimensions(), indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    for (int j = 0; j < node_ids->getTensorSize() + 1; ++j) {
      if ((j == 0 && i == 0)) {
        gpuCheckEqual(tree->getData()(i, j), 1);
      }
      else if ((j == 1 && i == 1) || (j == 1 && i == 2) || (j == 2 && i == 3) || (j == 2 && i == 4) ||
        (j == 2 && i == 5) || (j == 2 && i == 6)) {
        gpuCheckGreaterThan(tree->getData()(i, j), 0);
      }
      else {
        gpuCheckEqual(tree->getData()(i, j), 0);
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_singleSourceShortestPathGpu()
{
  // init the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // make the toy graph
  const int scale = 4;
  const int edge_factor = 8;
  const int n_links = std::pow(2, 4) * 8;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  KroneckerGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(scale, edge_factor, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // make the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixGpu<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);

  // run BFS
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> tree;
  BreadthFirstSearchGpu<int, float> breadth_first_search;
  breadth_first_search(0, node_ids, adjacency, tree, device);

  // test SSSP
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> path_lengths;
  SingleSourceShortestPathGpu<int, float> sssp;
  sssp(tree, path_lengths, device);
  path_lengths->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  Eigen::array<Eigen::Index, 1> indices_dims = { int(node_ids->getTensorSize()) };
  gpuCheckEqualNoLhsRhsPrint(path_lengths->getDimensions(), indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    if (i == 0) {
      gpuCheckEqual(path_lengths->getData()(i), 1);
    }
    else {
      gpuCheckGreaterThan(path_lengths->getData()(i), 0 - 1e-9); // Values can be zero
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_singleSourceShortestPath2Gpu()
{
  // init the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // make the toy graph
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> weights;
  BinaryTreeGraphGeneratorGpu<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // make the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixGpu<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);

  // run BFS
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 2>> tree;
  BreadthFirstSearchGpu<int, float> breadth_first_search;
  breadth_first_search(0, node_ids, adjacency, tree, device);

  // test SSSP
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> path_lengths;
  SingleSourceShortestPathGpu<int, float> sssp;
  sssp(tree, path_lengths, device);
  path_lengths->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  Eigen::array<Eigen::Index, 1> indices_dims = { int(node_ids->getTensorSize()) };
  gpuCheckEqualNoLhsRhsPrint(path_lengths->getDimensions(), indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    if (i == 0) {
      gpuCheckEqual(path_lengths->getData()(i), 1);
    }
    else {
      gpuCheckGreaterThan(path_lengths->getData()(i), 0);
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

int main(int argc, char** argv)
{
  gpuErrchk(cudaDeviceReset());
  test_indicesAndWeightsToAdjacencyMatrixGpu();
  test_breadthFirstSearchGpu();
  test_singleSourceShortestPathGpu();
  test_singleSourceShortestPath2Gpu();
  return 0;
}
#endif