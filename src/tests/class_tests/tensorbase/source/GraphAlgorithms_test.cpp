/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE GraphAlgorithms test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/core/GraphAlgorithmsDefaultDevice.h>
#include <TensorBase/core/GraphGeneratorsDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(graphAlgorithms)

BOOST_AUTO_TEST_CASE(indicesAndWeightsToAdjacencyMatrixDefaultDevice)
{
  // init the device
  Eigen::DefaultDevice device;

  // make the toy graph
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  BinaryTreeGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // test making the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixDefaultDevice<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { int(node_ids->getTensorSize()), int(node_ids->getTensorSize()) };
  BOOST_CHECK(adjacency->getDimensions() == indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    for (int j = 0; j < node_ids->getTensorSize(); ++j) {
      if ((j == 0 && i == 1) || (j == 0 && i == 2) || (j == 1 && i == 3) || (j == 1 && i == 4) ||
        (j == 2 && i == 5) || (j == 2 && i == 6)) {
        BOOST_CHECK(adjacency->getData()(i, j) > 0);
      }
      else {
        BOOST_CHECK_EQUAL(adjacency->getData()(i, j), 0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(breadthFirstSearchDefaultDevice)
{
  // init the device
  Eigen::DefaultDevice device;

  // make the toy graph
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  BinaryTreeGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // make the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixDefaultDevice<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);

  // test BFS
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> tree;
  BreadthFirstSearchDefaultDevice<int, float> breadth_first_search;
  breadth_first_search(0, node_ids, adjacency, tree, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { int(node_ids->getTensorSize()), int(node_ids->getTensorSize()) + 1 };
  BOOST_CHECK(tree->getDimensions() == indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    for (int j = 0; j < node_ids->getTensorSize() + 1; ++j) {
      if ((j == 0 && i == 0)) {
        BOOST_CHECK_EQUAL(tree->getData()(i, j), 1);
      }
      else if ((j == 1 && i == 1) || (j == 1 && i == 2) || (j == 2 && i == 3) || (j == 2 && i == 4) ||
        (j == 2 && i == 5) || (j == 2 && i == 6)) {
        BOOST_CHECK(tree->getData()(i, j) > 0);
      }
      else {
        BOOST_CHECK_EQUAL(tree->getData()(i, j), 0);
      }
    }
  }
  //std::cout << "adjacency\n" << adjacency->getData() << std::endl;
  //std::cout << "tree\n" << tree->getData() << std::endl;
}

BOOST_AUTO_TEST_CASE(singleSourceShortestPathDefaultDevice)
{
  // init the device
  Eigen::DefaultDevice device;

  // make the toy graph
  const int scale = 4;
  const int edge_factor = 8;
  const int n_links = std::pow(2, 4) * 8;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  KroneckerGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(scale, edge_factor, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // make the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixDefaultDevice<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);

  // run BFS
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> tree;
  BreadthFirstSearchDefaultDevice<int, float> breadth_first_search;
  breadth_first_search(0, node_ids, adjacency, tree, device);

  // test SSSP
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> path_lengths;
  SingleSourceShortestPathDefaultDevice<int, float> sssp;
  sssp(tree, path_lengths, device);
  Eigen::array<Eigen::Index, 1> indices_dims = { int(node_ids->getTensorSize()) };
  BOOST_CHECK(path_lengths->getDimensions() == indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    if (i == 0) {
      BOOST_CHECK_EQUAL(path_lengths->getData()(i), 1);
    }
    else {
      BOOST_CHECK(path_lengths->getData()(i) >= 0); // Values can be zero
    }
  }
  //std::cout << "adjacency\n" << adjacency->getData() << std::endl;
  //std::cout << "tree\n" << tree->getData() << std::endl;
  //std::cout << "path_lengths\n" << path_lengths->getData() << std::endl;
}

BOOST_AUTO_TEST_CASE(singleSourceShortestPath2DefaultDevice)
{
  // init the device
  Eigen::DefaultDevice device;

  // make the toy graph
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  BinaryTreeGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);

  // make the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixDefaultDevice<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);

  // run BFS
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> tree;
  BreadthFirstSearchDefaultDevice<int, float> breadth_first_search;
  breadth_first_search(0, node_ids, adjacency, tree, device);

  // test SSSP
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> path_lengths;
  SingleSourceShortestPathDefaultDevice<int, float> sssp;
  sssp(tree, path_lengths, device);
  Eigen::array<Eigen::Index, 1> indices_dims = { int(node_ids->getTensorSize()) };
  BOOST_CHECK(path_lengths->getDimensions() == indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    if (i == 0) {
      BOOST_CHECK_EQUAL(path_lengths->getData()(i), 1);
    }
    else {
      BOOST_CHECK(path_lengths->getData()(i) > 0);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()