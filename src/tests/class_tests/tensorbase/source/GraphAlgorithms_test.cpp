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

  // test making the Binary Tree graph
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> link_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  BinaryTreeGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  graph_generator.getNodeAndLinkIds(0, std::pow(2, depth), indices, node_ids, link_ids, device);

  // test making the adjacency matrix
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> adjacency;
  IndicesAndWeightsToAdjacencyMatrixDefaultDevice<int, float> to_adjacency;
  to_adjacency(node_ids, indices, weights, adjacency, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { int(node_ids->getTensorSize()), int(node_ids->getTensorSize()) };
  BOOST_CHECK(adjacency->getDimensions() == indices_dims);
  for (int i = 0; i < node_ids->getTensorSize(); ++i) {
    for (int j = 0; j < node_ids->getTensorSize(); ++j) {
      if ((j == 0 && i == 1) || (j == 0 && i == 2) || (j == 1 && i == 3) || (j == 1 && i == 4) ||
        (j == 2 && i == 5) || (j == 2 && i == 6) || (j == 3 && i == 7) || (j == 3 && i == 8)) {
        BOOST_CHECK_GT(adjacency->getData()(i, j), 0);
      }
      else {
        BOOST_CHECK_EQUAL(adjacency->getData()(i, j), 0);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()