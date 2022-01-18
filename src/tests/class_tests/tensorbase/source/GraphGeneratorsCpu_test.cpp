/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE GraphGenerators test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/core/GraphGeneratorsCpu.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(graphGeneratorsCpu)

BOOST_AUTO_TEST_CASE(kroneckerGraphGeneratorMakeKroneckerGraphCpu)
{
  // init the device
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);

  // test making the kronecker graph
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 2>> weights;
  KroneckerGraphGeneratorCpu<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(4, 8, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { std::pow(2, 4) * 8, 2 };
  BOOST_CHECK(indices->getDimensions() == indices_dims);
  Eigen::array<Eigen::Index, 2> weights_dims = { std::pow(2, 4) * 8, 1 };
  BOOST_CHECK(weights->getDimensions() == weights_dims);
  //std::cout << "indices\n"<< indices->getData() <<std::endl;
  //std::cout << "weights\n" << weights->getData() << std::endl;

  // test getting the node/link ids for the entire graph
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> link_ids;
  graph_generator.getNodeAndLinkIds(0, std::pow(2, 4) * 8, indices, node_ids, link_ids, device);
  Eigen::array<Eigen::Index, 1> link_ids_dims = { std::pow(2, 4) * 8 };
  BOOST_CHECK(link_ids->getDimensions() == link_ids_dims);
  for (int i = 0; i < link_ids_dims.at(0); ++i) {
    BOOST_CHECK_EQUAL(link_ids->getData()(i), i);
  }
  Eigen::array<Eigen::Index, 1> node_ids_dims = { std::pow(2, 4) * 8 };
  BOOST_CHECK(node_ids->getDimensions().at(0) <= std::pow(2, 4));
  //std::cout << "link_ids\n"<< link_ids->getData() <<std::endl;
  //std::cout << "node_ids\n" << node_ids->getData() << std::endl;

  // test getting the node/link ids for a subset
  node_ids.reset();
  link_ids.reset();
  graph_generator.getNodeAndLinkIds(8, 16, indices, node_ids, link_ids, device);
  link_ids_dims = Eigen::array<Eigen::Index, 1>({ 16 });
  BOOST_CHECK(link_ids->getDimensions() == link_ids_dims);
  for (int i = 0; i < link_ids_dims.at(0); ++i) {
    BOOST_CHECK_EQUAL(link_ids->getData()(i), 8 + i);
  }
  BOOST_CHECK(node_ids->getDimensions().at(0) <= 16);
  //std::cout << "link_ids\n"<< link_ids->getData() <<std::endl;
  //std::cout << "node_ids\n" << node_ids->getData() << std::endl;
}

BOOST_AUTO_TEST_CASE(BinaryTreeGraphGeneratorMakeBinaryTree)
{
  // init the device
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);
  const int depth = 3;
  const int n_nodes = std::pow(2, depth) - 1;
  const int n_links = std::pow(2, depth) - 2;

  // test making the Binary Tree graph
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 2>> weights;
  BinaryTreeGraphGeneratorCpu<int, float> graph_generator;
  graph_generator.makeBinaryTree(depth, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { n_links, 2 };
  BOOST_CHECK(indices->getDimensions() == indices_dims);
  std::vector<int> expected_in_nodes = { 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 };
  for (int i = 0; i < n_links; ++i) {
    BOOST_CHECK_EQUAL(indices->getData()(i, 0), expected_in_nodes.at(i));
    if (i % 2 == 0) BOOST_CHECK_EQUAL(indices->getData()(i, 1), expected_in_nodes.at(i) * 2 + 1);
    else BOOST_CHECK_EQUAL(indices->getData()(i, 1), expected_in_nodes.at(i) * 2 + 2);
  }
  Eigen::array<Eigen::Index, 2> weights_dims = { n_links, 1 };
  BOOST_CHECK(weights->getDimensions() == weights_dims);

  // test getting the node/link ids for the entire graph
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> link_ids;
  graph_generator.getNodeAndLinkIds(0, n_links, indices, node_ids, link_ids, device);
  BOOST_CHECK_EQUAL(link_ids->getDimensions().at(0), n_links);
  for (int i = 0; i < n_links; ++i) {
    BOOST_CHECK_EQUAL(link_ids->getData()(i), i);
  }
  BOOST_CHECK_EQUAL(node_ids->getDimensions().at(0), n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    BOOST_CHECK_EQUAL(node_ids->getData()(i), i);
  }

  // test getting the node/link ids for a subset
  node_ids.reset();
  link_ids.reset();
  graph_generator.getNodeAndLinkIds(2, 4, indices, node_ids, link_ids, device);
  BOOST_CHECK_EQUAL(link_ids->getDimensions().at(0), 4);
  for (int i = 0; i < 4; ++i) {
    BOOST_CHECK_EQUAL(link_ids->getData()(i), 2 + i);
  }
  BOOST_CHECK_EQUAL(node_ids->getDimensions().at(0), 6);
  for (int i = 0; i < 6; ++i) {
    BOOST_CHECK_EQUAL(node_ids->getData()(i), i + 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()