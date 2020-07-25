/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE GraphGenerators test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/core/GraphGeneratorsDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(graphGenerators)

BOOST_AUTO_TEST_CASE(kroneckerGraphGeneratorMakeKroneckerGraph)
{
  // init the device
  Eigen::DefaultDevice device;

  // test making the kronecker graph
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  KroneckerGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(4, 8, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { std::pow(2, 4) * 8, 2 };
  BOOST_CHECK(indices->getDimensions() == indices_dims);
  Eigen::array<Eigen::Index, 2> weights_dims = { std::pow(2, 4) * 8, 1 };
  BOOST_CHECK(weights->getDimensions() == weights_dims);
  //std::cout << "indices\n"<< indices->getData() <<std::endl;
  //std::cout << "weights\n" << weights->getData() << std::endl;

  // test getting the node/link ids for the entire graph
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> node_ids;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> link_ids;
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
  BOOST_CHECK(node_ids->getDimensions().at(0) <= 8);
  //std::cout << "link_ids\n"<< link_ids->getData() <<std::endl;
  //std::cout << "node_ids\n" << node_ids->getData() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()