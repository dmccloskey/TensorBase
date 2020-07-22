/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE GraphGenerators test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/core/GraphGeneratorsDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(graphGenerators)

BOOST_AUTO_TEST_CASE(kroneckerGraphGeneratorMakeKroneckerGraph)
{
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>> weights;
  Eigen::DefaultDevice device;
  KroneckerGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(4, 8, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { std::pow(2, 4) * 8, 2 };
  BOOST_CHECK(indices->getDimensions() == indices_dims);
  Eigen::array<Eigen::Index, 2> weights_dims = { std::pow(2, 4) * 8, 1 };
  BOOST_CHECK(weights->getDimensions() == weights_dims);
  //std::cout << "indices\n"<< indices->getData() <<std::endl;
  //std::cout << "weights\n" << weights->getData() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()