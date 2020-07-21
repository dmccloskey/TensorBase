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
  //KroneckerGraphGenerator<int, float, Eigen::DefaultDevice>::makeKroneckerGraph(26, 16, indices, weights, device);
  KroneckerGraphGeneratorDefaultDevice<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(2, 2, indices, weights, device);
  std::cout << "indices\n" << indices->getData() << std::endl;
  std::cout << "weights\n" << weights->getData() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()