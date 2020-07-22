/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE GraphGenerators test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/core/GraphGeneratorsCpu.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(graphGeneratorsCpu)

BOOST_AUTO_TEST_CASE(kroneckerGraphGeneratorMakeKroneckerGraphCpu)
{
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices;
  std::shared_ptr<TensorData<float, Eigen::ThreadPoolDevice, 2>> weights;
  Eigen::ThreadPool pool(1);  Eigen::ThreadPoolDevice device(&pool, 2);
  KroneckerGraphGeneratorCpu<int, float> graph_generator;
  graph_generator.makeKroneckerGraph(4, 8, indices, weights, device);
  Eigen::array<Eigen::Index, 2> indices_dims = { std::pow(2, 4) * 8, 2 };
  BOOST_CHECK(indices->getDimensions() == indices_dims);
  Eigen::array<Eigen::Index, 2> weights_dims = { std::pow(2, 4) * 8, 1 };
  BOOST_CHECK(weights->getDimensions() == weights_dims);
}

BOOST_AUTO_TEST_SUITE_END()