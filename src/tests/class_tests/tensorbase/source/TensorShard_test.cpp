/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorShard1 test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorShard.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorShard1)

/*TensorShard DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorShardDefaultDevice)
{
  TensorShard* ptr = nullptr;
  TensorShard* nullPointer = nullptr;
	ptr = new TensorShard();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorShardDefaultDevice)
{
  TensorShard* ptr = nullptr;
	ptr = new TensorShard();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(getNShardsDefaultDevice)
{
  Eigen::array<Eigen::Index, 3> dims_small = { 1,2,3 };
  TensorShard tensorShard;
  BOOST_CHECK_EQUAL(tensorShard.getNShards(dims_small), 1);

  Eigen::array<Eigen::Index, 4> dims_medium = { 1000,1000,1000,1000 };
  BOOST_CHECK_EQUAL(tensorShard.getNShards(dims_medium), 1001);

  Eigen::array<Eigen::Index, 7> dims_large = { 1000,1000,1000,1000,1000,1000,1000 };
  BOOST_CHECK_EQUAL(tensorShard.getNShards(dims_large), -1);
}

BOOST_AUTO_TEST_SUITE_END()