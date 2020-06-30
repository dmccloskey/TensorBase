/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorShard1 test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorShard.h>
#include <TensorBase/ml/TensorDataDefaultDevice.h>

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

BOOST_AUTO_TEST_CASE(getNMaxShardsDefaultDevice)
{
  Eigen::array<Eigen::Index, 3> dims_small = { 1,2,3 };
  BOOST_CHECK_EQUAL(TensorShard::getNMaxShards(dims_small), 1);

  Eigen::array<Eigen::Index, 4> dims_medium = { 1000,1000,1000,1000 };
  BOOST_CHECK_EQUAL(TensorShard::getNMaxShards(dims_medium), 1001);

  Eigen::array<Eigen::Index, 7> dims_large = { 1000,1000,1000,1000,1000,1000,1000 };
  BOOST_CHECK_EQUAL(TensorShard::getNMaxShards(dims_large), -1);
}

BOOST_AUTO_TEST_CASE(checkShardSpansDefaultDevice)
{
  Eigen::array<Eigen::Index, 4> dimensions_medium = { 0, 1000, 3, 0 };
  std::map<std::string, int> axes_to_dims_medium = { {"1", 0}, {"2", 1}, {"3", 2}, {"4", 3} };
  std::map<std::string, int> shard_span_medium = { {"1", 10}, {"2", 2e9}, {"3", -1}, {"4", 0} };
  TensorShard::checkShardSpans<4>(axes_to_dims_medium, dimensions_medium, shard_span_medium);
  BOOST_CHECK_EQUAL(shard_span_medium.at("1"), 10);
  BOOST_CHECK_EQUAL(shard_span_medium.at("2"), 1000);
  BOOST_CHECK_EQUAL(shard_span_medium.at("3"), 3);
  BOOST_CHECK_EQUAL(shard_span_medium.at("4"), 1e9);
}

BOOST_AUTO_TEST_CASE(getDefaultMaxDimensionsDefaultDevice)
{
  std::map<std::string, int> axes_to_dims_small = { {"1", 0}, {"2", 1}, {"3", 2} };
  std::map<std::string, int> shard_span_small = { {"1", 1}, {"2", 2}, {"3", 3} };
  auto dims_small = TensorShard::getDefaultMaxDimensions<3>(axes_to_dims_small, shard_span_small);
  BOOST_CHECK_EQUAL(dims_small.at(0), 18);
  BOOST_CHECK_EQUAL(dims_small.at(1), 36);
  BOOST_CHECK_EQUAL(dims_small.at(2), 54);

  std::map<std::string, int> axes_to_dims_medium = { {"1", 0}, {"2", 1}, {"3", 2}, {"4", 3} };
  std::map<std::string, int> shard_span_medium = { {"1", 10}, {"2", 20}, {"3", 30}, {"4", 40} };
  auto dims_medium = TensorShard::getDefaultMaxDimensions<4>(axes_to_dims_medium, shard_span_medium);
  BOOST_CHECK_EQUAL(dims_medium.at(0), 140);
  BOOST_CHECK_EQUAL(dims_medium.at(1), 280);
  BOOST_CHECK_EQUAL(dims_medium.at(2), 420);
  BOOST_CHECK_EQUAL(dims_medium.at(3), 560);
}

BOOST_AUTO_TEST_CASE(makeShardIndicesFromShardIDsDefaultDevice)
{
  Eigen::DefaultDevice device;

  // Setup the input data
  int nlabels = 6, maxlabels = 1000;
  Eigen::array<Eigen::Index, 3> dimensions = { nlabels,nlabels,nlabels };
  Eigen::array<Eigen::Index, 3> dimensions_max = { maxlabels,maxlabels,maxlabels };
  std::map<std::string, int> axes_to_dims = {{"1", 0}, {"2", 1}, {"3", 2}};
  std::map<std::string, int> shard_span = { {"1", nlabels}, {"2", nlabels}, {"3", nlabels} };
  std::map<std::string, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>> shard_ids;
  Eigen::Tensor<int, 1> shard_ids_values(nlabels);
  shard_ids_values.setConstant(1);
  for (const auto& axis_to_dim : axes_to_dims) {
    TensorDataDefaultDevice<int, 1> shard_ids_tmp(Eigen::array<Eigen::Index, 1>({ nlabels }));
    shard_ids_tmp.setData(shard_ids_values);
    shard_ids_tmp.syncHAndDData(device);
    shard_ids.emplace(axis_to_dim.first, std::make_shared<TensorDataDefaultDevice<int,1>>(shard_ids_tmp));
  }

  // Test for the shard indices
  TensorDataDefaultDevice<int, 3> indices_shard(dimensions);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_shard_ptr = std::make_shared<TensorDataDefaultDevice<int,3>>(indices_shard);
  indices_shard_ptr->setData();
  indices_shard_ptr->syncHAndDData(device);
  TensorShard::makeShardIndicesFromShardIDs(axes_to_dims, shard_span, dimensions, dimensions_max, shard_ids, indices_shard_ptr, device);
  indices_shard_ptr->syncHAndDData(device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_shard_ptr->getData()(i, j, k), 1);
      }
    }
  }

  // make the expected tensor indices
  std::map<std::string, int> shard_span_new = { {"1", 2}, {"2", 2}, {"3", 2} };
  int shard_n_indices = maxlabels/2;
  std::vector<int> shard_id_indices = { 0, 0, 1, 1, 2, 2 };
  Eigen::Tensor<int, 3> indices_test(nlabels, nlabels, nlabels);
  for (int i = 0; i < nlabels; ++i) {
    shard_ids_values(i) = shard_id_indices.at(i) + 1;
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = shard_id_indices.at(i) + shard_id_indices.at(j) * shard_n_indices + shard_id_indices.at(k) * shard_n_indices * shard_n_indices + 1;
      }
    }
  }

  // Remake the shard_ids
  for (const auto& axis_to_dim : axes_to_dims) {
    TensorDataDefaultDevice<int, 1> shard_ids_tmp(Eigen::array<Eigen::Index, 1>({ nlabels }));
    shard_ids_tmp.setData(shard_ids_values);
    shard_ids_tmp.syncHAndDData(device);
    shard_ids.at(axis_to_dim.first) = std::make_shared<TensorDataDefaultDevice<int, 1>>(shard_ids_tmp);
  }

  // Test for the shard indices
  indices_shard_ptr->setData();
  indices_shard_ptr->syncHAndDData(device);
  TensorShard::makeShardIndicesFromShardIDs(axes_to_dims, shard_span_new, dimensions, dimensions_max, shard_ids, indices_shard_ptr, device);
  indices_shard_ptr->syncHAndDData(device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_EQUAL(indices_shard_ptr->getData()(i, j, k), indices_test(i, j, k));
      }
    }
  }

}

BOOST_AUTO_TEST_SUITE_END()