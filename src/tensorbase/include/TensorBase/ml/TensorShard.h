/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORSHARD_H
#define TENSORBASE_TENSORSHARD_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>
#include <map>

namespace TensorBase
{
  /**
    @brief Class for managing the tensor shard ids
  */
  class TensorShard {
  public:
    TensorShard() = default;
    ~TensorShard() = default;

    /*
    @brief Determine the maximum number of shards given the tensor dimensions

    @param[in] tensor_dimensions The dimensions of the tensor

    @returns The number of shards or -1 if the number of shards exceeds the `max_shard_id` value
    */
    template<int TDim>
    int getNMaxShards(const Eigen::array<Eigen::Index, TDim>& tensor_dimensions) const;

    /*
    @brief Convert the 1D shard ID into a TDim indices tensor that describes the shart IDs of each Tensor element

    The conversion is done by the following algorithm:
      1. set Dim = 0 as the reference axis
      2. compute Tensor shard IDs i, j, k, ... as (index i - 1) + (index j - 1)*SUM(axis i - 1 to MAX axis size) + (index k - 1)*SUM(axis i - 1 to MAX axis size)*SUM(axis j - 1 to MAX axis size) ...
        where the - 1 is due to the indices starting at 1
        a. compute an adjusted axis index as (Index - 1) if Dim = 0 or as (Index - 1)*Prod[(Dim - 1).size() to Dim = 1]
        b. broadcast to the size of the tensor
        c. add all adjusted axis tensors together

    @param[in] axes_to_dims
    @param[in] shard_spans
    @param[in] tensor_dimensions
    @param[in] shard_ids pointer to the shard indices
    @param[out] indices_shard pointer to the TDim shard indices
    @param[in] device
    */
    template<typename DeviceT, int TDim>
    void makeShardIndicesFromShardIDs(const std::map<std::string, int>& axes_to_dims, const std::map<std::string, int>& shard_spans, const Eigen::array<Eigen::Index, TDim>& tensor_dimensions, 
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_ids,
      std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_shard, DeviceT& device) const;

  private:
    // NOTES: 250 = maximum dimensions of an Eigen::Tensor
    // NOTES: 18446744073709551615 = maximum value of an unsigned long long
    const int max_tensor_dimensions_ = 250;
    const int max_axis_size_ = 1e9;
    const int max_shard_index_ = 1e9;
    const int max_shard_id_ = 1e9;
  };
  template<int TDim>
  inline int TensorShard::getNMaxShards(const Eigen::array<Eigen::Index, TDim>& tensor_dimensions) const
  {
    long double n_shard_indices = 1;
    for (int i = 0; i < TDim; ++i) n_shard_indices *= tensor_dimensions.at(i);
    if (n_shard_indices / max_shard_index_ > max_shard_id_) return -1;
    else return n_shard_indices / max_shard_index_ + 1;
  }
  template<typename DeviceT, int TDim>
  inline void TensorShard::makeShardIndicesFromShardIDs(const std::map<std::string, int>& axes_to_dims, const std::map<std::string, int>& shard_spans, const Eigen::array<Eigen::Index, TDim>& tensor_dimensions, const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_ids, std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_shard, DeviceT& device) const
  {
    assert(axes_to_dims.size() == shard_spans.size());
    assert(axes_to_dims.size() == TDim);
    assert(shard_ids.size() == TDim);

    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_shard_values(indices_shard->getDataPointer().get(), indices_shard->getDimensions());

    // the number of shard ids along the axis
    int n_shard_ids_cumulative = 1;

    for (const auto& axis_to_index : axes_to_dims) {
      // determine the dimensions for reshaping and broadcasting the indices
      Eigen::array<Eigen::Index, TDim> indices_reshape_dimensions;
      Eigen::array<Eigen::Index, TDim> indices_bcast_dimensions;
      for (int i = 0; i < TDim; ++i) {
        if (i == axes_to_dims.at(axis_to_index.first)) {
          indices_reshape_dimensions.at(i) = tensor_dimensions.at(i);
          indices_bcast_dimensions.at(i) = 1;
        }
        else {
          indices_reshape_dimensions.at(i) = 1;
          indices_bcast_dimensions.at(i) = tensor_dimensions.at(i);
        }
      }

      // normalize and broadcast the indices across the tensor
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> shard_id_reshape(shard_ids.at(axis_to_index.first)->getDataPointer().get(), indices_reshape_dimensions);
      std::cout << "shard_id_reshape\n"<<shard_id_reshape << std::endl;
      auto shard_id_norm = (shard_id_reshape - shard_id_reshape.constant(1)) * shard_id_reshape.constant(n_shard_ids_cumulative);
      auto shard_id_bcast_values = shard_id_norm.broadcast(indices_bcast_dimensions);

      // update the indices_shard_values
      indices_shard_values.device(device) += shard_id_bcast_values;
      std::cout << "indices_shard_values\n"<<indices_shard_values << std::endl;

      // update the accumulative size
      n_shard_ids_cumulative *= ceil(float(max_axis_size_) / float(shard_spans.at(axis_to_index.first)));
    }
    indices_shard_values.device(device) += indices_shard_values.constant(1);
  }
};
#endif //TENSORBASE_TENSORSHARD_H