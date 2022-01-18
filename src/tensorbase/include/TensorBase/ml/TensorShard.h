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
  /// Helper set of methods for calculating the shard_span, shard_id, and shard_index
  struct TensorCollectionShardHelper {
    /// Round to 1 if below 1
    static int round_1(const int& dim_size, const double& shard_span_perc) {
      int shard_span = dim_size * shard_span_perc;
      if (shard_span <= 0) shard_span = 1;
      return shard_span;
    };
    /// Calculate the shard ID
    static int calc_shard_id(const int& shard_span, const int& i) {
      int shard_id = i / shard_span + 1;
      return shard_id;
    };
    /// Calculate the shard Index
    static int calc_shard_index(const int& shard_span, const int& i) {
      int shard_index = i % shard_span + 1;
      return shard_index;
    };
  };

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
    static int getNMaxShards(const Eigen::array<Eigen::Index, TDim>& tensor_dimensions);

    /*
    @brief Check the values of the shard span and replace the following invalid entries
      - values over the maximum shard index with the tensor dimensions
      - values less than or equal to zero with the tensor dimensions
      if the tensor dimensions are not give (i.e., set to 0) the value will be replaced with the maximum shard index

    @param[in] axes_to_dims
    @param[in] tensor_dimensions The dimensions of the tensor
    @param[in,out] shard_spans
    */
    template<int TDim>
    static void checkShardSpans(const std::map<std::string, int>& axes_to_dims, const Eigen::array<Eigen::Index, TDim>& tensor_dimensions, std::map<std::string, int>& shard_spans);

    /*
    @brief Determine the maximum dimension sizes given the number of tensor dimensions
      and the shard spans using the following algorithm

      max_dim_size(i) = log(max_shard_id)/log(TDim)*shard_span(i)

    @param[in] axes_to_dims
    @param[in] shard_spans

    @returns An array populated with the maximum dimension size
    */
    template<int TDim>
    static Eigen::array<Eigen::Index, TDim> getDefaultMaxDimensions(const std::map<std::string, int>& axes_to_dims, const std::map<std::string, int>& shard_spans);

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
    @param[in] max_dimensions
    @param[in] shard_ids pointer to the shard indices
    @param[out] indices_shard pointer to the TDim shard indices 
      It is assumed that the data has already be allocated
    @param[in] device
    */
    template<typename DeviceT, int TDim>
    static void makeShardIndicesFromShardIDs(const std::map<std::string, int>& axes_to_dims, const std::map<std::string, int>& shard_spans, const Eigen::array<Eigen::Index, TDim>& tensor_dimensions, const Eigen::array<Eigen::Index, TDim>& max_dimensions,
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_ids,
      std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_shard, DeviceT& device);

  private:
    // NOTES: 250 = maximum dimensions of an Eigen::Tensor
    // NOTES: 18446744073709551615 = maximum value of an unsigned long long
    static int getMaxTensorDimensions() { return 250; }
    static int getMaxAxisSize() { return 1e9; }
    static int getMaxShardIndex() { return 1e9; }
    static int getMaxShardID() { return 1e9; }
  };
  template<int TDim>
  inline int TensorShard::getNMaxShards(const Eigen::array<Eigen::Index, TDim>& tensor_dimensions)
  {
    long double n_shard_indices = 1;
    for (int i = 0; i < TDim; ++i) n_shard_indices *= tensor_dimensions.at(i);
    if (n_shard_indices / getMaxShardIndex() > getMaxShardID()) return -1;
    else return n_shard_indices / getMaxShardIndex() + 1;
  }
  template<int TDim>
  inline void TensorShard::checkShardSpans(const std::map<std::string, int>& axes_to_dims, const Eigen::array<Eigen::Index, TDim>& tensor_dimensions, std::map<std::string, int>& shard_spans)
  {
    assert(axes_to_dims.size() == TDim);
    assert(shard_spans.size() == TDim);
    for (auto& shard_span : shard_spans) {
      if (shard_span.second > getMaxShardIndex() && tensor_dimensions.at(axes_to_dims.at(shard_span.first)) <= 0) shard_span.second = getMaxShardIndex();
      else if (shard_span.second <= 0 && tensor_dimensions.at(axes_to_dims.at(shard_span.first)) <= 0) shard_span.second = getMaxShardIndex();
      else if (shard_span.second > getMaxShardIndex()) shard_span.second = tensor_dimensions.at(axes_to_dims.at(shard_span.first));
      else if (shard_span.second <= 0) shard_span.second = tensor_dimensions.at(axes_to_dims.at(shard_span.first));
    }
  }
  template<int TDim>
  inline Eigen::array<Eigen::Index, TDim> TensorShard::getDefaultMaxDimensions(const std::map<std::string, int>& axes_to_dims, const std::map<std::string, int>& shard_spans)
  {
    assert(axes_to_dims.size() == TDim);
    assert(shard_spans.size() == TDim);
    int max_dim_size = std::log(double(getMaxShardID()))/std::log(double(TDim));
    Eigen::array<Eigen::Index, TDim> maximum_dimensions;
    for (const auto& axis_to_index : axes_to_dims) maximum_dimensions.at(axis_to_index.second) = max_dim_size * shard_spans.at(axis_to_index.first);
    return maximum_dimensions;
  }
  template<typename DeviceT, int TDim>
  inline void TensorShard::makeShardIndicesFromShardIDs(const std::map<std::string, int>& axes_to_dims, const std::map<std::string, int>& shard_spans, const Eigen::array<Eigen::Index, TDim>& tensor_dimensions, const Eigen::array<Eigen::Index, TDim>& max_dimensions, const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_ids, std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_shard, DeviceT& device)
  {
    assert(axes_to_dims.size() == shard_spans.size());
    assert(axes_to_dims.size() == TDim);
    assert(shard_ids.size() == TDim);
    for (int i = 0; i < TDim; ++i) {
      assert(tensor_dimensions.at(i) <= max_dimensions.at(i));
    }

    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_shard_values(indices_shard->getDataPointer().get(), indices_shard->getDimensions());
    indices_shard_values.device(device) = indices_shard_values.constant(0);

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
      auto shard_id_norm = (shard_id_reshape - shard_id_reshape.constant(1)) * shard_id_reshape.constant(n_shard_ids_cumulative);
      auto shard_id_bcast_values = shard_id_norm.broadcast(indices_bcast_dimensions);

      // update the indices_shard_values
      indices_shard_values.device(device) += shard_id_bcast_values;

      // update the accumulative size
      n_shard_ids_cumulative *= ceil(float(max_dimensions.at(axes_to_dims.at(axis_to_index.first))) / float(shard_spans.at(axis_to_index.first)));
    }
    indices_shard_values.device(device) += indices_shard_values.constant(1);
  }
};
#endif //TENSORBASE_TENSORSHARD_H