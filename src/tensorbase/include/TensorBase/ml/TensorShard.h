/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORSHARD_H
#define TENSORBASE_TENSORSHARD_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace TensorBase
{
  /**
    @brief Class for managing the tensor shard ids
  */
  class TensorShard {
  public:
    TensorShard() = default;
    ~TensorShard() = default;

    template<int TDim>
    int getNShards(const Eigen::array<Eigen::Index, TDim>& tensor_dimensions) const;

  private:
    // NOTES: 250 = maximum dimensions of an Eigen::Tensor
    // NOTES: 18446744073709551615 = maximum value of an unsigned long long
    const int max_tensor_dimensions_ = 250;
    const int max_axis_size_ = 1e9;
    const int max_shard_index_ = 1e9;
    const int max_shard_id_ = 1e9;
  };
  template<int TDim>
  inline int TensorShard::getNShards(const Eigen::array<Eigen::Index, TDim>& tensor_dimensions) const
  {
    long double n_shard_indices = 1;
    for (int i = 0; i < TDim; ++i) n_shard_indices *= tensor_dimensions.at(i);
    if (n_shard_indices / max_shard_index_ > max_shard_id_) return -1;
    else return n_shard_indices / max_shard_index_ + 1;
  }
};
#endif //TENSORBASE_TENSORSHARD_H