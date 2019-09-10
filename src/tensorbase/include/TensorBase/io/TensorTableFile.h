/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORTABLEFILE_H
#define SMARTPEAK_TENSORTABLEFILE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace TensorBase
{
  /**
    @brief Class for storing TensorTable data
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorTableFile
  {
public:
    TensorTableFile() = default; ///< Default constructor
    ~TensorTableFile() = default; ///< Default destructor
 
    /**
      @brief Load data from file

      @param[in] filename The name of the data file
      @param[in, out] tensor_table The tensor table to load from file
      @param[in] device

      @returns Status True on success, False if not
    */ 
    static bool loadTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device);
 
    /**
      @brief Write data to file

      @param[in] filename The name of the data file
      @param[in] tensor_table The tensor table to write to file
      @param[in] device

      @returns Status True on success, False if not
    */ 
    static bool storeTensorTableBinary(const std::string& dir, const TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device);

    /**
      @brief Create a unique name for each TensorTableShard

      @param[in] dir The directory name
      @param[in] tensor_table_name The tensor table name
      @param[in] shard_id The id of the tensor table shard

      @returns A string with the filename for the TensorTableShard
    */
    static std::string makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id);
    
    /**
      @brief Determine the Tensor data shards that have been modified

      @param[in] is_modified
      @param[in] shard_id
      @param[in] shard_indices
      @param[out] modified_shard_id
      @param[out] modified_shard_indices
      @param[in] device
    */
    static void getModifiedShardIDs(
      const TensorTable<TensorT, DeviceT, TDim>& tensor_table,
      std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, DeviceT& device);

    static void makeSliceIndicesFromShardIndices(
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids,
      std::map<int, std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>>& slice_indices);

    static bool storeTensorTableShard(const std::string& filename,
      const Eigen::Tensor<TensorT, TDim>& tensor_data,
      const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices);

    static void getNotInMemoryShardIDs(
      const TensorTable<TensorT, DeviceT, TDim>& tensor_table,
      std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_id,
      std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_indices, DeviceT& device);

    static bool loadTensorTableShard(const std::string& filename,
      Eigen::Tensor<TensorT, TDim>& tensor_data,
      const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices);
  };

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile<TensorT, DeviceT, TDim>::loadTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device) {
    // determine the shards to read from disk

    // read in the shards and update the TensorTable data asyncronously

    // update the `in_memory` tensor table attribute

    return true
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile<TensorT, DeviceT, TDim>::storeTensorTableBinary(const std::string& dir, const TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device) {
    // determine the shards to write to disk
    std::shared_ptr<TensorData<int, DeviceT, 1>> modified_shard_ids;
    getModifiedShardIDs(tensor_table.getIsModified(), tensor_table.getShardId(), tensor_table.getShardIndices(), modified_shard_ids, device);
    std::map<int, std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>> slice_indices;
    makeSliceIndicesFromShardIndices(modified_shard_ids, slice_indices);

    // write the TensorTable shards to disk asyncronously
    for (const auto slice_index : slice_indices) {
      const std::string filename = makeTensorTableShardFilename(dir, tensor_table.getName(), slice_index.first);
      storeTensorTableShard(filename, tensor_table.getData(), slice_index.second)
    }

    // update the `is_modified` tensor table attribute

    return true;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::string TensorTableFile<TensorT, DeviceT, TDim>::makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id) {
    return dir + tensor_table_name + "_" + std::to_string(shard_id) + ".tts";
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTableFile<TensorT, DeviceT, TDim>::getModifiedShardIDs(
    const TensorTable<TensorT, DeviceT, TDim>& tensor_table,
    std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, DeviceT& device) {
    // Determine the slice indices and tensor length to track the shard ids
    std::vector<std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>> slice_indices;
    int dim_size = 0;
    for (int i = 0; i < TDim; ++i) {
      dim_size += tensor_table.getDimensions().at(i);
      Eigen::array<int, 1> offset, span;
      if (i == 0) { offset.at(0) = 0; }
      else { offset.at(0) = tensor_table.getDimensions().at(i-1); }
      span.at(0) = tensor_table.getDimensions().at(i);
      slice_indices.push_back(std::make_pair(offset, span));
    }

    // Make the tensor used to track the shard ids
    TensorDataDefaultDevice<int, 1> modified_shards(dim_size);
    modified_shards.setData();
    modified_shards.syncHAndDData(device);

    // Extract out the modified shard_ids
    for (const auto& axis_to_dim: tensor_table.axes_to_dims_) {

    }
  }
};
#endif //SMARTPEAK_TENSORTABLEFILE_H