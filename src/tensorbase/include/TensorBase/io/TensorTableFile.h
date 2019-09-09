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
  class TensorTableFile
  {
public:
    TensorTableFile() = default; ///< Default constructor
    ~TensorTableFile() = default; ///< Default destructor
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param tensor_table The tensor table to load from file

      @returns Status True on success, False if not
    */ 
    template<typename TensorT, typename DeviceT, int TDim>
    bool loadTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table);
 
    /**
      @brief Write data to file

      @param filename The name of the data file
      @param tensor_table The tensor table to write to file

      @returns Status True on success, False if not
    */ 
    template<typename TensorT, typename DeviceT, int TDim>
    bool storeTensorTableBinary(const std::string& dir, const TensorTable<TensorT, DeviceT, TDim>& tensor_table);

    /**
      @brief Create a unique name for each TensorTableShard

      @param dir The directory name
      @param tensor_table_name The tensor table name
      @param shard_id The id of the tensor table shard

      @returns A string with the filename for the TensorTableShard
    */
    std::string makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id);

    template<typename DeviceT>
    std::vector<int> getModifiedShardIDs(
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& is_modified,
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_id,
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_indices,
      std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_id,
      std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_indices);

    template<typename DeviceT, int TDim>
    std::vector<int> makeSliceIndicesFromShardIndices(
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_id,
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_indices,
      std::map<int, std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>>& slice_indices);

    template<typename TensorT, typename DeviceT, int TDim>
    bool storeTensorTableShard(const std::string& filename,
      const Eigen::Tensor<TensorT, TDim>& tensor_data,
      const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices);

    template<typename DeviceT>
    std::vector<int> getNotInMemoryShardIDs(
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& is_modified,
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_id,
      const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& shard_indices,
      std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_id,
      std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_indices);

    template<typename TensorT, typename DeviceT, int TDim>
    bool loadTensorTableShard(const std::string& filename,
      Eigen::Tensor<TensorT, TDim>& tensor_data,
      const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices);
  };

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile::loadTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table) {
    // determine the shards to read from disk

    // read in the shards and update the TensorTable data asyncronously

    // update the `in_memory` tensor table attribute

    return true
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile::storeTensorTableBinary(const std::string& dir, const TensorTable<TensorT, DeviceT, TDim>& tensor_table) {
    // determine the shards to write to disk
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> modified_shard_id, modified_shard_indices;
    getModifiedShardIDs(tensor_table.getIsModified(), tensor_table.getShardId(), tensor_table.getShardIndices(), modified_shard_id, modified_shard_indices);
    std::map<int, std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>> slice_indices;
    makeSliceIndicesFromShardIndices(modified_shard_id, modified_shard_indices, slice_indices);

    // write the TensorTable shards to disk asyncronously
    for (const auto slice_index : slice_indices) {
      const std::string filename = makeTensorTableShardFilename(dir, tensor_table.getName(), slice_index.first);
      storeTensorTableShard(filename, tensor_table.getData(), slice_index.second)
    }

    // update the `is_modified` tensor table attribute

    return true;
  }

  inline std::string TensorTableFile::makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id) {
    return dir + tensor_table_name + "_" + std::to_string(shard_id) + ".tts";
  }
}
#endif //SMARTPEAK_TENSORTABLEFILE_H