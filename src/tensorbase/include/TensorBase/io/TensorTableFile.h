/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORTABLEFILE_H
#define SMARTPEAK_TENSORTABLEFILE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableDefaultDevice.h>
#include <TensorBase/io/DataFile.h>

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

      The TensorData is transfered to the host, and the `is_modified` attribute is reset to all 0's

      @param[in] filename The name of the data file
      @param[in] tensor_table The tensor table to write to file
      @param[in] device

      @returns Status True on success, False if not
    */ 
    static bool storeTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device);

    /**
      @brief Create a unique name for each TensorTableShard

      @param[in] dir The directory name
      @param[in] tensor_table_name The tensor table name
      @param[in] shard_id The id of the tensor table shard

      @returns A string with the filename for the TensorTableShard
    */
    static std::string makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id);

    static bool storeTensorTableShard(const std::string& filename,
      const Eigen::Tensor<TensorT, TDim>& tensor_data,
      const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices);

    static bool loadTensorTableShard(const std::string& filename,
      Eigen::Tensor<TensorT, TDim>& tensor_data,
      const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices);

    //static void getNotInMemoryShardIDs(
    //  const TensorTable<TensorT, DeviceT, TDim>& tensor_table,
    //  std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_id,
    //  std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& modified_shard_indices, DeviceT& device);
  };

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile<TensorT, DeviceT, TDim>::loadTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device) {
    // determine the shards to read from disk
    std::shared_ptr<TensorData<int, DeviceT, 1>> not_in_memory_shard_ids;
    tensor_table.makeModifiedShardIDTensor(not_in_memory_shard_ids, device);
    if (not_in_memory_shard_ids->getTensorSize() == 0) {
      std::cout << "No shards have been modified." << std::endl;
      return false;
    }
    std::map<int, std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>> slice_indices;
    tensor_table.makeSliceIndicesFromShardIndices(not_in_memory_shard_ids, slice_indices, device);

    // read in the shards and update the TensorTable data asyncronously
    tensor_table.syncHAndDData(device);
    for (const auto slice_index : slice_indices) {
      const std::string filename = makeTensorTableShardFilename(dir, tensor_table.getName(), slice_index.first);
      loadTensorTableShard(filename, tensor_table.getData(), slice_index.second);
    }

    // update the `in_memory` tensor table attribute

    return true
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile<TensorT, DeviceT, TDim>::storeTensorTableBinary(const std::string& dir, TensorTable<TensorT, DeviceT, TDim>& tensor_table, DeviceT& device) {
    // determine the shards to write to disk
    std::shared_ptr<TensorData<int, DeviceT, 1>> modified_shard_ids;
    tensor_table.makeModifiedShardIDTensor(modified_shard_ids, device);
    if (modified_shard_ids->getTensorSize() == 0) {
      std::cout << "No shards have been modified." << std::endl;
      return false;
    }
    std::map<int, std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>> slice_indices;
    tensor_table.makeSliceIndicesFromShardIndices(modified_shard_ids, slice_indices, device);

    // write the TensorTable shards to disk asyncronously
    tensor_table.syncHAndDData(device);
    for (const auto slice_index : slice_indices) {
      const std::string filename = makeTensorTableShardFilename(dir, tensor_table.getName(), slice_index.first);
      storeTensorTableShard(filename, tensor_table.getData(), slice_index.second);
    }
    tensor_table.setDataStatus(false, true);

    // update the `is_modified` tensor table attribute
    for (auto& is_modified_map : tensor_table.getIsModified()) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified_values(is_modified_map.second->getDataPointer().get(), (int)is_modified_map.second->getTensorSize());
      is_modified_values.device(device) = is_modified_values.constant(1);
    }

    return true;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::string TensorTableFile<TensorT, DeviceT, TDim>::makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id) {
    return dir + tensor_table_name + "_" + std::to_string(shard_id) + ".tts";
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile<TensorT, DeviceT, TDim>::storeTensorTableShard(const std::string & filename, const Eigen::Tensor<TensorT, TDim>& tensor_data, const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices)
  {
    Eigen::Tensor<TensorT, TDim> shard_data = tensor_data.slice(slice_indices.first, slice_indices.second);
    DataFile::storeDataBinary<TensorT, TDim>(filename, shard_data);
    return true;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTableFile<TensorT, DeviceT, TDim>::loadTensorTableShard(const std::string & filename, Eigen::Tensor<TensorT, TDim>& tensor_data, const std::pair<Eigen::array<int, TDim>, Eigen::array<int, TDim>>& slice_indices)
  {
    return false;
  }
};
#endif //SMARTPEAK_TENSORTABLEFILE_H