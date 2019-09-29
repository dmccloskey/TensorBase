/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORCOLLECTIONFILE_H
#define SMARTPEAK_TENSORCOLLECTIONFILE_H

#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/array.hpp>
#include <cereal/archives/binary.hpp>

#include <TensorBase/ml/TensorCollection.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>

namespace TensorBase
{
  /**
    @brief Class reading and writing TensorCollections
  */
  template<typename DeviceT>
  class TensorCollectionFile
  {
public:
    TensorCollectionFile() = default; ///< Default constructor
    ~TensorCollectionFile() = default; ///< Default destructor
 
    /**
      @brief Load the entire tensor collection metadata and tensor data from disk

      @param filename The name of the binary data file
      @param tensor_collection The Tensor collection to load

      @returns Status True on success, False if not
    */ 
    bool loadTensorCollectionBinary(const std::string& filename, TensorCollection<DeviceT>& tensor_collection, DeviceT& device);

    /**
      @brief Load select tensor table tensor data from .csv file

      @param filename The name of the binary data file
      @param user_table_name The name of the user tensor table
      @param tensor_collection The Tensor collection to load

      @returns Status True on success, False if not
    */
    bool loadTensorTableFromCsv(const std::string& filename, const std::string& user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
 
    /**
      @brief Store the entire tensor collection metadata and tensor data to disk

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    bool storeTensorCollectionBinary(const std::string& filename, const TensorCollection<DeviceT>& tensor_collection, DeviceT& device);

    /**
      @brief Store select tensor table tensor data as .csv file

      @param filename The name of the binary data file
      @param user_table_name The name of user tensor table
      @param tensor_collection The Tensor collection to store

      @returns Status True on success, False if not
    */
    bool storeTensorTableFromCsv(const std::string& filename, const std::string& user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT& device);

    /**
      @brief Get the tensor table headers

      @param table_name The name of tensor table
      @param tensor_collection The Tensor collection to store

      @returns a pair of string vectors for the non-primary and primary axis headers
    */
    static std::pair<std::map<std::string, std::vector<std::string>>, std::vector<std::string>> getTensorTableHeaders(const std::string& table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
  };

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::loadTensorCollectionBinary(const std::string & filename, TensorCollection<DeviceT>& tensor_collection, DeviceT& device)
  {
    // Load in the serialized metadata
    std::ifstream ifs(filename, std::ios::binary);
    if (ifs.is_open()) {
      cereal::BinaryInputArchive iarchive(ifs);
      iarchive(tensor_collection);
      ifs.close();
    }

    // Load in the binarized tensor data
    for (auto& tensor_table_map : tensor_collection.tables_) {
      tensor_table_map.second->setAxes(); // initialize the axes
      for (auto& axis_map : tensor_table_map.second->getAxes()) {
        axis_map.second->setLabels(); // initialize the axes labels host/device memory
      }
      tensor_table_map.second->setData(); // initialize the data host/device memory
      tensor_table_map.second->loadTensorTableAxesBinary(tensor_table_map.second->getDir(), device);
      tensor_table_map.second->loadTensorTableBinary(tensor_table_map.second->getDir(), device);
    }

    return true;
  }

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::loadTensorTableFromCsv(const std::string & filename, const std::string & user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    // Get the .csv headers
    std::pair<std::map<std::string, std::vector<std::string>>, std::vector<std::string>> headers = getTensorTableHeaders(user_table_name, tensor_collection, device);

    // Calculate the number of columns of the .csv file
    int n_cols = 1;

    for (int i = 0; i < n_cols; ++i) {
      // Get the .csv data
    }

    return true;
  }

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::storeTensorCollectionBinary(const std::string & filename, const TensorCollection<DeviceT>& tensor_collection, DeviceT& device)
  {
    // Store the serialized metadata
    std::ofstream ofs(filename, std::ios::binary);
    //if (ofs.is_open() == false) {// Lines check to make sure the file is not already created
    cereal::BinaryOutputArchive oarchive(ofs);
    oarchive(tensor_collection);
    ofs.close();
    //}// Lines check to make sure the file is not already created

    // Store the binarized tensor data
    for (auto& tensor_table_map : tensor_collection.tables_) {
      tensor_table_map.second->storeTensorTableAxesBinary(tensor_table_map.second->getDir(), device);
      tensor_table_map.second->storeTensorTableBinary(tensor_table_map.second->getDir(), device);
    }
    return true;
  }

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::storeTensorTableFromCsv(const std::string & filename, const std::string & user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    // Get the .csv headers
    std::pair<std::map<std::string, std::vector<std::string>>, std::vector<std::string>> headers = getTensorTableHeaders(user_table_name, tensor_collection, device);

    // Calculate the number of columns of the .csv file
    int n_cols = 1;

    for (int i = 0; i < n_cols; ++i) {
      // Get the .csv data
    }

    return true;
  }

  template<typename DeviceT>
  inline std::pair<std::map<std::string, std::vector<std::string>>, std::vector<std::string>> TensorCollectionFile<DeviceT>::getTensorTableHeaders(const std::string & user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    // Make the expected headers
    std::vector<std::string> primary_headers;
    std::map<std::string, std::vector<std::string>> non_primary_headers;
    bool is_first_table = true;
    for (const std::string& table_name : tensor_collection.user_table_names_to_tensor_table_names_.at(user_table_name)) {
      for (const auto& axes_map : tensor_collection.tables_.at(table_name)->getAxes()) {
        // use the dimension names to create the non primary axis headers
        if (tensor_collection.tables_.at(table_name)->getDimFromAxisName(axes_map.first) != 0 && is_first_table) {
          Eigen::TensorMap<Eigen::Tensor<std::string, 1>>& dimensions = axes_map.second->getDimensions();
          std::vector<std::string> dimension_names(dimensions.data(), dimensions.data() + dimensions.size());
          non_primary_headers.emplace(axes_map.first, dimension_names);
        }
        // use the axis labels to create the primary axis headers
        else if (tensor_collection.tables_.at(table_name)->getDimFromAxisName(axes_map.first) == 0) {
          assert(axes_map.second->getDimensions().size() == 1);
          std::vector<std::string> labels = axes_map.second->getLabelsAsStrings(device);
          for (const std::string& label : labels) {
            primary_headers.push_back(label);
          }
        }
      }
      is_first_table = false;
    }
    return std::make_pair(non_primary_headers, primary_headers);
  }

  class TensorCollectionFileDefaultDevice : public TensorCollectionFile<Eigen::DefaultDevice> {};
};
#endif //SMARTPEAK_TENSORCOLLECTIONFILE_H