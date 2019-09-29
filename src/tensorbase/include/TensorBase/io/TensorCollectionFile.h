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
    bool storeTensorTableAsCsv(const std::string& filename, const std::string& user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT& device);

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

    // Parse the headers
    std::vector<std::string> header_line; //...
    std::vector<std::string> primary_axes_labels;
    for (const std::string& header : headers.first) {
      auto header_line_iter = std::find(header_line.begin(), header_line.end(), header);
      if (header_line_iter != header_line.end()) {
        primary_axes_labels.push_back(*header_line_iter);
      }
    }

    // Calculate the shard size for the non-primary axes
    int n_cols_shard = 1;
    
    // iterate through the file and insert shard by shard    
    int n_cols = 0;
    Eigen::Tensor<TensorT, 2> shard_data();
    for (int i = 0; i < n_cols_shard; ++i) { // ... iterate over each row instead of each n_cols_shard
      // Check that the end of the file has not been reached

      // Get the .csv data
      std::map<std::string, std::vector<std::string>> axes_labels_row;
      std::vector<std::string> data_row;
      std::vector<std::string> row_line; //...

      // Populate the non-primary axes labels
      for (const auto& non_primary_data : headers.second) {
        for (const std::string& header : non_primary_data.second) {
          //...
        }
      }
      //...

      // Populate the data row
      for (const std::string& header : headers.first) {
        //...
      }
      //...

      // update the shard iterator
      ++n_cols;
      if (n_cols == n_cols_shard) n_cols = 0;
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
  inline bool TensorCollectionFile<DeviceT>::storeTensorTableAsCsv(const std::string & filename, const std::string & user_table_name, TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    // Get the .csv headers
    std::pair<std::map<std::string, std::vector<std::string>>, std::vector<std::string>> headers = getTensorTableHeaders(user_table_name, tensor_collection, device);

    // Write the headers
    std::vector<std::string> headers_line;
    for (const auto& non_primary_headers : headers.first) {
      for (const std::string& header : non_primary_headers.second) {
        headers_line.push_back(header);
      }
    }
    for (const std::string& header : headers.second) {
      headers_line.push_back(header);
    }
    // ...

    // Calculate the number of columns of the .csv file
    int n_cols = 1;
    const std::string table_name = *(tensor_collection.user_table_names_to_tensor_table_names_.at(user_table_name).begin());
    for (int i = 0; i < tensor_collection.tables_.at(table_name)->getDimensions().size(); ++i) {
      if (i != 0) {
        n_cols *= tensor_collection.tables_.at(table_name)->getDimensions().at(i);
      }
    }

    for (int i = 0; i < n_cols; ++i) {
      // TODO need to iterate over each table that maps to the user table!
      // Get the .csv data
      std::vector<std::string> row_data = getCsvDataRow(i);
      std::map<std::string, std::vector<std::string>> axes_row_data = getCsvAxesLabelsRow(i);

      // Write the data row
      std::vector<std::string> row_line;
      for (const auto& non_primary_data : axes_row_data) { // TODO do this only for the first table!
        for (const std::string& data : non_primary_data) {
          row_line.push_back(data);
        }
      }
      for (const std::string& data : row_data) {
        row_line.push_back(data);
      }
      // ...
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