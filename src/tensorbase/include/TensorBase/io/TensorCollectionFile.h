/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORCOLLECTIONFILE_H
#define SMARTPEAK_TENSORCOLLECTIONFILE_H

#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/io/CSVWriter.h>
#include <TensorBase/io/csv.h>

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
    bool loadTensorCollectionBinary(const std::string& filename, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);

    /**
      @brief Load select tensor table tensor data from .csv file

      TODO:
      Currently, all of the data from .csv has to be read in and appended to the table at once
      Commented out code is provided that would add logic to append 1 shard at a time
      However, this would require improving the logic in TensorTable::insertIntoTableFromCsv 
        to align the sparse tensor to the axis labels instead of just appending it to the tensor data

      @param filename The name of the binary data file
      @param user_table_name The name of the user tensor table
      @param tensor_collection The Tensor collection to load

      @returns Status True on success, False if not
    */
    bool loadTensorTableFromCsv(const std::string& filename, const std::string& user_table_name, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
 
    /**
      @brief Store the entire tensor collection metadata and tensor data to disk

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    bool storeTensorCollectionBinary(const std::string& filename, const std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);

    /**
      @brief Store select tensor table tensor data as .csv file

      @param filename The name of the binary data file
      @param user_table_name The name of user tensor table
      @param tensor_collection The Tensor collection to store

      @returns Status True on success, False if not
    */
    bool storeTensorTableAsCsv(const std::string& filename, const std::string& user_table_name, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);

    /**
      @brief Get the tensor table headers

      @param table_name The name of tensor table
      @param tensor_collection The Tensor collection to store

      @returns a pair of maps for the non-primary (first) and primary headers (second) where
        the first is a key/value pair of axis name and a string vectors of dimension names
        and the second is a key/value pair of tables_names and a string vectors of axis labels
    */
    static std::pair<std::map<std::string, std::vector<std::string>>, std::map<std::string, std::vector<std::string>>> getTensorTableHeaders(const std::string& table_name, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
  };

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::loadTensorCollectionBinary(const std::string & filename, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Load in the serialized metadata
    std::ifstream ifs(filename, std::ios::binary);
    if (ifs.is_open()) {
      cereal::BinaryInputArchive iarchive(ifs);
      iarchive(tensor_collection);
      ifs.close();
    }

    // Load in the binarized tensor data
    for (auto& tensor_table_map : tensor_collection->tables_) {
      tensor_table_map.second->setAxes(device); // initialize the axes (NOTE: this resets all indices to their default values!)
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
  inline bool TensorCollectionFile<DeviceT>::loadTensorTableFromCsv(const std::string & filename, const std::string & user_table_name, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    // Get the .csv headers
    const std::string first_table_name = *(tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name).begin());
    std::pair<std::map<std::string, std::vector<std::string>>, std::map<std::string, std::vector<std::string>>> headers = getTensorTableHeaders(user_table_name, tensor_collection, device);
    
    // Determine the number of rows that will be read in
    auto count_rows = [](const std::string& filename) {
      csv::CSVReader reader(filename);
      csv::CSVRow row;
      int n_shard_size = 0;
      while (reader.read_row(row)) { ++n_shard_size; }
      return n_shard_size;
    };
    int n_shard_size = count_rows(filename);

    //// Calculate the shard size for the non-primary axes
    //int n_shard_size = tensor_collection->tables_.at(first_table_name)->getCsvShardSpans().at(1);

    // Prepare the data structures for calls to insertIntoTableFromCsv
    std::map<std::string, Eigen::Tensor<std::string, 2>> data_new;
    for (const auto& primary_data : headers.second) {
      Eigen::Tensor<std::string, 2> empty((int)primary_data.second.size(), n_shard_size);
      data_new.emplace(primary_data.first, empty);
    }
    std::map<std::string, Eigen::Tensor<std::string, 2>> labels_new;
    for (const auto& non_primary_data : headers.first) {
      Eigen::Tensor<std::string, 2> empty((int)non_primary_data.second.size(), n_shard_size);
      labels_new.emplace(non_primary_data.first, empty);
    }
    
    // iterate through the file and insert shard by shard   
    int n_cols = 0;
    csv::CSVReader reader(filename);
    for (csv::CSVRow& row: reader) {
      for (const std::string& table_name : tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name)) {
        // Get the .csv data for each axis row
        if (table_name == first_table_name) {
          // Get the .csv data for the non-primary axes labels
          for (const auto& non_primary_data : headers.first) {
            int header_iter = 0;
            for (const std::string& header : non_primary_data.second) {
              std::string label = row[header].get<>();
              labels_new.at(non_primary_data.first)(header_iter, n_cols) = label;
              ++header_iter;
            }
          }
        }

        // Get the .csv data for the data row
        for (const auto& primary_data : headers.second) {
          int header_iter = 0;
          for (const std::string& header : primary_data.second) {
            std::string cell = row[header].get<>();
            data_new.at(primary_data.first)(header_iter, n_cols) = cell;
            ++header_iter;
          }
        }
      }

      // update the shard iterator
      ++n_cols;
      //if (n_cols == n_shard_size) {
      //  // add in the shard to the tensor tables
      //  for (const std::string& table_name : tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name)) {
      //    if (table_name == first_table_name) {
      //      tensor_collection->tables_.at(table_name)->insertIntoTableFromCsv(labels_new, data_new.at(table_name), device);
      //    }
      //    else {
      //      tensor_collection->tables_.at(table_name)->insertIntoTableFromCsv(data_new.at(table_name), device);
      //    }
      //  }

      //  // reset the labels and data
      //  for (const auto& primary_data : headers.second) {
      //    Eigen::Tensor<std::string, 2> empty((int)primary_data.second.size(), n_shard_size);
      //    data_new.at(primary_data.first) = empty;
      //  }
      //  for (const auto& non_primary_data : headers.first) {
      //    Eigen::Tensor<std::string, 2> empty((int)non_primary_data.second.size(), n_shard_size);
      //    labels_new.at(non_primary_data.first) = empty;
      //  }
      //  n_cols = 0;
      //}
    }

    //// Insert the remaining rows
    //if (n_cols != 0) { // i.e., n_cols has not been reset back to 0 after reaching the shard size
    //  for (auto& labels_map : labels_new) {
    //    Eigen::TensorMap<Eigen::Tensor<std::string, 2>> resized(labels_map.second.data(), Eigen::array<Eigen::Index, 2>({ labels_map.second.dimension(0), n_cols }));
    //    Eigen::Tensor<std::string, 2> reshape = resized;
    //    labels_map.second = reshape;
    //  }
    //  for (const std::string& table_name : tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name)) {
    //    Eigen::TensorMap<Eigen::Tensor<std::string, 2>> resized(data_new.at(table_name).data(), Eigen::array<Eigen::Index, 2>({ data_new.at(table_name).dimension(0), n_cols }));
    //    Eigen::Tensor<std::string, 2> reshape = resized;
    //    data_new.at(table_name) = reshape;
    //    if (table_name == first_table_name) {
    //      tensor_collection->tables_.at(table_name)->insertIntoTableFromCsv(labels_new, data_new.at(table_name), device);
    //    }
    //    else {
    //      tensor_collection->tables_.at(table_name)->insertIntoTableFromCsv(data_new.at(table_name), device);
    //    }
    //  }
    //}

    // add in the shard to the tensor tables
    for (const std::string& table_name : tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name)) {
      if (table_name == first_table_name) {
        tensor_collection->tables_.at(table_name)->insertIntoTableFromCsv(labels_new, data_new.at(table_name), device);
      }
      else {
        tensor_collection->tables_.at(table_name)->insertIntoTableFromCsv(data_new.at(table_name), device);
      }
    }

    return true;
  }

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::storeTensorCollectionBinary(const std::string & filename, const std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Store the serialized metadata
    std::ofstream ofs(filename, std::ios::binary);
    //if (ofs.is_open() == false) {// Lines check to make sure the file is not already created
    cereal::BinaryOutputArchive oarchive(ofs);
    oarchive(tensor_collection);
    ofs.close();
    //}// Lines check to make sure the file is not already created

    // Store the binarized tensor data
    for (auto& tensor_table_map : tensor_collection->tables_) {
      tensor_table_map.second->storeTensorTableAxesBinary(tensor_table_map.second->getDir(), device);
      tensor_table_map.second->storeTensorTableBinary(tensor_table_map.second->getDir(), device);
    }
    return true;
  }

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::storeTensorTableAsCsv(const std::string & filename, const std::string & user_table_name, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    CSVWriter csvwriter(filename);

    // Get the .csv headers
    std::pair<std::map<std::string, std::vector<std::string>>, std::map<std::string, std::vector<std::string>>> headers = getTensorTableHeaders(user_table_name, tensor_collection, device);

    // Write the headers
    std::vector<std::string> headers_line;
    for (const auto& non_primary_headers : headers.first) {
      for (const std::string& header : non_primary_headers.second) {
        headers_line.push_back(header);
      }
    }
    for (const auto& primary_headers : headers.second) {
      for (const std::string& header : primary_headers.second) {
        headers_line.push_back(header);
      }
    }
    csvwriter.writeDataInRow(headers_line.begin(), headers_line.end());

    // Calculate the number of columns of the .csv file
    const std::string first_table_name = *(tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name).begin());
    Eigen::array<Eigen::Index, 2> data_dimensions = tensor_collection->tables_.at(first_table_name)->getCsvDataDimensions();

    // Write the rows
    for (int i = 0; i < data_dimensions.at(1); ++i) {
      std::vector<std::string> row_line;
      for (const std::string& table_name : tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name)) {
        // Get the .csv data for each axis row
        if (table_name == first_table_name) {
          std::map<std::string, std::vector<std::string>> axes_row_data = tensor_collection->tables_.at(table_name)->getCsvAxesLabelsRow(i);
          for (const auto& non_primary_data : axes_row_data) {
            for (const std::string& data : non_primary_data.second) {
              row_line.push_back(data);
            }
          }
        }

        // Get the .csv data for each table
        std::vector<std::string> row_data = tensor_collection->tables_.at(table_name)->getCsvDataRow(i);
        for (const std::string& data : row_data) {
          row_line.push_back(data);
        }
      }
      csvwriter.writeDataInRow(row_line.begin(), row_line.end());
    }

    return true;
  }

  template<typename DeviceT>
  inline std::pair<std::map<std::string, std::vector<std::string>>, std::map<std::string, std::vector<std::string>>> TensorCollectionFile<DeviceT>::getTensorTableHeaders(const std::string & user_table_name, std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    // Make the expected headers
    std::map<std::string, std::vector<std::string>> non_primary_headers, primary_headers;
    bool is_first_table = true;
    for (const std::string& table_name : tensor_collection->user_table_names_to_tensor_table_names_.at(user_table_name)) {
      for (const auto& axes_map : tensor_collection->tables_.at(table_name)->getAxes()) {
        // use the dimension names to create the non primary axis headers
        if (tensor_collection->tables_.at(table_name)->getDimFromAxisName(axes_map.first) != 0 && is_first_table) {
          Eigen::TensorMap<Eigen::Tensor<std::string, 1>>& dimensions = axes_map.second->getDimensions();
          std::vector<std::string> dimension_names(dimensions.data(), dimensions.data() + dimensions.size());
          non_primary_headers.emplace(axes_map.first, dimension_names);
        }
        // use the axis labels to create the primary axis headers
        else if (tensor_collection->tables_.at(table_name)->getDimFromAxisName(axes_map.first) == 0) {
          assert(axes_map.second->getDimensions().size() == 1);
          std::vector<std::string> labels = axes_map.second->getLabelsAsStrings(device);
          primary_headers.emplace(table_name, labels);
        }
      }
      is_first_table = false;
    }
    return std::make_pair(non_primary_headers, primary_headers);
  }
};
#endif //SMARTPEAK_TENSORCOLLECTIONFILE_H