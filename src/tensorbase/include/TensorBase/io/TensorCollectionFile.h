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
      @brief Load data from file

      @param filename The name of the binary data file
      @param tensor_collection The Tensor collection to load

      @returns Status True on success, False if not
    */ 
    bool loadTensorCollectionBinary(const std::string& filename, TensorCollection<DeviceT>& tensor_collection);
 
    /**
      @brief Load data from file

      @param filename The name of the data file
      @param data The data to load data into

      @returns Status True on success, False if not
    */ 
    bool storeTensorCollectionBinary(const std::string& filename, const TensorCollection<DeviceT>& tensor_collection);
  };

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::loadTensorCollectionBinary(const std::string & filename, TensorCollection<DeviceT>& tensor_collection)
  {
    std::ifstream ifs(filename, std::ios::binary);
    if (ifs.is_open()) {
      cereal::BinaryInputArchive iarchive(ifs);
      iarchive(tensor_collection);
      ifs.close();
    }
    return true;
  }

  template<typename DeviceT>
  inline bool TensorCollectionFile<DeviceT>::storeTensorCollectionBinary(const std::string & filename, const TensorCollection<DeviceT>& tensor_collection)
  {
    std::ofstream ofs(filename, std::ios::binary);
    //if (ofs.is_open() == false) {// Lines check to make sure the file is not already created
    cereal::BinaryOutputArchive oarchive(ofs);
    oarchive(tensor_collection);
    ofs.close();
    //}// Lines check to make sure the file is not already created
    return true;
  }

  class TensorCollectionFileDefaultDevice : public TensorCollectionFile<Eigen::DefaultDevice> {};
};
#endif //SMARTPEAK_TENSORCOLLECTIONFILE_H