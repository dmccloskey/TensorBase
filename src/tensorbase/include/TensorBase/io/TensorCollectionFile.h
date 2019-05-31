/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORCOLLECTIONFILE_H
#define SMARTPEAK_TENSORCOLLECTIONFILE_H

#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <TensorBase/ml/TensorCollection.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>

#include <cereal/archives/binary.hpp>

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
    bool storeDataBinary(const std::string& filename, const TensorCollection<DeviceT>& tensor_collection);
  };

  class TensorCollectionFileDefaultDevice: public TensorCollectionFile<Eigen::DefaultDevice>{};
};
#endif //SMARTPEAK_TENSORCOLLECTIONFILE_H