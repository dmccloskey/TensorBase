/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <map>

namespace TensorBase
{
  /**
    @brief Class for managing heterogenous Tensors
  */
  class TensorCollection
  {
  public:
    TensorCollection() = default;  ///< Default constructor
    ~TensorCollection() = default; ///< Default destructor

    template<typename T>
    void addTensorTable(const std::shared_ptr<T>& tensor_table);  ///< Tensor table adder
    void removeTensorTable(const std::string& table_name); ///< Tensor table remover
    void clear(); ///< clear the map of Tensor tables

    /*
    @brief get all of the table names in the collection

    @note `struct GetTableNamesHelper` is used to accumulate the table names

    @returns a vector of strings with the table names
    */
    std::vector<std::string> getTableNames() const;

    /*
    @brief write modified table shards to disk

    @returns true if successful, false otherwise
    */
    bool writeShardsToDisk();

    /*
    @brief reads shards from disk and updates
      the tensors in the collection

    @returns true if successful, false otherwise
    */
    bool readShardsFromDisk();

    std::map<std::string, std::shared_ptr<TensorTableConcept>> tables_; ///< tuple of std::shared_ptr TensorTables<TensorT, DeviceT, TDim>
  };

  template<typename T>
  inline void TensorCollection::addTensorTable(const std::shared_ptr<T>& tensor_table)
  {
    auto found = tables_.emplace(tensor_table->getName(), std::shared_ptr<TensorTableConcept>(new TensorTableWrapper<T>(tensor_table)));
  }

  inline void TensorCollection::removeTensorTable(const std::string & table_name)
  {
    //TODO
  }

  inline void TensorCollection::clear()
  {
    tables_.clear();
  }

  inline std::vector<std::string> TensorBase::TensorCollection::getTableNames() const
  {
    std::vector<std::string> names;
    for (const auto& ttable : tables_) {
      names.push_back(ttable.second->getName());
    }
    return names;
  }
};
#endif //TENSORBASE_TENSORCOLLECTION_H