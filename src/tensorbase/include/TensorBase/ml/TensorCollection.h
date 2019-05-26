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
  template<typename DeviceT>
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

    std::map<std::string, std::shared_ptr<TensorTableConcept<DeviceT>>> tables_; ///< tuple of std::shared_ptr TensorTables<TensorT, DeviceT, TDim>
  };

  template<typename DeviceT>
  template<typename T>
  inline void TensorCollection<DeviceT>::addTensorTable(const std::shared_ptr<T>& tensor_table)
  {
    auto found = tables_.emplace(tensor_table->getName(), std::shared_ptr<TensorTableConcept<DeviceT>>(new TensorTableWrapper<T, DeviceT>(tensor_table)));
    if (!found.second)
      std::cout << "The table " << tensor_table->getName() << " already exists in the collection." << std::endl;
  }

  template<typename DeviceT>
  inline void TensorCollection<DeviceT>::removeTensorTable(const std::string & table_name)
  {
    auto it = tables_.find(table_name);
    if (it != tables_.end())
      tables_.erase(it);
    else
      std::cout << "The table " << table_name << " doest not exist in the collection." << std::endl;
  }

  template<typename DeviceT>
  inline void TensorCollection<DeviceT>::clear()
  {
    tables_.clear();
  }

  template<typename DeviceT>
  inline std::vector<std::string> TensorCollection<DeviceT>::getTableNames() const
  {
    std::vector<std::string> names;
    for (const auto& ttable : tables_) {
      names.push_back(ttable.second->getName());
    }
    return names;
  }
};
#endif //TENSORBASE_TENSORCOLLECTION_H