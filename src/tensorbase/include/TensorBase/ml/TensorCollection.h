/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <map>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

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
    TensorCollection(const std::string& name) : name_(name) {};
    ~TensorCollection() = default; ///< Default destructor

    template<typename DeviceTOther>
    inline bool operator==(const TensorCollection<DeviceTOther>& other) const
    {
      bool meta_equal = std::tie(id_, name_) == std::tie(other.id_, other.name_);
      auto compare_maps = [](auto lhs, auto rhs) {return *(lhs.second.get()) == *(rhs.second.get()); };
      bool tables_equal = std::equal(tables_.begin(), tables_.end(), other.tables_.begin(), compare_maps);
      return meta_equal && tables_equal;
    }

    inline bool operator!=(const TensorCollection& other) const
    {
      return !(*this == other);
    }

    template<typename T>
    void addTensorTable(const std::shared_ptr<T>& tensor_table);  ///< Tensor table adder
    void addTensorTableConcept(const std::shared_ptr<TensorTableConcept<DeviceT>>& tensor_table);  ///< Tensor table concept adder
    void removeTensorTable(const std::string& table_name); ///< Tensor table remover
    std::shared_ptr<TensorTableConcept<DeviceT>> getTensorTableConcept(const std::string& table_name) const; ///< Tensor table concept getter
    void clear(); ///< clear the map of Tensor tables

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

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

    std::map<std::string, std::shared_ptr<TensorTableConcept<DeviceT>>> tables_; ///< map of Tensor tables

  protected:
    int id_ = -1;
    std::string name_ = "";
    
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(id_, name_, tables_);
    }
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
  inline void TensorCollection<DeviceT>::addTensorTableConcept(const std::shared_ptr<TensorTableConcept<DeviceT>>& tensor_table)
  {
    auto found = tables_.emplace(tensor_table->getName(), tensor_table);
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
  inline std::shared_ptr<TensorTableConcept<DeviceT>> TensorCollection<DeviceT>::getTensorTableConcept(const std::string & table_name) const
  {
    if (tables_.count(table_name) > 0) {
      return tables_.at(table_name);
    }
    else {
      std::cout << "The table " << table_name << " doest not exist in the collection." << std::endl;
      return nullptr;
    }
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

  class TensorCollectionDefaultDevice : public TensorCollection<Eigen::DefaultDevice>
  {
  public:
    using TensorCollection<Eigen::DefaultDevice>::TensorCollection;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorCollection<Eigen::DefaultDevice>>(this));
    }
  };

  class TensorCollectionCpu : public TensorCollection<Eigen::ThreadPoolDevice>
  {
  public:
    using TensorCollection<Eigen::ThreadPoolDevice>::TensorCollection;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorCollection<Eigen::ThreadPoolDevice>>(this));
    }
  };
};
#endif //TENSORBASE_TENSORCOLLECTION_H