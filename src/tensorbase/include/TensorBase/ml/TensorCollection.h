/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConcept.h>
#include <map>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/set.hpp>
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
      auto compare_sets = [](auto lhs, auto rhs) {return (lhs.first == rhs.first && lhs.second == rhs.second); };
      bool user_tables_equal = std::equal(user_table_names_to_tensor_table_names_.begin(), user_table_names_to_tensor_table_names_.end(), other.user_table_names_to_tensor_table_names_.begin(), compare_sets);
      return meta_equal && tables_equal && user_tables_equal;
    }

    inline bool operator!=(const TensorCollection& other) const
    {
      return !(*this == other);
    }

    template<typename T>
    void addTensorTable(const std::shared_ptr<T>& tensor_table, const std::string& user_table_name);  ///< Tensor table adder
    void addTensorTableConcept(const std::shared_ptr<TensorTableConcept<DeviceT>>& tensor_table, const std::string& user_table_name);  ///< Tensor table concept adder
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
    @brief get all of the table names in the collection that correspond to a user table name

    @returns a set of strings with the table names
    */
    std::set<std::string> getTableNamesFromUserName(const std::string& user_table_name) const;

    /*
    @brief Find the user table name that corresponds to the table name

    @returns The user table name
    */
    std::string getUserNameFromTableName(const std::string& table_name) const;

    /*
    @brief Links the `axes`, `indices`, `indices_view`, and `is_modified` shared_ptr so that queries made against
      one user table are also applied to all other user tables.  The method assumes that the
      first axis is split by type and differs among all user tables.
    */
    void linkAxesAndIndicesByUserTableName(const std::string& user_table_name, const std::string& p_axis_name);

    /*
    @brief Links the `axes`, `indices`, `indices_view`, and `is_modified` shared_ptr so that queries made against
      one table axis are applied to all other tables that have the same axis name.
    */
    void linkAxesAndIndicesByAxisName(const std::vector<std::string>& axes_names);

    std::map<std::string, std::shared_ptr<TensorTableConcept<DeviceT>>> tables_; ///< map of Tensor tables
    std::map<std::string, std::set<std::string>> user_table_names_to_tensor_table_names_; ///< map of user names to tensor names split based on type

  protected:
    int id_ = -1;
    std::string name_ = "";
    
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(id_, name_, tables_, user_table_names_to_tensor_table_names_);
    }
  };

  template<typename DeviceT>
  template<typename T>
  inline void TensorCollection<DeviceT>::addTensorTable(const std::shared_ptr<T>& tensor_table, const std::string& user_table_name)
  {
    // Add the tensor table
    auto found_table = tables_.emplace(tensor_table->getName(), std::shared_ptr<TensorTableConcept<DeviceT>>(new TensorTableWrapper<T, DeviceT>(tensor_table)));
    if (!found_table.second)
      std::cout << "The table " << tensor_table->getName() << " already exists in the collection." << std::endl;

    // Add the user_table_name
    std::set<std::string> table_names = { tensor_table->getName() };
    auto found_user_table = user_table_names_to_tensor_table_names_.emplace(user_table_name, table_names);
    if (!found_user_table.second)
      user_table_names_to_tensor_table_names_.at(user_table_name).insert(tensor_table->getName());

  }

  template<typename DeviceT>
  inline void TensorCollection<DeviceT>::addTensorTableConcept(const std::shared_ptr<TensorTableConcept<DeviceT>>& tensor_table, const std::string& user_table_name)
  {
    // Add the tensor table
    auto found = tables_.emplace(tensor_table->getName(), tensor_table);
    if (!found.second)
      std::cout << "The table " << tensor_table->getName() << " already exists in the collection." << std::endl;

    // Add the user_table_name
    std::set<std::string> table_names = { tensor_table->getName() };
    auto found_user_table = user_table_names_to_tensor_table_names_.emplace(user_table_name, table_names);
    if (!found_user_table.second)
      user_table_names_to_tensor_table_names_.at(user_table_name).insert(tensor_table->getName());
  }

  template<typename DeviceT>
  inline void TensorCollection<DeviceT>::removeTensorTable(const std::string & table_name)
  {
    // Remove the tensor table
    auto tables_it = tables_.find(table_name);
    if (tables_it != tables_.end())
      tables_.erase(tables_it);
    else
      std::cout << "The table " << table_name << " does not exist in the collection." << std::endl;

    // Remove the tensor table name from the user table names map
    for (auto& user_table_names_map : user_table_names_to_tensor_table_names_) {
      auto tables_it = user_table_names_map.second.find(table_name);
      if (tables_it != user_table_names_map.second.end())
        user_table_names_map.second.erase(tables_it);
    }

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
    user_table_names_to_tensor_table_names_.clear();
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

  template<typename DeviceT>
  inline std::set<std::string> TensorCollection<DeviceT>::getTableNamesFromUserName(const std::string & user_table_name) const
  {
    try {
      return user_table_names_to_tensor_table_names_.at(user_table_name);
    }
    catch (std::out_of_range& e) {
      std::cout << "User table name " << user_table_name << " does not exist." << std::endl;
      return std::set<std::string>();
    }
    catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      return std::set<std::string>();
    }
  }

  template<typename DeviceT>
  inline std::string TensorCollection<DeviceT>::getUserNameFromTableName(const std::string & table_name) const
  {
    std::string user_table_name;
    for (auto& user_table_names_map : user_table_names_to_tensor_table_names_) {
      if (user_table_names_map.second.count(table_name) > 0) {
        user_table_name = user_table_names_map.first;
      }
    }
    return user_table_name;
  }
  template<typename DeviceT>
  inline void TensorCollection<DeviceT>::linkAxesAndIndicesByUserTableName(const std::string& user_table_name, const std::string& p_axis_name)
  {
    // get the first encounter of all indices, indices_view, and is_modified
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices, indices_view, is_modified;
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> axes;
    for (auto& table_name : getTableNamesFromUserName(user_table_name)) {
      auto& table = getTensorTableConcept(table_name);
      for (auto& axis_to_dim : table->getAxesToDims()) {
        if (axis_to_dim.first != p_axis_name) {
          indices.emplace(axis_to_dim.first, table->getIndices().at(axis_to_dim.first));
          indices_view.emplace(axis_to_dim.first, table->getIndicesView().at(axis_to_dim.first));
          is_modified.emplace(axis_to_dim.first, table->getIsModified().at(axis_to_dim.first));
          axes.emplace(axis_to_dim.first, table->getAxes().at(axis_to_dim.first));
        }
      }
    }

    // set the indices, indices_view and is_modified using the same indices
    for (auto& table_name : getTableNamesFromUserName(user_table_name)) {
      auto& table = getTensorTableConcept(table_name);
      for (auto& axis_to_dim : table->getAxesToDims()) {
        if (axis_to_dim.first != p_axis_name) {
          table->getIndices().at(axis_to_dim.first) = indices.at(axis_to_dim.first);
          table->getIndicesView().at(axis_to_dim.first) = indices_view.at(axis_to_dim.first);
          table->getIsModified().at(axis_to_dim.first) = is_modified.at(axis_to_dim.first);
          table->getAxes().at(axis_to_dim.first) = axes.at(axis_to_dim.first);
        }
      }
    }
  }
  template<typename DeviceT>
  inline void TensorCollection<DeviceT>::linkAxesAndIndicesByAxisName(const std::vector<std::string>& axes_names)
  {
    // get the first encounter of all indices, indices_view, and is_modified
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices, indices_view, is_modified;
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> axes;

    // and then set the indices, indices_view, and is_modified using the same indices for all other encounters
    for (auto& table_name_to_table : tables_) {
      for (auto& axis_to_dim : table_name_to_table.second->getAxesToDims()) {
        if (std::count(axes_names.begin(), axes_names.end(), axis_to_dim.first) > 0) {
          auto found_indices = indices.emplace(axis_to_dim.first, table_name_to_table.second->getIndices().at(axis_to_dim.first));
          if (!found_indices.second) {
            table_name_to_table.second->getIndices().at(axis_to_dim.first) = indices.at(axis_to_dim.first);
          }
          auto found_indices_view = indices_view.emplace(axis_to_dim.first, table_name_to_table.second->getIndicesView().at(axis_to_dim.first));
          if (!found_indices_view.second) {
            table_name_to_table.second->getIndicesView().at(axis_to_dim.first) = indices_view.at(axis_to_dim.first);
          }
          auto found_indices_is_modified = is_modified.emplace(axis_to_dim.first, table_name_to_table.second->getIsModified().at(axis_to_dim.first));
          if (!found_indices_is_modified.second) {
            table_name_to_table.second->getIsModified().at(axis_to_dim.first) = is_modified.at(axis_to_dim.first);
          }
          auto found_indices_axes = axes.emplace(axis_to_dim.first, table_name_to_table.second->getAxes().at(axis_to_dim.first));
          if (!found_indices_axes.second) {
            table_name_to_table.second->getAxes().at(axis_to_dim.first) = axes.at(axis_to_dim.first);
          }
        }
      }
    }
  }
};
#endif //TENSORBASE_TENSORCOLLECTION_H