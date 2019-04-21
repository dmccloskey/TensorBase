/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>

namespace TensorBase
{
  /**
    @brief Class for managing heterogenous Tensors
  */
  template<class... TTables>
  class TensorCollection
  {
  public:
    TensorCollection() = default;  ///< Default constructor
    TensorCollection(TTables&... tTables) { 
      tables_ = std::make_tuple(tTables...); 
    };  ///< Default constructor
    ~TensorCollection() = default; ///< Default destructor

    void setTableNames(const std::vector<std::string>& names) { 
      tables_to_indices_.clear();
      for (const std::string&name : names)
        tables_to_indices_.emplace(name, tables_to_indices_.size());
    }; ///< table names setter
    std::vector<std::string> getTableNames() const { 
      std::vector<std::string> names;
      for (const auto& table_to_index_map : tables_to_indices_)
        names.push_back(table_to_index_map.first);
      return names; 
    }; ///< table neames getter

    int getTableIndex(const std::string& name);

    auto& getTable(const::std::string& name);  ///< non-const TensorTable getter
    const auto& getTable(const::std::string& name) const;  ///< const TensorTable getter

  private:
    std::map<std::string, int> tables_to_indices_;
    std::tuple<std::shared_ptr<TTables>...> tables_; ///< tuple of TensorTables<TensorT, DeviceT, TDim>
    
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
  };
  template<class ...TTables>
  inline int TensorCollection<TTables...>::getTableIndex(const std::string & name)
  {
    if (tables_to_indices_.count(name))
      return tables_to_indices_.at(name);
    else
      return -1;
  }
  template<class ...TTables>
  inline auto& TensorCollection<TTables...>::getTable(const std::string & name)
  {

  }
  template<class ...TTables>
  inline const auto& TensorCollection<TTables...>::getTable(const std::string & name) const
  { // Duplicate of the non-const getter...
    if (tables_to_indices_.count(name)) {
      int index = tables_to_indices_.at(name);
      return std::get<index>(tables_);
    }
  }
};
#endif //TENSORBASE_TENSORCOLLECTION_H