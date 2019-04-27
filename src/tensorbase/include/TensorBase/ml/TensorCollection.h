/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>
#include <map>

namespace TensorBase
{
  /**
    @brief Class for managing heterogenous Tensors
  */
  class TensorCollection
  {
    /// The erasure interface to Tensor Table
    class TensorTableConcept {
    public: 
      virtual std::string getName() const = 0; 
      virtual std::map<std::string, std::shared_ptr<TensorAxis>>& getAxes() = 0;
      virtual std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndices() = 0;
      virtual std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndicesView() = 0;
      virtual std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsModified() = 0;
      virtual std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getInMemory() = 0;
      virtual std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsShardable() = 0;
    };

    /// The erasure wrapper around the Tensor Table interface
    template<typename T>
    class TensorTableWrapper : public TensorTableConcept {
      const std::shared_ptr<T> tensor_table_;
    public:
      TensorTableWrapper(const std::shared_ptr<T>& tensor_table) : tensor_table_(tensor_table) {};
      std::string getName() const { return tensor_table_->getName(); };
      std::map<std::string, std::shared_ptr<TensorAxis>>& getAxes() { return tensor_table_->getAxes(); };
      std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndices() { return tensor_table_->getIndices(); };
      std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndicesView() { return tensor_table_->getIndicesView(); };
      std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsModified() { return tensor_table_->getIsModified(); };
      std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getInMemory() { return tensor_table_->getInMemory(); };
      std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsShardable() { return tensor_table_->getIsShardable(); };
    };
  public:
    TensorCollection() = default;  ///< Default constructor
    ~TensorCollection() = default; ///< Default destructor

    template<typename T>
    void addTensorTable(const std::shared_ptr<T>& tensor_table);

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

    std::map<std::string, std::shared_ptr<TensorTableConcept>> tensor_tables_; ///< tuple of std::shared_ptr TensorTables<TensorT, DeviceT, TDim>
  };

  template<typename T>
  inline void TensorCollection::addTensorTable(const std::shared_ptr<T>& tensor_table)
  {
    tensor_tables_.emplace(tensor_table->getName(), std::shared_ptr<TensorTableConcept>(new TensorTableWrapper<T>(tensor_table)));
  }

  inline std::vector<std::string> TensorBase::TensorCollection::getTableNames() const
  {
    std::vector<std::string> names;
    for (const auto& ttable : tensor_tables_) {
      names.push_back(ttable.second->getName());
    }
    return names;
  }
};
#endif //TENSORBASE_TENSORCOLLECTION_H