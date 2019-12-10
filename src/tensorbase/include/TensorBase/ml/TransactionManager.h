/**TODO:  Add copyright*/

#ifndef TENSORBASE_TRANSACTIONMANAGER_H
#define TENSORBASE_TRANSACTIONMANAGER_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorOperation.h>
#include <deque>
#include <TensorBase/io/TensorCollectionFile.h>

namespace TensorBase
{
  /**
    @brief Class for managing heterogenous Tensors
  */
  template<typename DeviceT>
  class TransactionManager
  {
  public:
    TransactionManager() = default;  ///< Default constructor
    ~TransactionManager() = default; ///< Default destructor

    void setTensorCollection(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection) { tensor_collection_ = tensor_collection; }; ///< tensor_collection setter
    std::shared_ptr<TensorCollection<DeviceT>> getTensorCollection() { return tensor_collection_; }; ///< tensor_collection getter

    void setMaxOperations(const int& max_operations) { max_operations_ = max_operations; }; ///< max_operations setter
    int getMaxOperations() const { return max_operations_; }; ///< max_operations getter

    int getCurrentIndex() const { return current_index_; };  ///< current_index getter

    /*
    @brief Adds the operation to the queue

    TODO: add tests

    @returns True if successfull
    */
    bool addOperation(std::shared_ptr<TensorOperation<DeviceT>>& tensor_operation);

    /*
    @brief Execute the operations in the queue in order (if devices.size() == 1)
      or in parallel (if devices.size() > 1)

    TODO: implement
    - launch a new thread for each device
    - launch all operations that affect a different table in parallel

    @returns True if successfull
    */
    //bool executeOperations(std::vector<DeviceT>& devices);

    /*
    @brief Adds the operation to the history, and then executes the operation
      on the managed TensorCollection

    @returns True if successfull
    */
    bool executeOperation(std::shared_ptr<TensorOperation<DeviceT>>& tensor_operation, DeviceT& device);

    /*
    @brief Undoes the previous operation (in memory)

    @returns True if successfull
    */
    bool undo(DeviceT& device);

    /*
    @brief Redoes the previously undone operation (in memory)

    @returns True if successfull
    */
    bool redo(DeviceT& device);

    /*
    @brief Commits all changes made to the tensor collection in memory
      by writting the changed tensor collection data to disk

    The method clears the tensor_operations_ list and resets the current_index to 0

    @returns True if successfull
    */
    bool commit(DeviceT& device);

    /*
    @brief Undoes all changes made to the tensor collection in memory
      to the previous commit

    @returns True if successfull
    */
    bool rollback(DeviceT& device);

    void clear(); ///< clear the tensor operations, commit indices, and current_index_

  protected:
    std::shared_ptr<TensorCollection<DeviceT>> tensor_collection_; ///< the managed TensorCollection object
    int max_operations_ = 25; ///< the maximum size of `tensor_operations_`
    int current_index_ = -1; ///< current position in the tensor_operations_ vector
    std::deque<std::shared_ptr<TensorOperation<DeviceT>>> tensor_operations_; ///< vector of tensor operations
  };

  template<typename DeviceT>
  inline bool TransactionManager<DeviceT>::addOperation(std::shared_ptr<TensorOperation<DeviceT>>& tensor_operation)
  {
    // Check if the operation history limit has been reached
    if (tensor_operations_.size() >= max_operations_) {
      tensor_operations_.pop_front();
      --current_index_;
    }

    // Add the operation to the history
    tensor_operations_.push_back(tensor_operation);
  }

  template<typename DeviceT>
  inline bool TransactionManager<DeviceT>::executeOperation(std::shared_ptr<TensorOperation<DeviceT>>& tensor_operation, DeviceT& device)
  {
    // Check if the operation history limit has been reached
    if (tensor_operations_.size() >= max_operations_) {
      tensor_operations_.pop_front();
    }

    // Add the operation to the history
    tensor_operations_.push_back(tensor_operation);
    current_index_ = tensor_operations_.size() - 1;

    // Execute the operation
    try {
      tensor_operation->redo(tensor_collection_, device);
      return true;
    }
    catch (const std::exception& e) {
      std::cout << "Exception: " << e.what() << std::endl;
      return false;
    }
  }
  template<typename DeviceT>
  inline bool TransactionManager<DeviceT>::undo(DeviceT & device)
  {
    // Check that the operations history is not empty
    if (current_index_ == -1) {
      std::cout << "There are no operations to undo." << std::endl;
      return false;
    }

    // Execute undo
    try {
      tensor_operations_.at(current_index_)->undo(tensor_collection_, device);
      --current_index_;
      return true;
    }
    catch (const std::exception& e) {
      std::cout << "Exception: " << e.what() << std::endl;
      return false;
    }
  }
  template<typename DeviceT>
  inline bool TransactionManager<DeviceT>::redo(DeviceT & device)
  {
    // Check that no operation in the operations history has been undone
    if (tensor_operations_.size() == current_index_ + 1) {
      std::cout << "There are no operations to redo." << std::endl;
      return false;
    }

    // Execute redo
    try {
      tensor_operations_.at(current_index_ + 1)->redo(tensor_collection_, device);
      ++current_index_;
      return true;
    }
    catch (const std::exception& e) {
      std::cout << "Exception: " << e.what() << std::endl;
      return false;
    }
  }
  template<typename DeviceT>
  inline bool TransactionManager<DeviceT>::commit(DeviceT& device)
  {
    // Write the current tensor_collection to disk
    TensorCollectionFile<DeviceT> data;
    std::string filename = tensor_collection_->getName() + ".TensorCollection";
    data.storeTensorCollectionBinary(filename, tensor_collection_, device);

    // Reset all is_modified attributes to false
    for (auto& table_map : tensor_collection_->tables_) {
      for (auto& is_modified_map : table_map.second->getIsModified()) {        
        Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified_values(is_modified_map.second->getDataPointer().get(), is_modified_map.second->getTensorSize());
        is_modified_values.device(device) = is_modified_values.constant(0);
      }
    }

    // Clear the operations
    clear();
    return true;
  }

  template<typename DeviceT>
  inline bool TransactionManager<DeviceT>::rollback(DeviceT & device)
  {
    // Check that the operations history is not empty
    if (current_index_ == 0) {
      std::cout << "There are no operations to rollback." << std::endl;
      return false;
    }

    // Execute undo for all operations in the history
    try {
      for (; current_index_ >= 0; --current_index_) {
        tensor_operations_.at(current_index_)->undo(tensor_collection_, device);
      }
      clear();
      return true;
    }
    catch (const std::exception& e) {
      std::cout << "Exception: " << e.what() << std::endl;
      return false;
    }
  }

  template<typename DeviceT>
  inline void TransactionManager<DeviceT>::clear()
  {
    tensor_operations_.clear();
    current_index_ = -1;
  }
};
#endif //TENSORBASE_TRANSACTIONMANAGER_H