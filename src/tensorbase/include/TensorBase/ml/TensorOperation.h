/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>

namespace TensorBase
{
  /**
    @brief Abstract base class for all Tensor operations involving insertions, deletions, and updates
  */
  template<typename DeviceT>
  class TensorOperation
  {
  public:
    TensorOperation() = default;
    virtual ~TensorOperation() = default;
    virtual void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
    virtual void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
  };

  /**
    @brief Class for appending data to a Tensor
  */
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorAppendToAxis: public TensorOperation<DeviceT> {
  public:
    TensorAppendToAxis() = default;
    TensorAppendToAxis(const std::string& table_name, const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values) :
      table_name_(table_name), axis_name_(axis_name), labels_(labels), values_(values) {};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
  protected:
    std::string table_name_; // Redo/Undo
    std::string axis_name_; // Redo/Undo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Redo/Undo
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Redo/Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // TODO: check that the table_name and axis name exist
    // TODO: check that the dimensions of the values are compatible with a tensor concatenation give the dimensions of the Tensor table and specified axis
    // TODO: check that the dimensions of the labels match the dimensions of the axis labels

    // Append to the axis
    tensor_collection->tables_.at(table_name_)->appendToAxis(axis_name_, labels_, values_->getDataPointer(), indices_, device);
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Delete from the axis
    tensor_collection->tables_.at(table_name_)->deleteFromAxis(axis_name_, indices_, device);
  }

  /**
    @brief Class for deleting data from a Tensor
  */
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorDeleteFromAxis : public TensorOperation<DeviceT> {
  public:
    TensorDeleteFromAxis() = default;
    TensorDeleteFromAxis(const std::string& table_name, const std::string& axis_name, const std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)>& select_function) :
      table_name_(table_name), axis_name_(axis_name), select_function_(select_function) {};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    virtual void allocateMemoryForValues(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
  protected:
    std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)> select_function_; // Redo
    std::string table_name_; // Undo/Redo
    std::string axis_name_; // Undo/Redo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Undo
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // TODO: check that the table_name and axis name exist

    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Extract out the labels to delete from the `indices_view`
    tensor_collection->tables_.at(table_name_)->makeIndicesFromIndicesView(axis_name_, indices_, device);
    tensor_collection->tables_.at(table_name_)->resetIndicesView(axis_name_, device);

    // TOOD: report the number of slices that will be deleted

    // Determine the dimensions of the deletion and allocate to memory
    allocateMemoryForValues(tensor_collection, device);

    // Delete the selected labels
    tensor_collection->tables_.at(table_name_)->deleteFromAxis(axis_name_, indices_, labels_, values_->getDataPointer(), device);
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Insert the deleted labels
    tensor_collection->tables_.at(table_name_)->insertIntoAxis(axis_name_, labels_, values_->getDataPointer(), indices_, device);
  }

  /**
    @brief DefaultDevice specialization of `TensorDeleteFromAxis`
  */
  template<typename LabelsT, typename TensorT, int TDim>
  class TensorDeleteFromAxisDefaultDevice : public TensorDeleteFromAxis<LabelsT, TensorT, Eigen::DefaultDevice, TDim> {
  public:
    using TensorDeleteFromAxis<LabelsT, TensorT, Eigen::DefaultDevice, TDim>::TensorDeleteFromAxis;
    void allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device) override;
  };

  template<typename LabelsT, typename TensorT, int TDim>
  inline void TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, TDim>::allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice & device)
  {
    // Determine the dimensions of the values that will be deleted
    Eigen::array<Eigen::Index, TDim> dimensions_new;
    for (auto& axis_map: tensor_collection->tables_.at(this->table_name_)->getAxes()) {
      dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(axis_map.second->getName())) = axis_map.second->getNLabels();
    }
    dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(this->axis_name_)) -= this->indices_->getTensorSize();

    // Allocate memory for the values
    TensorDataDefaultDevice<TensorT, TDim> values_tmp(dimensions_new);
    values_tmp.setData();
    this->values_ = std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(values_tmp);
  }

  /**
    @brief Class for updating data of a Tensor based on an optional select function

    Use cases:
      1. The update replaces all of the data in the specified `tensor_table` if no select_function is specified
      2. The update a selected contiguous region (i.e., continuous column or row, 2D sheet, 3D cube, etc.,) in the specified `tensor_table`
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorUpdateValues : public TensorOperation<DeviceT> {
  public:
    TensorUpdateValues() = default;
    TensorUpdateValues(const std::string& table_name, const std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)>& select_function, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new) :
      table_name_(table_name), select_function_(select_function), values_new_(values_new) {};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> getValuesOld() const { return values_old_; };
  protected:
    std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)> select_function_; // Redo/Undo
    std::string table_name_; // Undo/Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new_ = nullptr; // Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_old_ = nullptr; // Undo
  };
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdateValues<TensorT, DeviceT, TDim>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // TODO: check that the table_name exist

    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // TODO: check that the dimensions of the values are compatible with the selected Tensor Table Data

    // Update the values with the `values_new` and copy the original values into the `values_old`
    values_old_ = values_new_->copy(device);
    values_new_->syncHAndDData(device);
    values_old_->syncHAndDData(device);
    tensor_collection->tables_.at(table_name_)->updateTensorDataValues(values_new_->getDataPointer(), values_old_->getDataPointer(), device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdateValues<TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Update the values with the `values_old`
    tensor_collection->tables_.at(table_name_)->updateTensorDataValues(values_old_->getDataPointer(), device);
  }

  /**
    @brief Class for setting Tensor data to a specified constant value based on an optional select function

    Use cases: The update replaces all selected data with a particular value in the specified `tensor_table`
  */
  template<typename TensorT, typename DeviceT>
  class TensorUpdateConstant : public TensorOperation<DeviceT> {
  public:
    TensorUpdateConstant() = default;
    TensorUpdateConstant(const std::string& table_name, const std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)>& select_function, const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values_new) :
      table_name_(table_name), select_function_(select_function), values_new_(values_new) {};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> getValuesOld() const { return values_old_; };
  protected:
    std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)> select_function_; // Redo/Undo
    std::string table_name_; // Undo/Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> values_new_ = nullptr; // Redo
    std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> values_old_ = nullptr; // Undo: Sparse table
  };
  template<typename TensorT, typename DeviceT>
  inline void TensorUpdateConstant<TensorT, DeviceT>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // TODO: check that the table_name exist

    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Update the values with the `values_new`
    values_new_->syncHAndDData(device);
    tensor_collection->tables_.at(table_name_)->updateTensorDataConstant(values_new_, values_old_, device);

    // Reset the table indices
    for (auto& axes_map: tensor_collection->tables_.at(table_name_)->getAxes())
      tensor_collection->tables_.at(table_name_)->resetIndicesView(axes_map.first, device);
  }
  template<typename TensorT, typename DeviceT>
  inline void TensorUpdateConstant<TensorT, DeviceT>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Update the values with the `values_old`
    tensor_collection->tables_.at(table_name_)->updateTensorDataFromSparseTensorTable(values_old_, device);
  }

  class TensorAppendToDimension;
  class TensorDeleteFromDimension;

  class TensorAddAxis; // TODO: implement as a Tensor Broadcast
  class TensorDeleteAxis;  // TODO: implement as a Tensor Chip

  /**
    @brief Class for adding a `TensorTable` to a `TensorCollection`
  */
  template<typename T, typename DeviceT>
  class TensorAddTable : public TensorOperation<DeviceT> {
  public:
    TensorAddTable() = default;
    TensorAddTable(const std::shared_ptr<T>& table, const std::string& user_table_name) :
      table_(table), user_table_name_(user_table_name){};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
  protected:
    std::shared_ptr<T> table_; // undo/redo
    std::string user_table_name_; // undo/redo
  };

  template<typename T, typename DeviceT>
  inline void TensorAddTable<T, DeviceT>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    tensor_collection->addTensorTable(table_, user_table_name_);
  }

  template<typename T, typename DeviceT>
  inline void TensorAddTable<T, DeviceT>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    tensor_collection->removeTensorTable(table_->getName());
  }
  
  /**
    @brief Class for removing a `TensorTable` from a `TensorCollection`
  */
  template<typename DeviceT>
  class TensorDropTable : public TensorOperation<DeviceT> {
  public:
    TensorDropTable() = default;
    TensorDropTable(const std::string& table_name) :
      table_name_(table_name) {};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device);
  protected:
    std::string table_name_; // redo
    std::string user_table_name_; // undo
    std::shared_ptr<TensorTableConcept<DeviceT>> table_; // undo
  };
  template<typename DeviceT>
  inline void TensorDropTable<DeviceT>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    // Copy the table and then remove it from the collection
    table_ = tensor_collection->getTensorTableConcept(table_name_);
    user_table_name_ = tensor_collection->getUserNameFromTableName(table_name_);
    tensor_collection->removeTensorTable(table_name_);
  }
  template<typename DeviceT>
  inline void TensorDropTable<DeviceT>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT & device)
  {
    // Restore the table to the collection
    tensor_collection->addTensorTableConcept(table_, user_table_name_);
  }
};
#endif //TENSORBASE_TENSOROPERATION_H