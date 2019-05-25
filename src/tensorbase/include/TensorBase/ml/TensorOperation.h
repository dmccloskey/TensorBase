/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>

namespace TensorBase
{
  /**
    @brief Abstract base class for all Tensor Insert/Update/Delete operations
  */
  template<typename DeviceT>
  class TensorOperation
  {
  public:
    TensorOperation() = default;
    virtual ~TensorOperation() = default;
    virtual void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device) = 0;
    virtual void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device) = 0;
  };

  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorAppendToAxis: public TensorOperation<DeviceT> {
  public:
    TensorAppendToAxis() = default;
    TensorAppendToAxis(const std::string& table_name, const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values) :
      table_name_(table_name), axis_name_(axis_name), labels_(labels), values_(values) {};
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
  private:
    std::string table_name_; // Redo/Undo
    std::string axis_name_; // Redo/Undo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Redo/Undo
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Redo/Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Append to the axis
    tensor_collection.tables_.at(table_name_)->appendToAxis(axis_name_, labels_, values_->getDataPointer(), indices_, device);
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Delete from the axis
    tensor_collection.tables_.at(table_name_)->deleteFromAxis(axis_name_, indices_, device);
  }

  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorDeleteFromAxis : public TensorOperation<DeviceT> {
  public:
    TensorDeleteFromAxis() = default;
    TensorDeleteFromAxis(const std::string& table_name, const std::string& axis_name, const std::function<void(TensorCollection<DeviceT>& tensor_collection, DeviceT& device)>& select_function) :
      table_name_(table_name), axis_name_(axis_name), select_function_(select_function) {};
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
  private:
    std::function<void(TensorCollection<DeviceT>& tensor_collection, DeviceT& device)> select_function_; // Redo
    std::string table_name_; // Undo/Redo
    std::string axis_name_; // Undo/Redo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Undo
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Delete the selected labels
    //tensor_collection.tables_.at(table_name)->deleteFromAxis(axis_name_, indices_, labels_, values_->getDataPointer(), device); // TODO overload that returns the values that were deleted
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Append the deleted labels
    //tensor_collection.tables_.at(table_name)->insertIntoAxis(axis_name_, indices_, labels_, values_->getDataPointer(), device); // TODO
  }

  template<typename TensorT, typename DeviceT, int TDim>
  class TensorUpdate : public TensorOperation<DeviceT> {
  public:
    TensorUpdate() = default;
    TensorUpdate(const std::string& table_name, const std::function<void(TensorCollection<DeviceT>& tensor_collection, DeviceT& device)>& select_function, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new) :
      table_name_(table_name), select_function_(select_function), values_new_(values_new) {};
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> getValuesOld() const { return values_old_; };
  private:
    std::function<void(TensorCollection<DeviceT>& tensor_collection, DeviceT& device)> select_function_; // Redo/Undo
    std::string table_name_; // Undo/Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new_; // Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_old_; // Undo
  };
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Update the values with the `values_new` and copy the original values into the `values_old`
    values_old_ = values_new_->copy(device);
    values_new_->syncHAndDData(device);
    values_old_->syncHAndDData(device);
    tensor_collection.tables_.at(table_name_)->updateTensorData(values_new_->getDataPointer(), values_old_->getDataPointer(), device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Update the values with the `values_old`
    tensor_collection.tables_.at(table_name_)->updateTensorData(values_old_->getDataPointer(), device);
  }

  class TensorAppendToDimension;
  class TensorDeleteFromDimension;

  class TensorAddAxis; // TODO: implement as a Tensor Broadcast
  class TensorDeleteAxis;  // TODO: implement as a Tensor Chip
  
  class TensorAddTables;
  class TensorDropTables;
};
#endif //TENSORBASE_TENSOROPERATION_H