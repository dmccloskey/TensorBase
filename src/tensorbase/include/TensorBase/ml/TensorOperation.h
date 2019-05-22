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
    virtual void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device) = 0;
    virtual void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device) = 0;
  };

  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorAppendToAxis: public TensorOperation<DeviceT> {
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    std::string table_name_; // Redo/Undo
    std::string axis_name_; // Redo/Undo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Redo/Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Redo/Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        //tensor_collection.tables_.at(table_name)->appendToAxis(
        //  axis_name_, labels_, values_->getDataPointer(), values_->getDimensions(), device); // TODO
      }
    }
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        //tensor_collection.tables_.at(table_name)->deleteFromAxis(
        //  axis_name_, labels_, device); // TODO
      }
    }
  }

  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorDeleteFromAxis : public TensorOperation<DeviceT> {
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    std::function<void(TensorCollection<DeviceT>& tensor_collection, DeviceT& device)> select_function_; // Redo
    std::string table_name_; // Undo/Redo
    std::string axis_name_; // Undo/Redo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Delete the selected labels
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        //tensor_collection.tables_.at(table_name)->deleteFromAxis(
        //  axis_name_, labels_, values_, device); // TODO overload that returns the values that were deleted
      }
    }
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        //tensor_collection.tables_.at(table_name)->appendToAxis(
        //  axis_name_, labels_, values_, device); // TODO
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  class TensorUpdate : public TensorOperation<DeviceT> {
  public:
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
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

    // Update the values with the `values_new`
    values_old_ = values_new_->copy(device);
    values_new_->syncHAndDData(device);
    values_old_->syncHAndDData(device);
    tensor_collection.tables_.at(table_name_)->updateTensorDataConcept(values_new_->getDataPointer(), values_old_->getDataPointer(), device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Update the values with the `values_old`
    tensor_collection.tables_.at(table_name_)->updateTensorDataConcept(values_old_->getDataPointer(), device);
  }

  class TensorAppendToDimension;
  class TensorDeleteFromDimension;

  class TensorAddAxis; // TODO: implement as a Tensor Broadcast
  class TensorDeleteAxis;  // TODO: implement as a Tensor Chip
  
  class TensorAddTables;
  class TensorDropTables;
};
#endif //TENSORBASE_TENSOROPERATION_H