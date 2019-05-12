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
  class TensorOperation
  {
    virtual void redo(TensorCollection& tensor_collection) = 0;
    virtual void undo(TensorCollection& tensor_collection) = 0;
  };

  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorAppendToAxis: public TensorOperation {
    void redo(TensorCollection& tensor_collection);
    void undo(TensorCollection& tensor_collection);
    std::string table_name_; // Redo/Undo
    std::string axis_name_; // Redo/Undo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Redo/Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Redo/Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection & tensor_collection)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        //tensor_collection.tables_.at(table_name)->appendToAxis(
        //  axis_name_, labels_, values_, device); // TODO
      }
    }
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection & tensor_collection)
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
  class TensorDeleteFromAxis : public TensorOperation {
    void redo(TensorCollection& tensor_collection);
    void undo(TensorCollection& tensor_collection);
    std::function<void(TensorCollection& tensor_collection)> select_function_; // Redo
    std::string table_name_; // Undo/Redo
    std::string axis_name_; // Undo/Redo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection & tensor_collection)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection);

    // Delete the selected labels
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        //tensor_collection.tables_.at(table_name)->deleteFromAxis(
        //  axis_name_, labels_, values_, device); // TODO optional overload that returns the labels and values that were deleted
      }
    }
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection & tensor_collection)
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
  class TensorUpdate : public TensorOperation {
    void redo(TensorCollection& tensor_collection);
    void undo(TensorCollection& tensor_collection);
    std::function<void(TensorCollection& tensor_collection)> select_function_; // Redo/Undo
    std::string table_name_; // Undo/Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new_; // Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_old_; // Undo
  };
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::redo(TensorCollection & tensor_collection)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection);

    // Update the values with the `values_new`
    //tensor_collection.tables_.at(table_name)->updateTensorData(values_new_, device); // TODO
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::undo(TensorCollection & tensor_collection)
  {
    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection);

    // Update the values with the `values_old`
    //tensor_collection.tables_.at(table_name)->updateTensorData(values_old, device); // TODO
  }

  class TensorAppendToDimension;
  class TensorDeleteFromDimension;

  class TensorAddAxis;
  class TensorDeleteAxis;
  
  class TensorAddTables;
  class TensorDropTables;
};
#endif //TENSORBASE_TENSOROPERATION_H