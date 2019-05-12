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

  /*TODO: Move to seperate files!*/
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorAppendToAxis {
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
        tensor_collection.tables_.at(table_name)->appendToAxis(
          axis_name_, labels_, values_, device);
      }
    }
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection & tensor_collection)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        tensor_collection.tables_.at(table_name)->deleteFromAxis(
          axis_name_, labels_, device);
      }
    }
  }

  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorDeleteFromAxis {
    void redo(TensorCollection& tensor_collection);
    void undo(TensorCollection& tensor_collection);
    // TODO: select clauses for Redo
    std::string table_name; // Undo/Redo
    std::string axis_name; // Undo/Redo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values; // Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection & tensor_collection)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        tensor_collection.tables_.at(table_name)->deleteFromAxis(
          axis_name_, labels_, values_, device); // optional overload that returns the labels and values that were deleted
      }
    }
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection & tensor_collection)
  {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(table_name_)->getAxes()) {
      if (axis.first == axis_name_) {
        tensor_collection.tables_.at(table_name)->appendToAxis(
          axis_name_, labels_, values_, device);
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  class TensorUpdate {
    void redo(TensorCollection& tensor_collection);
    void undo(TensorCollection& tensor_collection);
    // TODO Select statements
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new; // Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_old; // Undo
  };
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::redo(TensorCollection & tensor_collection)
  {
    // TODO: execute select statements
    // TODO: update the values with the `values_new`
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::undo(TensorCollection & tensor_collection)
  {
    // TODO: execute select statements
    // TODO: update the values with the `values_old`
  }

  class TensorAppendToDimension;
  class TensorDeleteFromDimension;

  class TensorAddAxis;
  class TensorDeleteAxis;
  
  class TensorAddTables;
  class TensorDropTables;
};
#endif //TENSORBASE_TENSOROPERATION_H