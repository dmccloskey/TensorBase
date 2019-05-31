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
  protected:
    std::string table_name_; // Redo/Undo
    std::string axis_name_; // Redo/Undo
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_; // Redo/Undo
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_; // Undo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_; // Redo/Undo
  };
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // TODO: check that the table_name and axis name exist
    // TODO: check that the dimensions of the values are compatible with a tensor concatenation give the dimensions of the Tensor table and specified axis
    // TODO: check that the dimensions of the labels match the dimensions of the axis labels

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
    virtual void allocateMemoryForValues(TensorCollection<DeviceT>& tensor_collection, DeviceT& device) = 0;
  protected:
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
    // TODO: check that the table_name and axis name exist

    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // Extract out the labels to delete from the `indices_view`
    tensor_collection.tables_.at(table_name_)->makeIndicesFromIndicesView(axis_name_, indices_, device);
    tensor_collection.tables_.at(table_name_)->resetIndicesView(axis_name_, device);

    // TOOD: report the number of slices that will be deleted

    // Determine the dimensions of the deletion and allocate to memory
    allocateMemoryForValues(tensor_collection, device);

    // Delete the selected labels
    tensor_collection.tables_.at(table_name_)->deleteFromAxis(axis_name_, indices_, labels_, values_->getDataPointer(), device);
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // Insert the deleted labels
    tensor_collection.tables_.at(table_name_)->insertIntoAxis(axis_name_, labels_, values_->getDataPointer(), indices_, device);
  }

  template<typename LabelsT, typename TensorT, int TDim>
  class TensorDeleteFromAxisDefaultDevice : public TensorDeleteFromAxis<LabelsT, TensorT, Eigen::DefaultDevice, TDim> {
  public:
    using TensorDeleteFromAxis<LabelsT, TensorT, Eigen::DefaultDevice, TDim>::TensorDeleteFromAxis;
    void allocateMemoryForValues(TensorCollection<Eigen::DefaultDevice>& tensor_collection, Eigen::DefaultDevice& device) override;
  };

  template<typename LabelsT, typename TensorT, int TDim>
  inline void TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, TDim>::allocateMemoryForValues(TensorCollection<Eigen::DefaultDevice>& tensor_collection, Eigen::DefaultDevice & device)
  {
    // Determine the dimensions of the values that will be deleted
    Eigen::array<Eigen::Index, TDim> dimensions_new;
    for (auto& axis_map: tensor_collection.tables_.at(this->table_name_)->getAxes()) {
      dimensions_new.at(tensor_collection.tables_.at(this->table_name_)->getDimFromAxisName(axis_map.second->getName())) = axis_map.second->getNLabels();
    }
    dimensions_new.at(tensor_collection.tables_.at(this->table_name_)->getDimFromAxisName(this->axis_name_)) -= this->indices_->getTensorSize();

    // Allocate memory for the values
    TensorDataDefaultDevice<TensorT, TDim> values_tmp(dimensions_new);
    values_tmp.setData();
    this->values_ = std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(values_tmp);
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
  protected:
    std::function<void(TensorCollection<DeviceT>& tensor_collection, DeviceT& device)> select_function_; // Redo/Undo
    std::string table_name_; // Undo/Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new_; // Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_old_; // Undo
  };
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdate<TensorT, DeviceT, TDim>::redo(TensorCollection<DeviceT> & tensor_collection, DeviceT& device)
  {
    // TODO: check that the table_name exist

    // Execute the select methods on the tensor_collection
    select_function_(tensor_collection, device);

    // TODO: check that the dimensions of the values are compatible with the selected Tensor Table Data

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
  
  template<typename T, typename DeviceT>
  class TensorAddTable : public TensorOperation<DeviceT> {
  public:
    TensorAddTable() = default;
    TensorAddTable(const std::shared_ptr<T>& table) :
      table_(table) {};
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
  protected:
    std::shared_ptr<T> table_; // undo/redo
  };

  template<typename T, typename DeviceT>
  inline void TensorAddTable<T, DeviceT>::redo(TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    tensor_collection.addTensorTable(table_);
  }

  template<typename T, typename DeviceT>
  inline void TensorAddTable<T, DeviceT>::undo(TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    tensor_collection.removeTensorTable(table_->getName());
  }

  template<typename DeviceT>
  class TensorDropTable : public TensorOperation<DeviceT> {
  public:
    TensorDropTable() = default;
    TensorDropTable(const std::string& table_name) :
      table_name_(table_name) {};
    void redo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
    void undo(TensorCollection<DeviceT>& tensor_collection, DeviceT& device);
  protected:
    std::string table_name_; // redo
    std::shared_ptr<TensorTableConcept<DeviceT>> table_; // undo
  };
  template<typename DeviceT>
  inline void TensorDropTable<DeviceT>::redo(TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    // Copy the table and then remove it from the collection
    table_ = tensor_collection.getTensorTableConcept(table_name_);
    tensor_collection.removeTensorTable(table_name_);
  }
  template<typename DeviceT>
  inline void TensorDropTable<DeviceT>::undo(TensorCollection<DeviceT>& tensor_collection, DeviceT & device)
  {
    // Restore the table to the collection
    tensor_collection.addTensorTableConcept(table_);
  }
};
#endif //TENSORBASE_TENSOROPERATION_H