/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>

namespace TensorBase
{
  struct TensorOperationLog {
    TensorOperationLog() = default;
    TensorOperationLog(const int& n_labels_deleted,
      const int& n_labels_updated,
      const int& n_labels_inserted,
      const int& n_indices_deleted,
      const int& n_indices_updated,
      const int& n_indices_inserted,
      const int& n_data_deleted,
      const int& n_data_updated,
      const int& n_data_inserted) {
      n_labels_deleted_ = n_labels_deleted;
      n_labels_updated_ = n_labels_updated;
      n_labels_inserted_ = n_labels_inserted;
      n_indices_deleted_ = n_indices_deleted;
      n_indices_updated_ = n_indices_updated;
      n_indices_inserted_ = n_indices_inserted;
      n_data_deleted_ = n_data_deleted;
      n_data_updated_ = n_data_updated;
      n_data_inserted_ = n_data_inserted;
    }
    int n_labels_deleted_ = 0;
    int n_labels_updated_ = 0;
    int n_labels_inserted_ = 0;
    int n_indices_deleted_ = 0;
    int n_indices_updated_ = 0;
    int n_indices_inserted_ = 0;
    int n_data_deleted_ = 0;
    int n_data_updated_ = 0;
    int n_data_inserted_ = 0;
  };

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
    void setRedoLog(const TensorOperationLog& operation_log) { redo_log_ = operation_log; }
    void setUndoLog(const TensorOperationLog& operation_log) { undo_log_ = operation_log; }
    TensorOperationLog const getRedoLog() const { return redo_log_; }
    TensorOperationLog const getUndoLog() const { return undo_log_; }
    std::string getRedoLogAsString() const;
    std::string getUndoLogAsString() const;
  private:
    TensorOperationLog redo_log_ = TensorOperationLog(); ///< tensor table changes reported by `redo` method
    TensorOperationLog undo_log_ = TensorOperationLog(); ///< tensor table changes reported by `undo` method
  };
  template<typename DeviceT>
  inline std::string TensorOperation<DeviceT>::getRedoLogAsString() const
  {
    std::string report = "Redo Log report:";
    if (redo_log_.n_labels_deleted_) report = report + " " + redo_log_.n_labels_deleted_ + " deleted;";
    if (redo_log_.n_labels_updated_) report = report + " " + redo_log_.n_labels_updated_ + " updated;";
    if (redo_log_.n_labels_inserted_) report = report + " " + redo_log_.n_labels_inserted_ + " inserted;";
    if (redo_log_.n_indices_deleted_) report = report + " " + redo_log_.n_indices_deleted_ + " deleted;";
    if (redo_log_.n_indices_updated_) report = report + " " + redo_log_.n_indices_updated_ + " updated;";
    if (redo_log_.n_indices_inserted_) report = report + " " + redo_log_.n_indices_inserted_ + " inserted;";
    if (redo_log_.n_data_deleted_) report = report + " " + redo_log_.n_data_deleted_ + " deleted;";
    if (redo_log_.n_data_updated_) report = report + " " + redo_log_.n_data_updated_ + " updated;";
    if (redo_log_.n_data_inserted_) report = report + " " + redo_log_.n_data_inserted_ + " inserted;";
    return report;
  }
  template<typename DeviceT>
  inline std::string TensorOperation<DeviceT>::getUndoLogAsString() const
  {
    std::string report = "Undo Log report:";
    if (undo_log_.n_labels_deleted_) report = report + " " + undo_log_.n_labels_deleted_ + " deleted;";
    if (undo_log_.n_labels_updated_) report = report + " " + undo_log_.n_labels_updated_ + " updated;";
    if (undo_log_.n_labels_inserted_) report = report + " " + undo_log_.n_labels_inserted_ + " inserted;";
    if (undo_log_.n_indices_deleted_) report = report + " " + undo_log_.n_indices_deleted_ + " deleted;";
    if (undo_log_.n_indices_updated_) report = report + " " + undo_log_.n_indices_updated_ + " updated;";
    if (undo_log_.n_indices_inserted_) report = report + " " + undo_log_.n_indices_inserted_ + " inserted;";
    if (undo_log_.n_data_deleted_) report = report + " " + undo_log_.n_data_deleted_ + " deleted;";
    if (undo_log_.n_data_updated_) report = report + " " + undo_log_.n_data_updated_ + " updated;";
    if (undo_log_.n_data_inserted_) report = report + " " + undo_log_.n_data_inserted_ + " inserted;";
    return report;
  }

  /**
    @brief Class for appending data to a Tensor
  */
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  class TensorAppendToAxis: public TensorOperation<DeviceT> {
  public:
    TensorAppendToAxis() = default;
    TensorAppendToAxis(const std::string& table_name, const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values) :
      table_name_(table_name), axis_name_(axis_name), labels_(labels), values_(values) {};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
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
		// Check that the table_name exist
		if (tensor_collection->tables_.count(table_name_) < 1) {
			std::cout << "The table " << table_name_ << " does not exist in the collection." << std::endl;
			return;
		}

		// Check that the axis name exist
		try {
			tensor_collection->tables_.at(table_name_)->getDimFromAxisName(axis_name_);
		}
		catch (std::out_of_range& e) {
			std::cout << "The axis " << axis_name_ << " does not exist in the table " << table_name_ << "." << std::endl;
			return;
		}
    // TODO: check that the dimensions of the values are compatible with a tensor concatenation give the dimensions of the Tensor table and specified axis
    // TODO: check that the dimensions of the labels match the dimensions of the axis labels

    // Append to the axis
    labels_->syncDData(device);
    values_->syncDData(device);
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    tensor_collection->tables_.at(table_name_)->appendToAxis(axis_name_, labels_, values_->getDataPointer(), indices_, device);

    // Log the changes
    this->setRedoLog(TensorOperationLog(0,0, labels_->getTensorSize(),0,0,indices_->getTensorSize(),0,0,values_->getTensorSize()));
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorAppendToAxis<LabelsT, TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Delete from the axis
    if (!indices_->getDataStatus().second) indices_->syncHAndDData(device);
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    tensor_collection->tables_.at(table_name_)->deleteFromAxis(axis_name_, indices_, device);

    // Log the changes
    this->setUndoLog(TensorOperationLog(labels_->getTensorSize(), 0, 0, indices_->getTensorSize(), 0, 0, values_->getTensorSize(), 0, 0));
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
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
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
		// Check that the table_name exist
		if (tensor_collection->tables_.count(table_name_) < 1) {
			std::cout << "The table " << table_name_ << " does not exist in the collection." << std::endl;
			return;
		}

		// Check that the axis name exist
		try {
			tensor_collection->tables_.at(table_name_)->getDimFromAxisName(axis_name_);
		}
		catch (std::out_of_range & e) {
			std::cout << "The axis " << axis_name_ << " does not exist in the table " << table_name_ << "." << std::endl;
			return;
		}

    // Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    select_function_(tensor_collection, device);

    // Extract out the labels to delete from the `indices_view`
    tensor_collection->tables_.at(table_name_)->makeIndicesFromIndicesView(axis_name_, indices_, device);
    tensor_collection->tables_.at(table_name_)->resetIndicesView(axis_name_, device);

    // Determine the dimensions of the deletion and allocate to memory
    allocateMemoryForValues(tensor_collection, device);

    // Delete the selected labels
    values_->syncDData(device);
    tensor_collection->tables_.at(table_name_)->deleteFromAxis(axis_name_, indices_, labels_, values_->getDataPointer(), device);

		// Reset the indices view
		for (const auto& axes_to_dims : tensor_collection->tables_.at(table_name_)->getAxesToDims()) {
			tensor_collection->tables_.at(table_name_)->resetIndicesView(axes_to_dims.first, device);
		}

    // Log the changes
    this->setRedoLog(TensorOperationLog(labels_->getTensorSize(), 0, 0, indices_->getTensorSize(), 0, 0, values_->getTensorSize(), 0, 0));
  }
  template<typename LabelsT, typename TensorT, typename DeviceT, int TDim>
  inline void TensorDeleteFromAxis<LabelsT, TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Insert the deleted labels
    indices_->syncDData(device);
    values_->syncDData(device);
    labels_->syncDData(device);
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    tensor_collection->tables_.at(table_name_)->insertIntoAxis(axis_name_, labels_, values_->getDataPointer(), indices_, device);

    // Log the changes
    this->setUndoLog(TensorOperationLog(0, 0, labels_->getTensorSize(), 0, 0, indices_->getTensorSize(), 0, 0, values_->getTensorSize()));
  }

	/**
		@brief Class for updating data of a Tensor based on an optional select function.

		IMPORTANT: the select function must "apply" the select clauses, which will reduce the data and axes in place, and then update the reduced data with the new values

		Use cases:
			1. The update replaces all of the data in the specified `tensor_table` if no select_function is specified
			2. The update a selected contiguous region (i.e., continuous column or row, 2D sheet, 3D cube, etc.,) in the specified `tensor_table`
	*/
	template<typename TensorT, typename DeviceT, int TDim>
	class TensorUpdateSelectValues : public TensorOperation<DeviceT> {
	public:
		TensorUpdateSelectValues() = default;
		TensorUpdateSelectValues(const std::string& table_name, const std::function<void(std::shared_ptr<TensorCollection<DeviceT>> & tensor_collection, DeviceT & device)>& select_function, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new) :
			table_name_(table_name), select_function_(select_function), values_new_(values_new) {};
		void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
		void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
		std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> getValuesOld() const { return values_old_; };
	protected:
		std::function<void(std::shared_ptr<TensorCollection<DeviceT>> & tensor_collection, DeviceT & device)> select_function_; // Redo/Undo
		std::string table_name_; // Undo/Redo
		std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new_ = nullptr; // Redo
		std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_old_ = nullptr; // Undo
	};
	template<typename TensorT, typename DeviceT, int TDim>
	inline void TensorUpdateSelectValues<TensorT, DeviceT, TDim>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
	{
		// Check that the table_name exist
		if (tensor_collection->tables_.count(table_name_) < 1) {
			std::cout << "The table " << table_name_ << " does not exist in the collection." << std::endl;
			return;
		}

		// Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
		select_function_(tensor_collection, device);

		// Check that the dimensions of the values are compatible with the selected Tensor Table Data
		// Mismatch indicates that the select method was written incorrectly
		assert(tensor_collection->tables_.at(table_name_)->getDataTensorSize() == values_new_->getTensorSize());

		// Update the values with the `values_new` and copy the original values into the `values_old`
		values_old_ = values_new_->copyToHost(device);
		values_old_->syncDData(device);
    values_new_->syncDData(device);
		tensor_collection->tables_.at(table_name_)->updateSelectTensorDataValues(values_new_->getDataPointer(), values_old_->getDataPointer(), device);

    // Log the changes
    this->setRedoLog(TensorOperationLog(0, 0, 0, 0, 0, 0, 0, values_new_->getTensorSize(), 0));
	}
	template<typename TensorT, typename DeviceT, int TDim>
	inline void TensorUpdateSelectValues<TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
	{
		// Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
		select_function_(tensor_collection, device);

		// Check that the dimensions of the values are compatible with the selected Tensor Table Data
		// Mismatch indicates that the select method was written incorrectly
		assert(tensor_collection->tables_.at(table_name_)->getDataTensorSize() == values_old_->getTensorSize());

		// Update the values with the `values_old`
    values_old_->syncDData(device);
		tensor_collection->tables_.at(table_name_)->updateSelectTensorDataValues(values_old_->getDataPointer(), device);

    // Log the changes
    this->setUndoLog(TensorOperationLog(0, 0, 0, 0, 0, 0, 0, values_old_->getTensorSize(), 0));
	}

  /**
    @brief Class for updating data of a Tensor based on an optional select function.

		IMPORTant: the select function does not "apply" the select clauses, but instead updates the data based on the `indices_view`

    Use cases:
      1. The update replaces all of the data in the specified `tensor_table` if no select_function is specified
      2. The update a selected contiguous region (i.e., continuous column or row, 2D sheet, 3D cube, etc.,) in the specified `tensor_table`
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorUpdateValues : public TensorOperation<DeviceT> {
  public:
		TensorUpdateValues() = default;
		TensorUpdateValues(const std::string& table_name, const std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)>& select_function, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new) :
      table_name_(table_name), select_function_(select_function), values_new_(values_new){};
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
		std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> getValuesOld() const { return values_old_; };
  protected:
    std::function<void(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)> select_function_; // Redo/Undo
    std::string table_name_; // Undo/Redo
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_new_ = nullptr; // Redo
		std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> values_old_ = nullptr; // Undo: Sparse table
  };
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdateValues<TensorT, DeviceT, TDim>::redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Check that the table_name exist
		if (tensor_collection->tables_.count(table_name_) < 1) {
			std::cout << "The table " << table_name_ << " does not exist in the collection." << std::endl;
			return;
		}

    // Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    select_function_(tensor_collection, device);
		
		// Update the values with the `values_new` and copy the original values into the `values_old`
    values_new_->syncDData(device);
		tensor_collection->tables_.at(table_name_)->updateTensorDataValues(values_new_->getDataPointer(), values_old_, device);

		// Reset the indices view
		for (const auto& axes_to_dims : tensor_collection->tables_.at(table_name_)->getAxesToDims()) {
			tensor_collection->tables_.at(table_name_)->resetIndicesView(axes_to_dims.first, device);
		}

    // Log the changes
    this->setRedoLog(TensorOperationLog(0, 0, 0, 0, 0, 0, 0, values_new_->getTensorSize(), 0));
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorUpdateValues<TensorT, DeviceT, TDim>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    select_function_(tensor_collection, device);

		// Update the values with the `values_old`
    values_old_->syncDData(device);
		tensor_collection->tables_.at(table_name_)->updateTensorDataFromSparseTensorTable(values_old_, device);

		// Reset the indices view
		for (const auto& axes_to_dims : tensor_collection->tables_.at(table_name_)->getAxesToDims()) {
			tensor_collection->tables_.at(table_name_)->resetIndicesView(axes_to_dims.first, device);
		}

    // Log the changes
    this->setUndoLog(TensorOperationLog(0, 0, 0, 0, 0, 0, 0, values_old_->getDataTensorSize(), 0));
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
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
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
		// Check that the table_name exist
		if (tensor_collection->tables_.count(table_name_) < 1) {
			std::cout << "The table " << table_name_ << " does not exist in the collection." << std::endl;
			return;
		}

    // Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    select_function_(tensor_collection, device);

    // Update the values with the `values_new`
    values_new_->syncDData(device);
    tensor_collection->tables_.at(table_name_)->updateTensorDataConstant(values_new_, values_old_, device);

    // Reset the table indices
    for (const auto& axes_to_dims : tensor_collection->tables_.at(table_name_)->getAxesToDims())
      tensor_collection->tables_.at(table_name_)->resetIndicesView(axes_to_dims.first, device);

    // Log the changes
    this->setRedoLog(TensorOperationLog(0, 0, 0, 0, 0, 0, 0, values_old_->getDataTensorSize(), 0));
  }
  template<typename TensorT, typename DeviceT>
  inline void TensorUpdateConstant<TensorT, DeviceT>::undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device)
  {
    // Execute the select methods on the tensor_collection
    tensor_collection->tables_.at(table_name_)->syncAxesAndIndicesDData(device);
    tensor_collection->tables_.at(table_name_)->syncDData(device);
    select_function_(tensor_collection, device);

    // Update the values with the `values_old`
    values_old_->syncDData(device);
    tensor_collection->tables_.at(table_name_)->updateTensorDataFromSparseTensorTable(values_old_, device);

		// Reset the table indices
		for (const auto& axes_to_dims : tensor_collection->tables_.at(table_name_)->getAxesToDims())
			tensor_collection->tables_.at(table_name_)->resetIndicesView(axes_to_dims.first, device);

    // Log the changes
    this->setUndoLog(TensorOperationLog(0, 0, 0, 0, 0, 0, 0, values_old_->getDataTensorSize(), 0));
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
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
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
    void redo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
    void undo(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override;
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