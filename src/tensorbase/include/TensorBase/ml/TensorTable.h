/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLE_H
#define TENSORBASE_TENSORTABLE_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxisConcept.h>
#include <TensorBase/ml/TensorClauses.h>
#include <map>
#include <array>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/array.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    @brief Class for managing Tensor data and associated Axes
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorTable
  {
  public:
    TensorTable() = default;  ///< Default constructor
    TensorTable(const std::string& name) : name_(name) {};
    virtual ~TensorTable() = default; ///< Default destruct

    template<typename TensorTOther, typename DeviceTOther, int TDimOther>
    inline bool operator==(const TensorTable<TensorTOther, DeviceTOther, TDimOther>& other) const
    {
      bool meta_equal = std::tie(id_, name_, dimensions_, axes_to_dims_) == std::tie(other.id_, other.name_, other.dimensions_, other.axes_to_dims_);
      auto compare_maps = [](auto lhs, auto rhs) {return *(lhs.second.get()) == *(rhs.second.get()); };
      bool axes_equal = std::equal(axes_.begin(), axes_.end(), other.axes_.begin(), compare_maps);
      bool indices_equal = std::equal(indices_.begin(), indices_.end(), other.indices_.begin(), compare_maps);
      bool indices_view_equal = std::equal(indices_view_.begin(), indices_view_.end(), other.indices_view_.begin(), compare_maps);
      bool is_shardable_equal = std::equal(is_shardable_.begin(), is_shardable_.end(), other.is_shardable_.begin(), compare_maps);
      bool in_memory_equal = std::equal(in_memory_.begin(), in_memory_.end(), other.in_memory_.begin(), compare_maps);
      bool is_modified_equal = std::equal(is_modified_.begin(), is_modified_.end(), other.is_modified_.begin(), compare_maps);
      return meta_equal && axes_equal && indices_equal && indices_view_equal && is_shardable_equal
        && in_memory_equal && is_modified_equal;
    }

    inline bool operator!=(const TensorTable& other) const
    {
      return !(*this == other);
    }

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

    template<typename T>
    void addTensorAxis(const std::shared_ptr<T>& tensor_axis);  ///< Tensor axis adder

    /**
      @brief Tensor Axes setter

      The method sets the tensor axes and initializes the indices, indices_view, is_modified, in_memory, and
      is_shardable attributes after all axes have been added
    */
    virtual void setAxes() = 0;

    /**
      @brief DeviceT specific initializer
    */
    virtual void initData() = 0;

    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>>& getAxes() { return axes_; }; ///< axes getter
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> getAxes() const { return axes_; }; ///< axes getter

    // TODO: combine into a single member called `indices` of Tensor dimensions 5 x indices_length
    //       in order to improve performance of all TensorInsert and TensorDelete methods
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndices() { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndicesView() { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsModified() { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getInMemory() { return in_memory_; }; ///< in_memory getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsShardable() { return is_shardable_; }; ///< is_shardable getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndices() const { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndicesView() const { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIsModified() const { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getInMemory() const { return in_memory_; }; ///< in_memory getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIsShardable() const { return is_shardable_; }; ///< is_shardable getter

    Eigen::array<Eigen::Index, TDim>& getDimensions() { return dimensions_; }  ///< dimensions getter
    int getDimFromAxisName(const std::string& axis_name) { return axes_to_dims_.at(axis_name); }
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& getData() { return data_; }; ///< data getter
    void clear();  ///< clears the axes and all associated data

    bool syncIndicesHAndDData(DeviceT& device); ///< Sync the host and device indices data
    void setIndicesDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices data
    std::map<std::string, std::pair<bool, bool>> getIndicesDataStatus(); ///< Get the status of the host and device indices data

    bool syncIndicesViewHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setIndicesViewDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getIndicesViewDataStatus(); ///< Get the status of the host and device indices view data

    bool syncIsModifiedHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setIsModifiedDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getIsModifiedDataStatus(); ///< Get the status of the host and device indices view data

    bool syncInMemoryHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setInMemoryDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getInMemoryDataStatus(); ///< Get the status of the host and device indices view data

    bool syncIsShardableHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setIsShardableDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getIsShardableDataStatus(); ///< Get the status of the host and device indices view data

    bool syncAxesHAndDData(DeviceT& device); ///< Sync the host and device axes data
    void setAxesDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device axes data
    std::map<std::string, std::pair<bool, bool>> getAxesDataStatus(); ///< Get the status of the host and device axes data

    bool syncHAndDData(DeviceT& device) { return data_->syncHAndDData(device); };  ///< Sync the host and device tensor data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { data_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device tensor data
    std::pair<bool, bool> getDataStatus() { return data_->getDataStatus(); };   ///< Get the status of the host and device tensor data

    /*
    @brief Select Tensor Axis that will be included in the view

    The selection is done according to the following algorithm:
      1. The `select_labels` are reshaped and broadcasted to a 2D tensor of labels_size x select_labels_size
      2. The `labels` for the axis are broadcasted to a 2D tensor of labels_size x select_labels_size
      3. The `indices` for the axis are broadcasted to a 2D tensor of labels_size x select_labels_size
      4. The indices that correspond to matches between the `select_labels` and `labels` bcast tensors are selected
      5. The result is reduced and normalized to a 1D Tensor of 0 or 1 of size labels_size
      6. The `indices_view` is updated by multiplication by the 1D select Tensor

    @param[in] axis_name
    @param[in] dimension_index
    @param[in] select_labels_data
    @param[in] device
    */
    template<typename LabelsT>
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, DeviceT& device);
    
    void resetIndicesView(const std::string& axis_name, DeviceT& device); ///< copy over the indices values to the indices view
    void zeroIndicesView(const std::string& axis_name, DeviceT& device); ///< set the indices view to zero

    /*
    @brief Order Tensor Axis View

    @param[in] axis_name
    @param[in] dimension_index
    @param[in] select_labels
    @param[in] device
    */
    template<typename LabelsT>
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device);

    virtual int getFirstIndexFromIndicesView(const std::string& axis_name, DeviceT& device) = 0; ///< Helper method to get the first index

    /*
    @brief Apply a where selection clause to the Tensor Axis View

    @param[in] axis_name
    @param[in] dimension_index
    @param[in] select_labels
    @param[in] values
    @param[in] comparitor
    @param[in] modifier
    @param[in] within_continuator
    @param[in] prepend_continuator
    @param[in] device
    */
    template<typename LabelsT, typename T>
    void whereIndicesViewConcept(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<T, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device);
    template<typename LabelsT>
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device);

    /*
    @brief Broadcast the axis indices view across the entire tensor
      and allocate to memory

    @param[out] indices_view_bcast ([in] empty pointer)
    @param[in] axis_name
    */
    virtual void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, const std::string& axis_name, DeviceT& device) = 0;
    
    /*
    @brief Extract data from the Tensor based on a select index tensor
      and allocate to memory

    @param[in] indices_view_bcast The indices (0 or 1) to select from
    @param[out] tensor_select The selected tensor with reduced dimensions according to the indices_view_bcast indices (empty pointer)
    @param[in] axis_name The name of the axis to reduce along
    @param[in] n_select The size of the reduced dimensions
    @param[in] device
    */
    virtual void reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, DeviceT& device) = 0;

    /*
    @brief Select indices from the tensor based on a selection criteria
      and allocate to memory

    @param[out] indices_select The indices that passed or did not pass the selection criteria ([in] empty pointer)
    @param[in] values_select The values to use for comparison
    @param[in] tensor_select The to apply the selection criteria to
    @param[in] axis_name The name of the axis to reduce along
    @param[in] n_select The size of the reduced dimensions
    @param[in] comparitor The logical comparitor to apply
    @param[in] modifier The logical modifier to apply to the comparitor (i.e., Not; currently not implemented)
    @param[in] device
    */
    virtual void selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, DeviceT& device) = 0;

    /*
    @brief Apply the indices select to the indices view for the respective axis
      using the logical continuator and prepender

    @param[in] indices_select The indices that passed or did not pass the selection criteria
    @param[in] axis_name_select The name of the axis that the selection is being applied against
    @param[in] axis_name The name of the axis to apply the selection on
    @param[in] within_continuator
    @param[in] prepend_continuator
    @param[in] device
    */
    void applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices_select, const std::string & axis_name_select, const std::string& axis_name, const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device);
    void applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, 2>>& indices_select, const std::string & axis_name_select, const std::string& axis_name, const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device);
    template<int TDim_ = TDim, typename = std::enable_if_t<(TDim_ > 2)>>
    void applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const std::string & axis_name_select, const std::string& axis_name, const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device);

    /*
    @brief Slice out the 1D Tensor that will be sorted on

    @param[out] tensor_sort The 1D Tensor to sort ([in] empty pointer)
    @param[in] axis_name_sort The name of the axis that the sort will be based on
    @param[in] label_index_sort The label index that the sort will be based on
    @param[in] axis_name_apply The name of the axis that the sort will be applied to
    @param[in] device
    */
    virtual void sliceTensorDataForSort(std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& tensor_sort, const std::string& axis_name_sort, const int& label_index_sort, const std::string& axis_name_apply, DeviceT& device) = 0;

    /*
    @brief Sort the axis indices view based on the values of the 1D tensor slice

    @param[in] tensor_sort The 1D Tensor to sort
    @param[in] axis_name_apply The name of the axis that the sort will be applied to
    @param[in] device
    */
    void sortTensorDataSlice(const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& tensor_sort, const std::string & axis_name_apply, const sortOrder::order& order_by, DeviceT& device);

    /*
    @brief Select and reduce the tensor data in place according to the current indices view

    @param[in] device
    */
    void selectTensorData(DeviceT& device);

    /*
    @brief Convert the 1D indices view into a TDim indices Tensor to use for downstream TensorData selection

    The conversion is done by the following algorithm:
      1. normalizing all indices (converting to either zero or one)
      2. broadcasting all indices to the size of the Tensor
      3. multiplying all indices together

    @param[out] indices_select Pointer to the indices Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSelectIndicesFromIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, DeviceT& device) = 0;

    /*
    @brief Select the Tensor data and return the selected/reduced data

    Overloads are provided for TDim or 1D reduced tensor data output

    TODO: add unit test coverage

    @param[out] tensor_select The selected/reduced tensor data ([in] empty pointer)
    @param[in] indices_select The broadcasted indices view to perform the selection on
    @param[in] device
    */
    virtual void getSelectTensorDataFromIndicesView(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, DeviceT& device) = 0;

    /*
    @brief Sort the Tensor Data based on the sort order defined by the indices view

    The tensor data and all of the axes are sorted according to the values of the indices view.
    The indices view are reset at the end of the operation
    */
    void sortTensorData(DeviceT& device);
    
    /*
    @brief Convert the 1D indices view into a TDim indices tensor that describes the sort order.
       The Tensor is ordering with respect to the first dimension (i.e., TDim = 0)

    The conversion is done by the following algorithm:
      1. set Dim = 0 as the reference axis
      2. compute Tensor indices i, j, k, ... as (index i - 1) + (index j - 1)*axis_i.size() + (index k - 1)*axis_i.size()*axis_j.size() ...
        where the - 1 is due to the indices starting at 1
        a. compute an adjusted axis index as (Index - 1) if Dim = 0 or as (Index - 1)*Prod[(Dim - 1).size() to Dim = 1]
        b. broadcast to the size of the tensor
        c. add all adjusted axis tensors together

    @param[out] indices_sort pointer to the indices sort Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSortIndicesViewFromIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_sort, DeviceT& device) = 0;

    /*
    @brief Update the tensor data with the given values and optionally return the original values

    @param[in] values_new The new tensor data
    @param[out] values_old The old tensor data ([in] memory should already have been allocated and synced)
    @param[in] device
    */
    template<typename T>
    void updateTensorDataConcept(const std::shared_ptr<T>& values_new, std::shared_ptr<T>& values_old, DeviceT& device);
    template<typename T>
    void updateTensorDataConcept(const std::shared_ptr<T>& values_new, DeviceT& device);
    void updateTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_old, DeviceT& device);
    void updateTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, DeviceT& device);
    void updateTensorDataValues(const std::shared_ptr<TensorT>& values_new, std::shared_ptr<TensorT>& values_old, DeviceT& device);
    void updateTensorDataValues(const std::shared_ptr<TensorT>& values_new, DeviceT& device);

    /*
    @brief Update the tensor data by a constant value and optionally return the original values

    @param[in] values_new The new tensor data
    @param[out] values_old The old tensor data in the form of a Sparse 2D TensorTable ([in] memory should already have been allocated and synced)
    @param[in] device
    */
    void updateTensorDataConstant(const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT& device);

    /*
    @brief Copy the selected values into a "Sparse" Tensor Table representation

    @param[out] sparse_table The "Sparse" Tensor table representation ([in] empty pointer)
      where
      - The Tensor Dimensions are the names of the axes dimensions
      - The axis for the tensor table are of nDimensions = TDim and nLabels = # of selected items where
        Dimensions are integers from 0 to TDim and the labels are the indices of the selected data
      - The Data for the Tensor Table is a 1D TensorData of length = # of selected items
    @param[in] device
    */
    void getSelectTensorDataAsSparseTensorTable(std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& sparse_table, DeviceT& device);

    /*
    @brief Convert the 1D indices view into a "Sparse" 2D tensor axis labels representation

    The conversion is done by the following algorithm:
      1. determine the size of each indices
      2. allocate memory for the linearized indices
      3. select out the non-zero elements
      4. iterate through each index in order and assign the values

    @param[out] sparse_select Pointer to the selected tensor data ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSparseAxisLabelsFromIndicesView(std::shared_ptr<TensorData<int, DeviceT, 2>>& sparse_select, DeviceT& device) = 0;

    /*
    @brief Create a Sparse 2D table representation

    @param[in] sparse_dimensions The dimension names of the axis
    @param[in] sparse_labels The 2D label names of the axis
    @param[in] sparse_data The data to initialize the sparse table with
    @param[out] sparse_table The Sparse 2D Tensor Table ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSparseTensorTable(const Eigen::Tensor<std::string, 1>& sparse_dimensions, const std::shared_ptr<TensorData<int, DeviceT, 2>>& sparse_labels, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& sparse_data, std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& sparse_table, DeviceT& device) = 0;

    /*
    @brief Update the TensorTable data according to the values in the "Sparse" Tensor Table representation

    @param[in] axes The axis for the tensor table of nDimensions (Dim 0) = TDim and nLabels (Dim 1) = # of selected items where
      Dimensions are integers from 0 to TDim and the labels are the indices of the selected data
    @param[in] data The 1D TensorData of length = # of selected items
    @param[in] device
    */
    void updateTensorTableFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorT, DeviceT, 1>>& sparse_table, DeviceT& device);

    /*
    @brief Append new labels to the specified axis and append new data to the Tensor Data at the specified axis

    NOTE: that the existing indexes of each axis are not changed to allow for undo operations

    @param[in] axis_name The axis to append the labels and data to
    @param[in] labels The labels to append to the axis
    @param[in] values The values to append to the tensor data along the specified axis
    @param[out] indices A 1D Tensor of indices that were added ([in] empty pointer)
    @param[in] device
    */
    template<typename LabelsT, typename T>
    void appendToAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    template<typename LabelsT>
    void appendToAxis(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    
    /*
    @brief Make the extended indices that will be appended to the existing indices

    @param[in] axis_name The axis to append the labels and data to
    @param[in] n_labels The number of labels to extend the indices by
    @param[out] indices A 1D Tensor of indices that were added ([in] empty pointer)
    @param[in] device
    */
    virtual void makeAppendIndices(const std::string& axis_name, const int& n_labels, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;

    /*
    @brief Expand the size of the axis by concatenating the new indices

    @param[in] indices The indices to add
    */
    void appendToIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device);

    /*
    @brief Delete a specified number of labels and sized slice of the TensorData at a specified position
      with an optional return of the deleted labels and tensordata

    NOTE: that the existing indexes of each axis are not changed to allow for undo operations

    @param[in] axis_name The axis to append the labels and data to
    @param[in] indices A 1D Tensor of indices to delete
    @param[out] labels The labels that were deleted ([in] empty pointer)
    @param[out] values The values that were deleted ([in] memory should already have been allocated and synced)
    @param[in] device
    */
    template<typename LabelsT, typename T>
    void deleteFromAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T>& values, DeviceT& device);
    template<typename LabelsT>
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT>& values, DeviceT& device);
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);

    /*
    @brief Convert a 1D indices tensor to a TDim indices Tensor to use for downstream TensorData selection

    The conversion is done by the following algorithm:
      1. normalizing the indices (converting to either zero or one)
      2. broadcasting the indices to the size of the Tensor

    @param[in] axis_name The name of the axis that the indices correspond to
    @param[in] indices The `indices_view` -like Tensor to broadcast for selection
    @param[out] indices_select Pointer to the indices Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSelectIndicesFromIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, DeviceT& device) = 0;

    /*
    @brief Select the Tensor data and return the selected/reduced data

    Overloads are provided for TDim or 1D reduced tensor data output

    @param[out] tensor_select The selected/reduced tensor data ([in] empty pointer)
    @param[in] indices_select The broadcasted indices view to perform the selection on
    @param[in] device
    */
    virtual void getSelectTensorDataFromIndices(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const Eigen::array<Eigen::Index, TDim>& dimensions_select, DeviceT& device) = 0;

    /*
    @brief Convert a 1D tensor of indices to a selection indices the size of the indices view

    The conversion is done by the following algorithm:
      1. broadcast the `indices_view` to a 2D tensor of indices_view_size x indices_size
      2. broadcast the `indices` to a 2D tensor of indices_view_size x indics_size
      3. select and return 0 or 1 for a match
      4. reduce to a 1D tensor of size indices_view_size

    @param[in] axis_name The name of the axis
    @param[out] indices_select Pointer to the indices Tensor
    @param[in] indices Pointer to the indices Tensor
    @param[in] invert_selection Boolean option to invert the selection (i.e., select indices that do not match)
    @param[in] device
    */
    void makeIndicesViewSelectFromIndices(const std::string & axis_name, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices_select, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, const bool& invert_selection, DeviceT& device);

    /*
    @brief Convert the indices view to a reduced 1D tensor of non zero indices

    The conversion is done by the following algorithm:
      1. normalize the `indices_view` to zero and 1
      2. determine the size of the indices and allocate to memory
      3. select only the non-zero indices

    @param[in] axis_name The name of the axis
    @param[out] indices Pointer to the indices Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;

    /*
    @brief Shrink the size of the axis by removing the selected indices

    @param[in] indices
    */
    void deleteFromIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device);

    /*
    @brief Insert new labels to the specified axis and insert new data to the Tensor Data at the specified axis

    The insert operation is done using the following algorithm:
      1. Append the new labels to the axis and append the new data to the Tensor Data
      2. Swap the append indices for the user provided indices
      3. Apply the `indices_view` sort to the Tensor Data and the Axis labels
      4. Reset the `indices_view`

    @param[in] axis_name The axis to append the labels and data to
    @param[in] labels The labels to insert into the axis
    @param[in] values The values to insert into the tensor data along the specified axis
    @param[in] indices A 1D Tensor of indices to specify where the labels and data should be inserted
    @param[in] device
    */
    template<typename LabelsT, typename T>
    void insertIntoAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    template<typename LabelsT>
    void insertIntoAxis(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    
  protected:
    int id_ = -1;
    std::string name_ = "";

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>();
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> axes_; ///< primary axis is dim=0

    // TODO: combine into a single member called `indices` of Tensor dimensions 5 x indices_length
    //       in order to improve performance of all TensorInsert and TensorDelete methods
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices_; ///< starting at 1
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices_view_; ///< sorted and/or selected indices
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> is_modified_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> in_memory_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> is_shardable_;

    std::map<std::string, int> axes_to_dims_;
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> data_; ///< The actual tensor data
    
  private:
  	friend class cereal::access;
  	template<class Archive>
  	void serialize(Archive& archive) {
  		archive(id_, name_, dimensions_, axes_, indices_, indices_view_, is_modified_, in_memory_, is_shardable_,
        axes_to_dims_, data_);
  	}
  };

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::addTensorAxis(const std::shared_ptr<T>& tensor_axis)
  {
    auto found = axes_.emplace(tensor_axis->getName(), std::shared_ptr<TensorAxisConcept<DeviceT>>(new TensorAxisWrapper<T, DeviceT>(tensor_axis)));
  }

  template<typename TensorT, typename DeviceT, int TDim>
  void TensorTable<TensorT, DeviceT, TDim>::clear() {
    axes_.clear();
    dimensions_ = Eigen::array<Eigen::Index, TDim>();
    indices_.clear();
    indices_view_.clear();
    is_modified_.clear();
    in_memory_.clear();
    is_shardable_.clear();
    data_.reset();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncIndicesHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : indices_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setIndicesDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : indices_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getIndicesDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : indices_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncIndicesViewHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : indices_view_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setIndicesViewDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : indices_view_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getIndicesViewDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : indices_view_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncIsModifiedHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : is_modified_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setIsModifiedDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : is_modified_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getIsModifiedDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : is_modified_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncInMemoryHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : in_memory_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setInMemoryDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : in_memory_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getInMemoryDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : in_memory_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncIsShardableHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : is_shardable_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setIsShardableDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : is_shardable_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getIsShardableDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : is_shardable_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncAxesHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& axis_map : axes_) {
      bool synced_tmp = axis_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setAxesDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& axis_map : axes_) {
      axis_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getAxesDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& axis_map : axes_) {
      statuses.emplace(axis_map.first, axis_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::selectIndicesView(const std::string & axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, DeviceT & device)
  {
    // reshape to match the axis labels shape and broadcast the length of the labels
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_names_selected_reshape(select_labels->getDataPointer().get(), 1, (int)select_labels->getData().size());
 
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<int, 2>({ (int)axes_.at(axis_name)->getNLabels(), 1 }));
    // broadcast the axis labels the size of the labels queried
    std::shared_ptr<LabelsT> labels_data;
    axes_.at(axis_name)->getLabelsDataPointer(labels_data);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(labels_data.get(), (int)axes_.at(axis_name)->getNDimensions(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = (labels_reshape.chip(dimension_index, 0)).broadcast(Eigen::array<int, 2>({ 1, (int)select_labels->getData().size() }));

    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, (int)select_labels->getData().size() }));

    // select the indices and reduce back to a 1D Tensor
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
    auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 })).clip(0, 1);

    // update the indices view based on the selection
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels());
    indices_view.device(device) = indices_view * selected_sum;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortIndicesView(const std::string & axis_name, const int & dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT & device)
  {
    // create a copy of the indices view
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
    assert(indices_view_copy->syncHAndDData(device));

    // select the `labels` indices from the axis labels and store in the current indices view
    selectIndicesView(axis_name, dimension_index, select_labels, device);

    // sort the indices view
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    auto indices_view_selected = (indices_view_values != indices_view_values.constant(0)).select(indices_view_values, indices_view_values.constant(1e9));
    indices_view_values.device(device) = indices_view_selected;
    indices_view_.at(axis_name)->sort("ASC", device);
    indices_view_.at(axis_name)->syncHAndDData(device);

    // extract out the label
    int label_index = getFirstIndexFromIndicesView(axis_name, device) - 1;
    indices_view_.at(axis_name)->syncHAndDData(device);

    // revert back to the origin indices view
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_copy_values(indices_view_copy->getDataPointer().get(), indices_view_copy->getDimensions());
    indices_view_values.device(device) = indices_view_copy_values;

    // iterate through each axis and apply the sort
    for (const auto& axis_to_name : axes_to_dims_) {
      if (axis_to_name.first == axis_name) continue;

      // Slice out the tensor that will be used for sorting
      std::shared_ptr<TensorData<TensorT, DeviceT, 1>> tensor_sort;
      sliceTensorDataForSort(tensor_sort, axis_name, label_index, axis_to_name.first, device);
      
      // Sort the axis index view
      sortTensorDataSlice(tensor_sort, axis_to_name.first, order_by, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT, typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::whereIndicesViewConcept(const std::string & axis_name, const int & dimension_index, 
    const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, const std::shared_ptr<TensorData<T, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor & comparitor, const logicalModifiers::logicalModifier & modifier, const logicalContinuators::logicalContinuator & within_continuator, const logicalContinuators::logicalContinuator & prepend_continuator, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      const auto values_cast = std::reinterpret_pointer_cast<TensorData<TensorT, DeviceT, 1>>(values); // required for compilation: no conversion should be done
      whereIndicesView(axis_name, dimension_index, select_labels, values_cast, comparitor, modifier, within_continuator, prepend_continuator, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, 
    const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
    const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
    // create a copy of the indices view
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
    assert(indices_view_copy->syncHAndDData(device));

    // select the `labels` indices from the axis labels and store in the current indices view
    selectIndicesView(axis_name, dimension_index, select_labels, device);

    // Reduce the Tensor to `n_labels` using the `labels` indices as the selection criteria
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_view_bcast;
    broadcastSelectIndicesView(indices_view_bcast, axis_name, device);
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor_select;
    reduceTensorDataToSelectIndices(indices_view_bcast, tensor_select, axis_name, select_labels->getData().size(), device);

    // Determine the indices that pass the selection criteria
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select;
    selectTensorIndicesOnReducedTensorData(indices_select, values, tensor_select, axis_name, select_labels->getData().size(), comparitor, modifier, device);

    // revert back to the origin indices view
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indicies_view_values(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indicies_view_copy_values(indices_view_copy->getDataPointer().get(), indices_view_copy->getDimensions());
    indicies_view_values.device(device) = indicies_view_copy_values;
    
    // update all other tensor indices view based on the selection criteria tensor
    for (const auto& axis_to_name: axes_to_dims_) {
      if (axis_to_name.first == axis_name) continue;
      applyIndicesSelectToIndicesView(indices_select, axis_name, axis_to_name.first, within_continuator, prepend_continuator, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::resetIndicesView(const std::string& axis_name, DeviceT& device)
  {
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices(indices_.at(axis_name)->getDataPointer().get(), indices_.at(axis_name)->getDimensions());
    indices_view.device(device) = indices;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::zeroIndicesView(const std::string & axis_name, DeviceT& device)
  {
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    indices_view.device(device) = indices_view.constant(0);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices_select, const std::string & axis_name_select, const std::string & axis_name, const logicalContinuators::logicalContinuator & within_continuator, const logicalContinuators::logicalContinuator & prepend_continuator, DeviceT & device)
  {
    // apply the continuator reduction, then...
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());
    if (within_continuator == logicalContinuators::logicalContinuator::OR) {

      // build the continuator reduction indices for the OR within continuator
      Eigen::array<int, 1> reduction_dims;
      int index = 0;
      for (const auto& axis_to_name_red : this->axes_to_dims_) {
        if (axis_to_name_red.first != axis_name) {
          reduction_dims.at(index) = axis_to_name_red.second;
          ++index;
        }
      }

      // apply the OR continuator reduction
      auto indices_view_update = indices_select_values.sum(reduction_dims).clip(0, 1);  //ensure a max value of 1

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
    else if (within_continuator == logicalContinuators::logicalContinuator::AND) {
      // apply the AND continuator reduction along the axis_name_selection dim
      Eigen::array<Eigen::Index, 1> reduction_dims = { this->axes_to_dims_.at(axis_name_select) };
      auto indices_view_update_prod = indices_select_values.prod(reduction_dims);

      // apply a normalized sum (OR) continuator across all other dimensions
      auto indices_view_update = indices_view_update_prod;

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, 2>>& indices_select, const std::string & axis_name_select, const std::string & axis_name, const logicalContinuators::logicalContinuator & within_continuator, const logicalContinuators::logicalContinuator & prepend_continuator, DeviceT & device)
  {
    // apply the continuator reduction, then...
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());
    if (within_continuator == logicalContinuators::logicalContinuator::OR) {

      // build the continuator reduction indices for the OR within continuator
      Eigen::array<int, 1> reduction_dims;
      int index = 0;
      for (const auto& axis_to_name_red : this->axes_to_dims_) {
        if (axis_to_name_red.first != axis_name) {
          reduction_dims.at(index) = axis_to_name_red.second;
          ++index;
        }
      }

      // apply the OR continuator reduction
      auto indices_view_update = indices_select_values.sum(reduction_dims).clip(0, 1);  //ensure a max value of 1

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
    else if (within_continuator == logicalContinuators::logicalContinuator::AND) {
      // apply the AND continuator reduction along the axis_name_selection dim
      Eigen::array<Eigen::Index, 1> reduction_dims = { this->axes_to_dims_.at(axis_name_select) };
      auto indices_view_update_prod = indices_select_values.prod(reduction_dims);

      // apply a normalized sum (OR) continuator across all other dimensions
      auto indices_view_update = indices_view_update_prod;

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<int TDim_, typename>
  inline void TensorTable<TensorT, DeviceT, TDim>::applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const std::string & axis_name_select, const std::string & axis_name, const logicalContinuators::logicalContinuator & within_continuator, const logicalContinuators::logicalContinuator & prepend_continuator, DeviceT & device)
  {
    // apply the continuator reduction, then...
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());
    if (within_continuator == logicalContinuators::logicalContinuator::OR) {

      // build the continuator reduction indices for the OR within continuator
      Eigen::array<int, TDim - 1> reduction_dims;
      int index = 0;
      for (const auto& axis_to_name_red : this->axes_to_dims_) {
        if (axis_to_name_red.first != axis_name) {
          reduction_dims.at(index) = axis_to_name_red.second;
          ++index;
        }
      }

      // apply the OR continuator reduction
      auto indices_view_update = indices_select_values.sum(reduction_dims).clip(0, 1);  //ensure a max value of 1

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
    else if (within_continuator == logicalContinuators::logicalContinuator::AND) {
      // apply the AND continuator reduction along the axis_name_selection dim
      Eigen::array<Eigen::Index, 1> reduction_dims = { this->axes_to_dims_.at(axis_name_select) };
      auto indices_view_update_prod = indices_select_values.prod(reduction_dims);

      // apply a normalized sum (OR) continuator across all other dimensions
      if (TDim > 2) {
        Eigen::array<int, TDim - 2> reduction_dims_sum;
        int index = 0;
        for (const auto& axis_to_name_red : this->axes_to_dims_) {
          if (axis_to_name_red.first != axis_name && axis_to_name_red.first != axis_name_select) {
            if (this->axes_to_dims_.at(axis_name_select) <= this->axes_to_dims_.at(axis_to_name_red.first))
              reduction_dims_sum.at(index) = axis_to_name_red.second - 1; // prod dim was lost
            else
              reduction_dims_sum.at(index) = axis_to_name_red.second;
            ++index;
          }
        }
        auto indices_view_update = indices_view_update_prod.sum(reduction_dims_sum).clip(0,1); // ensure a max value of 1

        // update the indices view based on the prepend_continuator
        if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
          indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
        }
        else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
          indices_view.device(device) = indices_view * indices_view_update;
        }
      }

      // no other dims to worry about, use as is.
      else {
        auto indices_view_update = indices_view_update_prod;

        // update the indices view based on the prepend_continuator
        if (prepend_continuator == logicalContinuators::logicalContinuator::OR) {
          indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
        }
        else if (prepend_continuator == logicalContinuators::logicalContinuator::AND) {
          indices_view.device(device) = indices_view * indices_view_update;
        }
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortTensorDataSlice(const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& tensor_sort, const std::string & axis_name_apply, const sortOrder::order & order_by, DeviceT & device)
  {
    // sort the slice
    if (order_by == sortOrder::order::ASC) {
      tensor_sort->sortIndices(this->indices_view_.at(axis_name_apply), "ASC", device);
    }
    else if (order_by == sortOrder::order::DESC) {
      tensor_sort->sortIndices(this->indices_view_.at(axis_name_apply), "DESC", device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::selectTensorData(DeviceT & device)
  {
    // make the selection indices from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select;
    makeSelectIndicesFromIndicesView(indices_select, device);

    // select the tensor data based on the selection indices and update
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor_select;
    getSelectTensorDataFromIndicesView(tensor_select, indices_select, device);

    // resize each axis based on the indices view
    for (const auto& axis_to_name : axes_to_dims_) {
      axes_.at(axis_to_name.first)->deleteFromAxis(indices_view_.at(axis_to_name.first), device);
    }

    // remake the axes and move over the tensor data
    setAxes();
    data_ = tensor_select;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortTensorData(DeviceT & device)
  {
    // make the sort index tensor from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_sort;
    makeSortIndicesViewFromIndicesView(indices_sort, device);

    // apply the sort indices to the tensor data
    data_->sort(indices_sort, device);

    //sort each of the axis labels then reset the indices view
    for (const auto& axis_to_index: axes_to_dims_) {
      axes_.at(axis_to_index.first)->sortLabels(indices_view_.at(axis_to_index.first), device);
      resetIndicesView(axis_to_index.first, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataConcept(const std::shared_ptr<T>& values_new, std::shared_ptr<T>& values_old, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_new_copy = std::reinterpret_pointer_cast<TensorT>(values_new);
      auto values_old_copy = std::reinterpret_pointer_cast<TensorT>(values_old);
      updateTensorDataValues(values_new_copy, values_old_copy, device);
    }
  }
  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataConcept(const std::shared_ptr<T>& values_new, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_new_copy = std::reinterpret_pointer_cast<TensorT>(values_new);
      updateTensorDataValues(values_new_copy, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_old, DeviceT & device)
  {
    assert(values_new->getDimensions() == data_->getDimensions());

    // copy the old values
    values_old = data_->copy(device);

    // assign the new values
    updateTensorDataValues(values_new, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, DeviceT & device)
  {
    assert(values_new->getDimensions() == data_->getDimensions());

    // assign the new values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values_new->getDataPointer().get(), values_new->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    data_values.device(device) = values_new_values;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValues(const std::shared_ptr<TensorT>& values_new, std::shared_ptr<TensorT>& values_old, DeviceT & device)
  {
    // copy the old values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_old_values(values_old.get(), data_->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    values_old_values.device(device) = data_values;

    // assign the new values
    updateTensorDataValues(values_new, device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValues(const std::shared_ptr<TensorT>& values_new, DeviceT & device)
  {
    // assign the new values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values_new.get(), data_->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    data_values.device(device) = values_new_values;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataConstant(const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT & device)
  {
    // copy the old values
    getSelectTensorDataAsSparseTensorTable(values_old, device);

    // create the reshape dimensions
    Eigen::array<Eigen::Index, TDim> reshape_dimensions;
    for (int i = 0; i < TDim; ++i) {
      reshape_dimensions.at(i) = 1;
    }

    // assign the new values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values_new->getDataPointer().get(), reshape_dimensions);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    data_values.device(device) = values_new_values.broadcast(data_->getDimensions());
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::getSelectTensorDataAsSparseTensorTable(std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& sparse_table, DeviceT & device)
  {
    // make the selection indices from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select;
    makeSelectIndicesFromIndicesView(indices_select, device);

    // select the tensor data based on the selection indices
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> sparse_data;
    getSelectTensorDataFromIndicesView(sparse_data, indices_select, device);

    // reshape to a 1D representation (Column-wise)
    // [Is this even needed???]
    Eigen::array<Eigen::Index, TDim> new_dimensions;
    for (int i = 0; i < TDim; ++i) {
      new_dimensions.at(i) = 1;
      new_dimensions.at(0) *= sparse_data->getDimensions().at(i);
    }
    sparse_data->setDimensions(new_dimensions);

    // create a linear representation of the selected indices view (Column-wise)
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> sparse_labels;
    makeSparseAxisLabelsFromIndicesView(sparse_labels, device);

    // create the axis dimensions
    Eigen::Tensor<std::string, 1> sparse_dimensions(this->axes_to_dims_.size());
    for (const auto& axis_to_name : this->axes_to_dims_) {
      sparse_dimensions(axis_to_name.second) = axis_to_name.first;
    }

    // create the "Sparse" TensorAxes
    makeSparseTensorTable(sparse_dimensions, sparse_labels, sparse_data, sparse_table, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT, typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::appendToAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_copy = std::reinterpret_pointer_cast<TensorT>(values);
      appendToAxis(axis_name, labels, values_copy, indices, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // Append the new labels to the axis
    axes_.at(axis_name)->appendLabelsToAxis(labels, device);

    // Copy the current data
    auto data_copy = data_->copy(device);
    data_copy->syncHAndDData(device);

    // Make the extended axis indices and append to the index
    makeAppendIndices(axis_name, labels->getDimensions().at(1), indices, device);
    appendToIndices(axis_name, indices, device);

    // Resize and reset the current data
    Eigen::array<Eigen::Index, TDim> new_dimensions = data_->getDimensions();
    new_dimensions.at(axes_to_dims_.at(axis_name)) += labels->getDimensions().at(1);
    data_->setDimensions(new_dimensions);
    data_->setData();
    data_->syncHAndDData(device);

    // Determine the dimensions for the values
    Eigen::array<Eigen::Index, TDim> value_dimensions;
    for (const auto& axis_to_index: axes_to_dims_) {
      if (axis_to_index.first == axis_name)
        value_dimensions.at(axis_to_index.second) = labels->getDimensions().at(1);
      else
        value_dimensions.at(axis_to_index.second) = data_->getDimensions().at(axes_to_dims_.at(axis_to_index.first));
    }

    // Concatenate the new data with the existing tensor data along the axis dimension
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values.get(), value_dimensions);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy_values(data_copy->getDataPointer().get(), data_copy->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    data_values.device(device) = data_copy_values.concatenate(values_new_values, axes_to_dims_.at(axis_name));
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::appendToIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // copy the current indices
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_copy = indices_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> is_modified_copy = is_modified_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> in_memory_copy = in_memory_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> is_shardable_copy = is_shardable_.at(axis_name)->copy(device);
    indices_copy->syncHAndDData(device);
    indices_view_copy->syncHAndDData(device);
    is_modified_copy->syncHAndDData(device);
    in_memory_copy->syncHAndDData(device);
    is_shardable_copy->syncHAndDData(device);

    // resize and reset the indices
    Eigen::array<Eigen::Index, 1> new_dimensions = indices_.at(axis_name)->getDimensions();
    new_dimensions.at(0) += indices->getDimensions().at(0);
    indices_.at(axis_name)->setDimensions(new_dimensions); 
    indices_.at(axis_name)->setData();
    indices_.at(axis_name)->syncHAndDData(device);
    indices_view_.at(axis_name)->setDimensions(new_dimensions); 
    indices_view_.at(axis_name)->setData();
    indices_view_.at(axis_name)->syncHAndDData(device);
    is_modified_.at(axis_name)->setDimensions(new_dimensions); 
    is_modified_.at(axis_name)->setData();
    is_modified_.at(axis_name)->syncHAndDData(device);
    in_memory_.at(axis_name)->setDimensions(new_dimensions); 
    in_memory_.at(axis_name)->setData();
    in_memory_.at(axis_name)->syncHAndDData(device);
    is_shardable_.at(axis_name)->setDimensions(new_dimensions); 
    is_shardable_.at(axis_name)->setData();
    is_shardable_.at(axis_name)->syncHAndDData(device);

    // create a dummy single value tensor of 1 of the same length as the indices
    std::shared_ptr<TensorData<int, DeviceT, 1>> ones = indices->copy(device);
    ones->syncHAndDData(device);

    // concatenate the new indices
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_new_values(indices->getDataPointer().get(), (int)indices->getTensorSize(), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> ones_concatenate(ones->getDataPointer().get(), (int)ones->getTensorSize(), 1);
    ones_concatenate.device(device) = indices_new_values.constant(1);

    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_copy_values(indices_copy->getDataPointer().get(), (int)indices_copy->getTensorSize(), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_copy_values(indices_view_copy->getDataPointer().get(), (int)indices_view_copy->getTensorSize(), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> is_modified_copy_values(is_modified_copy->getDataPointer().get(), (int)is_modified_copy->getTensorSize(), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> in_memory_copy_values(in_memory_copy->getDataPointer().get(), (int)in_memory_copy->getTensorSize(), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> is_shardable_copy_values(is_shardable_copy->getDataPointer().get(), (int)is_shardable_copy->getTensorSize(), 1);

    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_values(indices_view_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> is_modified_values(is_modified_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> in_memory_values(in_memory_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0), 1);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> is_shardable_values(is_shardable_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0), 1);

    indices_values.device(device) = indices_copy_values.concatenate(indices_new_values, 0);
    indices_view_values.device(device) = indices_view_copy_values.concatenate(indices_new_values, 0);
    is_modified_values.device(device) = is_modified_copy_values.concatenate(ones_concatenate, 0);
    in_memory_values.device(device) = in_memory_copy_values.concatenate(ones_concatenate, 0);
    is_shardable_values.device(device) = is_shardable_copy_values.concatenate(ones_concatenate, 0);

    // update the dimensions
    dimensions_.at(axes_to_dims_.at(axis_name)) += indices->getTensorSize();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT, typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromAxisConcept(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T>& values, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_copy = std::reinterpret_pointer_cast<TensorT>(values);
      deleteFromAxis(axis_name, indices, labels, values_copy, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT>& values, DeviceT & device)
  {
    // Make the selection indices for copying from the labels
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_select_labels_copy;
    makeIndicesViewSelectFromIndices(axis_name, indices_select_labels_copy, indices, false, device);

    // Copy the labels prior to deleting
    axes_.at(axis_name)->selectFromAxis(indices_select_labels_copy, labels, device);

    // Make the selection indices for copying the tensor data
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select_data_copy;
    makeSelectIndicesFromIndices(axis_name, indices_select_labels_copy, indices_select_data_copy, device);

    // Copy the data prior to deleting 
    // TODO: could be made more efficient to avoid the memory allocation in `reduceTensorDataToSelectIndices`
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> values_copy_ptr;
    reduceTensorDataToSelectIndices(indices_select_data_copy, values_copy_ptr, axis_name, indices->getTensorSize(), device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_copy(values_copy_ptr->getDataPointer().get(), values_copy_ptr->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_values(values.get(), values_copy_ptr->getDimensions());
    values_values.device(device) = values_copy;

    deleteFromAxis(axis_name, indices, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // Make the selection indices for deleting the labels
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_select_labels_delete;
    makeIndicesViewSelectFromIndices(axis_name, indices_select_labels_delete, indices, true, device);

    // Delete from the axis
    axes_.at(axis_name)->deleteFromAxis(indices_select_labels_delete, device);

    // Make the selection indices for deleting from the tensor data
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select_data_delete;
    makeSelectIndicesFromIndices(axis_name, indices_select_labels_delete, indices_select_data_delete, device);

    // select the tensor data based on the selection indices and update
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor_select;
    Eigen::array<Eigen::Index, TDim> dimensions_select = getDimensions();
    dimensions_select.at(axes_to_dims_.at(axis_name)) -= indices->getTensorSize();
    getSelectTensorDataFromIndices(tensor_select, indices_select_data_delete, dimensions_select, device);
    data_ = tensor_select;

    // reduce the indices
    deleteFromIndices(axis_name, indices, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::makeIndicesViewSelectFromIndices(const std::string & axis_name, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices_select, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, const bool& invert_selection, DeviceT & device)
  {
    // Reshape and broadcast the indices
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    auto indices_bcast = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)indices_view_.at(axis_name)->getTensorSize(), 1 }));

    // broadcast the indices view
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)indices_view_.at(axis_name)->getTensorSize(), 1);
    auto indices_view_bcast = indices_view.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)indices->getTensorSize() }));

    // assign the output data
    indices_select = indices_view_.at(axis_name)->copy(device);
    indices_select->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());

    // Select and reduce
    if (invert_selection) {
      auto indices_selected_2d = (indices_view_bcast == indices_bcast).select(indices_view_bcast.constant(0), indices_view_bcast);
      auto indices_selected = indices_selected_2d.prod(Eigen::array<int, 1>({ 1 })).clip(0, 1);
      indices_select_values.device(device) = indices_selected;
    }
    else {
      auto indices_selected_2d = (indices_view_bcast == indices_bcast).select(indices_view_bcast, indices_view_bcast.constant(0));
      auto indices_selected = indices_selected_2d.sum(Eigen::array<int, 1>({ 1 })).clip(0, 1);
      indices_select_values.device(device) = indices_selected;
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromIndices(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // determine the new indices sizes
    Eigen::array<Eigen::Index, 1> new_dimensions = indices_.at(axis_name)->getDimensions();
    new_dimensions.at(0) -= indices->getDimensions().at(0);

    // copy and resize the current indices
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_copy = indices_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> is_modified_copy = is_modified_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> in_memory_copy = in_memory_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> is_shardable_copy = is_shardable_.at(axis_name)->copy(device);
    indices_copy->setDimensions(new_dimensions); 
    indices_copy->setData();
    indices_view_copy->setDimensions(new_dimensions); 
    indices_view_copy->setData();
    is_modified_copy->setDimensions(new_dimensions); 
    is_modified_copy->setData();
    in_memory_copy->setDimensions(new_dimensions); 
    in_memory_copy->setData();
    is_shardable_copy->setDimensions(new_dimensions); 
    is_shardable_copy->setData();
    indices_copy->syncHAndDData(device);
    indices_view_copy->syncHAndDData(device);
    is_modified_copy->syncHAndDData(device);
    in_memory_copy->syncHAndDData(device);
    is_shardable_copy->syncHAndDData(device);

    // make the selection tensor based off of the selection indices
    std::shared_ptr<TensorData<int, DeviceT, 1>> selection_indices;
    makeIndicesViewSelectFromIndices(axis_name, selection_indices, indices, true, device);

    // select the values based on the indices
    indices_.at(axis_name)->select(indices_copy, selection_indices, device);
    indices_view_.at(axis_name)->select(indices_view_copy, selection_indices, device);
    is_modified_.at(axis_name)->select(is_modified_copy, selection_indices, device);
    in_memory_.at(axis_name)->select(in_memory_copy, selection_indices, device);
    is_shardable_.at(axis_name)->select(is_shardable_copy, selection_indices, device);

    // swap the indices
    indices_.at(axis_name) = indices_copy;
    indices_view_.at(axis_name) = indices_view_copy;
    is_modified_.at(axis_name) = is_modified_copy;
    in_memory_.at(axis_name) = in_memory_copy;
    is_shardable_.at(axis_name) = is_shardable_copy;

    // update the dimensions
    dimensions_.at(axes_to_dims_.at(axis_name)) -= indices->getTensorSize();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT, typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::insertIntoAxisConcept(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_copy = std::reinterpret_pointer_cast<TensorT>(values);
      insertIntoAxis(axis_name, labels, values_copy, indices, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // Append the new labels and values
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_append;
    appendToAxis(axis_name, labels, values, indices_append, device);

    // Swap the appended indices for the original indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_.at(axis_name)->getDataPointer().get(), indices_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_old_values(indices->getDataPointer().get(), indices->getDimensions());
    Eigen::array<int, 1> offsets = { (int)indices_view_.at(axis_name)->getTensorSize() - (int)indices->getTensorSize() };
    Eigen::array<int, 1> extents = { (int)indices->getTensorSize() };
    indices_view_values.slice(offsets, extents).device(device) = indices_old_values;
    indices_values.slice(offsets, extents).device(device) = indices_old_values;

    // TODO: Why is this needed on the GPU?
    // BUG: Indices and Data appear not to sync correctly
//#if COMPILE_WITH_CUDA
//    this->syncIndicesHAndDData(device);
//    this->syncHAndDData(device);
//    this->syncAxesHAndDData(device);
//    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
//    this->syncIndicesHAndDData(device);
//    this->syncHAndDData(device);
//    this->syncAxesHAndDData(device);
//#endif

    // Sort the indices
    indices_.at(axis_name)->sort("ASC", device); // NOTE: this could fail if there are 0's in the index!

    // Sort the axis and tensor based on the indices view
    sortTensorData(device);
  }
};
#endif //TENSORBASE_TENSORTABLE_H