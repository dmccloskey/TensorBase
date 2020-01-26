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
#include <TensorBase/io/DataFile.h>
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
  int MAX_INT = 1e9;

  /**
    @brief Class for managing Tensor data and associated Axes
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorTable
  {
  public:
    TensorTable() = default;  ///< Default constructor
    TensorTable(const std::string& name) : name_(name) {};
    TensorTable(const std::string& name, const std::string& dir) : name_(name), dir_(dir) {};
    virtual ~TensorTable() = default; ///< Default destruct

    template<typename TensorTOther, typename DeviceTOther, int TDimOther>
    inline bool operator==(const TensorTable<TensorTOther, DeviceTOther, TDimOther>& other) const
    {
      bool meta_equal = std::tie(id_, name_, dimensions_, axes_to_dims_) == std::tie(other.id_, other.name_, other.dimensions_, other.axes_to_dims_);
      auto compare_maps = [](auto lhs, auto rhs) {return *(lhs.second.get()) == *(rhs.second.get()); };
      bool axes_equal = std::equal(axes_.begin(), axes_.end(), other.axes_.begin(), compare_maps);
      bool indices_equal = std::equal(indices_.begin(), indices_.end(), other.indices_.begin(), compare_maps);
      bool indices_view_equal = std::equal(indices_view_.begin(), indices_view_.end(), other.indices_view_.begin(), compare_maps);
      bool shard_id_equal = std::equal(shard_id_.begin(), shard_id_.end(), other.shard_id_.begin(), compare_maps);
      bool not_in_memory_equal = std::equal(not_in_memory_.begin(), not_in_memory_.end(), other.not_in_memory_.begin(), compare_maps);
      bool is_modified_equal = std::equal(is_modified_.begin(), is_modified_.end(), other.is_modified_.begin(), compare_maps);
      return meta_equal && axes_equal && indices_equal && indices_view_equal && shard_id_equal
        && not_in_memory_equal && is_modified_equal;
    }

    inline bool operator!=(const TensorTable& other) const
    {
      return !(*this == other);
    }

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

    void setDir(const std::string& dir) { dir_ = dir; }; ///< dir setter
    std::string getDir() const { return dir_; }; ///< dir getter

    template<typename T>
    void addTensorAxis(const std::shared_ptr<T>& tensor_axis);  ///< Tensor axis adder

    /**
      @brief Tensor Axes setter

      The method sets the tensor axes and initializes the indices, indices_view, is_modified, not_in_memory,
      shard_id, and shard_indices attributes after all axes have been added

      @param[in] device
    */
    virtual void setAxes(DeviceT& device) = 0;

    /**
      @brief DeviceT specific initializer

      @param[in] new_dimensions An array specifying the new data dimensions
    */
    virtual void initData(const Eigen::array<Eigen::Index, TDim>& new_dimensions, DeviceT& device) = 0;
    virtual void initData(DeviceT& device) = 0;

    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>>& getAxes() { return axes_; }; ///< axes getter
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> getAxes() const { return axes_; }; ///< axes getter

    // TODO: combine into a single member called `indices` of Tensor dimensions 5 x indices_length
    //       in order to improve performance of all TensorInsert and TensorDelete methods
    // UPDATE: attempted to do so, but lost performance in having to extract out indices_view and other 
    //       as seperate TensorData objects for sort/select operations (see feat/TensorTableFile branch head)
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndices() { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndicesView() { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsModified() { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getNotInMemory() { return not_in_memory_; }; ///< not_in_memory getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getShardId() { return shard_id_; }; ///< shard_id getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getShardIndices() { return shard_indices_; }; ///< shard_indicies getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndices() const { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndicesView() const { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIsModified() const { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getNotInMemory() const { return not_in_memory_; }; ///< not_in_memory getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getShardId() const { return shard_id_; }; ///< shard_id getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getShardIndices() const { return shard_indices_; }; ///< shard_indicies getter

    void setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions) { dimensions_ = dimensions; }; ///< dimensions setter
    Eigen::array<Eigen::Index, TDim> getDimensions() const { return dimensions_; }  ///< dimensions getter
    int getDimFromAxisName(const std::string& axis_name) const { return axes_to_dims_.at(axis_name); }
		std::map<std::string, int> getAxesToDims() const { return axes_to_dims_; }  ///< axes_to_dims getter
    void clear(const bool& clear_shard_spans = true);  ///< clears the axes and all associated data

    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() { return data_->getData(); } ///< data_->getData() wrapper
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() const { return data_->getData(); } ///< data_->getData() wrapper
    Eigen::array<Eigen::Index, TDim> getDataDimensions() const { return data_->getDimensions(); } ///< data_->getDimensions() wrapper
    size_t getDataTensorBytes() const { return data_->getTensorBytes(); } ///< data_->getTensorBytes() wrapper
    size_t getDataTensorSize() const { return data_->getTensorSize(); } ///< data_->getTensorSize() wrapper
    std::shared_ptr<TensorT[]> getDataPointer() { return data_->getDataPointer(); } ///< data_->getDataPointer() wrapper

    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter (NOTE: must sync the `data` AND `not_in_memory`/`is_modified` attributes!)
    void setData(); ///< data setter (NOTE: must sync the `data` AND `not_in_memory`/`is_modified` attributes!)
    void setDataShards(const std::shared_ptr<TensorData<int, DeviceT, 1>>& not_in_memory_shard_ids, DeviceT& device); ///< data setter that allocates memory only for the specified shards
    void convertDataFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, DeviceT& device); ///< data setter (NOTE: must sync the `data` AND `not_in_memory`/`is_modified` attributes!)

    bool syncIndicesHAndDData(DeviceT& device); ///< Sync the host and device indices data
    void setIndicesDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices data
    std::map<std::string, std::pair<bool, bool>> getIndicesDataStatus(); ///< Get the status of the host and device indices data

    bool syncIndicesViewHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setIndicesViewDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getIndicesViewDataStatus(); ///< Get the status of the host and device indices view data

    bool syncIsModifiedHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setIsModifiedDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getIsModifiedDataStatus(); ///< Get the status of the host and device indices view data

    bool syncNotInMemoryHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setNotInMemoryDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getNotInMemoryDataStatus(); ///< Get the status of the host and device indices view data

    bool syncShardIdHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setShardIdDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getShardIdDataStatus(); ///< Get the status of the host and device indices view data

    bool syncShardIndicesHAndDData(DeviceT& device); ///< Sync the host and device indices view data
    void setShardIndicesDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device indices view data
    std::map<std::string, std::pair<bool, bool>> getShardIndicesDataStatus(); ///< Get the status of the host and device indices view data

    bool syncAxesHAndDData(DeviceT& device); ///< Sync the host and device axes data
    void setAxesDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device axes data
    std::map<std::string, std::pair<bool, bool>> getAxesDataStatus(); ///< Get the status of the host and device axes data

    bool syncHAndDData(DeviceT& device) { return data_->syncHAndDData(device); };  ///< Sync the host and device tensor data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { data_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device tensor data
    std::pair<bool, bool> getDataStatus() { return data_->getDataStatus(); };   ///< Get the status of the host and device tensor data

    bool syncAxesAndIndicesDData(DeviceT& device); ///< Transfer all axes and indices data to the device (if not already)
    bool syncAxesAndIndicesHData(DeviceT& device); ///< Transfer all axes and indices data to the host (if not already)
    bool syncDData(DeviceT& device); ///< Transfer tensor data to the device (if not already)
    bool syncHData(DeviceT& device); ///< Transfer tensor data to the host (if not already)

    template<typename T>
    void getDataPointer(std::shared_ptr<T[]>& data_copy); ///< TensorTableConcept data getter

    void setShardSpans(const std::map<std::string, int>& shard_spans) { shard_spans_ = shard_spans; }; ///< shard_span setter
    std::map<std::string, int> getShardSpans() const { return shard_spans_; }; ///< shard_span getter

    /*
    @brief Reset the shard indices based on the current `shard_span`.
      Overloads are provided that perform the update on or off the device

    @param[in] device
    */
    void reShardIndices(DeviceT& device);
    void reShardIndices();

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
    @param[in] select_labels_data A single dimensions of the axis labels
    @param[in] device
    */
    template<typename LabelsT>
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, DeviceT& device);

    /*
    @brief Select Tensor Axis that will be included in the view

    The selection is done according to the following algorithm: [TODO: update!]
      1. The `select_labels` are reshaped and broadcasted to a 2D tensor of labels_size x select_labels_size
      2. The `labels` for the axis are broadcasted to a 2D tensor of labels_size x select_labels_size
      3. The `indices` for the axis are broadcasted to a 2D tensor of labels_size x select_labels_size
      4. The indices that correspond to matches between the `select_labels` and `labels` bcast tensors are selected
      5. The result is reduced and normalized to a 1D Tensor of 0 or 1 of size labels_size
      6. The `indices_view` is updated by multiplication by the 1D select Tensor

    @param[in] axis_name
    @param[in] select_labels_data The full 2D axis labels
    @param[in] device
    */
    template<typename LabelsT>
    void selectIndicesView(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, DeviceT& device);
    
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
    template<typename LabelsT>
    void sortIndicesView(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, const sortOrder::order& order_by, DeviceT& device);
    void sortIndicesView_(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices_view_copy, const sortOrder::order& order_by, DeviceT& device);

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

    Limitations: The algorithm works when selecting for true values.
      e.g., the need to switch `not_in_memory` to `not_in_memory` when selecting for not in memory values

    @param[in] indices_component A TensorTable component e.g., indices_, indices_view_, shard_id_, or shard_index_
    @param[out] indices_select Pointer to the indices Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSelectIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& indices_component, std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, DeviceT& device) const = 0;

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

    @param[in] indices_component A TensorTable component e.g., indices_, indices_view_, shard_id_, or shard_index_
    @param[out] indices_sort pointer to the indices sort Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeSortIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& indices_component, std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_sort, DeviceT& device) const = 0;

    /*
    @brief Update the tensor data with the given values and optionally return the original values. 
			The method assumes that the TensorTable Data, Indices, and Axes have already be selected and reduced.

    @param[in] values_new The new tensor data
    @param[out] values_old The old tensor data ([in] memory should already have been allocated and synced)
    @param[in] device
    */
    template<typename T>
    void updateSelectTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, std::shared_ptr<T[]>& values_old, DeviceT& device);
    template<typename T>
    void updateSelectTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, DeviceT& device);
    void updateSelectTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_old, DeviceT& device);
    void updateSelectTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, DeviceT& device);
    void updateSelectTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, std::shared_ptr<TensorT[]>& values_old, DeviceT& device);
    void updateSelectTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, DeviceT& device);

		/*
		@brief Update the tensor data with the given values and optionally return the original values.

		@param[in] values_new The new tensor data
		@param[out] values_old The old tensor data in the form of a Sparse 2D TensorTable ([in] empty pointer)
		@param[in] device
		*/
		template<typename T>
		void updateTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, std::shared_ptr<TensorTable<T, DeviceT, 2>>& values_old, DeviceT& device);
		void updateTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT& device);
		template<typename T>
		void updateTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, DeviceT& device);
		void updateTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, DeviceT& device);

    /*
    @brief Update the tensor data by a constant value and optionally return the original values

    @param[in] values_new The new tensor data
    @param[out] values_old The old tensor data in the form of a Sparse 2D TensorTable ([in] empty pointer)
    @param[in] device
    */
    template<typename T>
    void updateTensorDataConstantConcept(const std::shared_ptr<TensorData<T, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<T, DeviceT, 2>>& values_old, DeviceT& device);
    void updateTensorDataConstant(const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT& device);

    /*
    @brief Update the tensor data from a Sparse 2D TensorTable

    NOTE: The Indices view will be reset during the method

    @param[in] values_old The old tensor data in the form of a Sparse 2D TensorTable
    @param[in] device
    */
    template<typename T>
    void updateTensorDataFromSparseTensorTableConcept(const std::shared_ptr<TensorTable<T, DeviceT, 2>>& values_old, DeviceT& device);
    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT& device);

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
    @brief Append new labels to the specified axis and append new data to the Tensor Data at the specified axis

    NOTE: that the existing indexes of each axis are not changed to allow for undo operations

    @param[in] axis_name The axis to append the labels and data to
    @param[in] labels The labels to append to the axis
    @param[in] values The values to append to the tensor data along the specified axis
    @param[out] indices A 1D Tensor of indices that were added ([in] empty pointer)
    @param[in] device
    */
    template<typename LabelsT, typename T>
    void appendToAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T[]>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    template<typename LabelsT>
    void appendToAxis(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT[]>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    
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
    void deleteFromAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T[]>& values, DeviceT& device);
    template<typename LabelsT>
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT[]>& values, DeviceT& device);
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
    void insertIntoAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T[]>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    template<typename LabelsT>
    void insertIntoAxis(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT[]>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    
    /**
      @brief Load data from file

      @param[in] filename The name of the data file
      @param[in] device

      @returns Status True on success, False if not
    */
    virtual bool loadTensorTableBinary(const std::string& dir, DeviceT& device) = 0;

    /**
      @brief Create a unique name for each TensorTableShard

      @param[in] dir The directory name
      @param[in] tensor_table_name The tensor table name
      @param[in] shard_id The id of the tensor table shard

      @returns A string with the filename for the TensorTableShard
    */
    static std::string makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id);

    /**
    @brief Determine the Tensor data shards that have been modified from the
      `is_modified` and `shard_id` members, and make an ordered 1D Tensor with
      unique TensorData shard ids

    @param[out] modified_shard_id
    @param[in] device
    */
    void makeModifiedShardIDTensor(std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, DeviceT& device) const;

    /**
      @brief Write data to file

      The TensorData is transfered to the host, and the `is_modified` attribute is reset to all 0's

      @param[in] filename The name of the data file
      @param[in] device

      @returns Status True on success, False if not
    */
    virtual bool storeTensorTableBinary(const std::string& dir, DeviceT& device) = 0;

    /**
    @brief Make the shard_id 1D tensor with unique TensorData shard_id
      based on the unique and num_runs data generated from the calls to sort and runLengthEncode

    @param[out] modified_shard_id
    @param[in] unique
    @param[in] num_runs
    @param[in] device
    */
    virtual void makeShardIDTensor(std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, std::shared_ptr<TensorData<int, DeviceT, 1>>& unique, std::shared_ptr<TensorData<int, DeviceT, 1>>& num_runs, DeviceT & device) const = 0;

    /*
    @brief Convert the 1D shard ID into a TDim indices tensor that describes the shart IDs of each Tensor element

    The conversion is done by the following algorithm:
      1. set Dim = 0 as the reference axis
      2. compute Tensor shard IDs i, j, k, ... as (index i - 1) + (index j - 1)*SUM(axis i - 1) + (index k - 1)*SUM(axis i - 1)*SUM(axis j - 1) ...
        where the - 1 is due to the indices starting at 1
        a. compute an adjusted axis index as (Index - 1) if Dim = 0 or as (Index - 1)*Prod[(Dim - 1).size() to Dim = 1]
        b. broadcast to the size of the tensor
        c. add all adjusted axis tensors together

    @param[out] indices_sort pointer to the indices sort Tensor ([in] empty pointer)
    @param[in] device
    */
    virtual void makeShardIndicesFromShardIDs(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_shard, DeviceT& device) const = 0;

    /*
    @brief Apply run length encode algorithm to a TensorData object

    @param[in] data Pointer to the data to apply the algorithm to
    @param[out] unique pointer to the unique data ([in] empty pointer)
    @param[out] count pointer to the count data ([in] empty pointer)
    @param[out] n_runs pointer to the number of runs data ([in] empty pointer)
    @param[in] device
    */
    virtual void runLengthEncodeIndex(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& data, std::shared_ptr<TensorData<int, DeviceT, 1>>& unique, std::shared_ptr<TensorData<int, DeviceT, 1>>& count, std::shared_ptr<TensorData<int, DeviceT, 1>>& n_runs, DeviceT & device) const = 0;

    /**
    @brief Determine the slice indices to extract out the TensorData shards

    @param[in] modified_shard_id An ordered 1D tensor with unique TensorData shard ids
    @param[out] slice_indices A map of shard_id to slice indices
    @param[in] device
    */
    virtual void makeSliceIndicesFromShardIndices(const std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, std::map<int, std::pair<Eigen::array<Eigen::Index, TDim>, Eigen::array<Eigen::Index, TDim>>>& slice_indices, DeviceT& device) const = 0;

    /**
    @brief Determine the Tensor data shards that are not in memory and are selected from the
      `not_in_memory`, `indices_view` and `shard_id` members, and make an ordered 1D Tensor with
      unique TensorData shard ids

    @param[out] modified_shard_id
    @param[in] device
    */
    void makeNotInMemoryShardIDTensor(std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, DeviceT& device) const;

    /**
      @brief Write Axes labels to file

      TODO: add a check to store only modified axes labels

      @param[in] dir The name of the directory
      @param[in] device

      @returns Status True on success, False if not
    */
    bool storeTensorTableAxesBinary(const std::string& dir, DeviceT& device);

    /**
      @brief Load Axes labels from file

      TODO: add a check to load only modified axes labels

      @param[in] dir The name of the directory
      @param[in] device

      @returns Status True on success, False if not
    */
    bool loadTensorTableAxesBinary(const std::string& dir, DeviceT& device);

    /**
      @brief Get a string vector representation of the data at a specified
        row after reshaping the data to 2D

      NOTE: all operations are done on the CPU!
      TODO: add another template for a different DeviceT for use with multi-threading

      @param[in] row_num The row number to fetch

      @returns String vector of data
    */
    std::vector<std::string> getCsvDataRow(const int& row_num);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int> = 0>
    std::vector<std::string> getCsvDataRowAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span, const Eigen::array<Eigen::Index, 2>& reshape) const;
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, bool>::value, int> = 0>
    std::vector<std::string> getCsvDataRowAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span, const Eigen::array<Eigen::Index, 2>& reshape) const;
    template<typename T = TensorT, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
    std::vector<std::string> getCsvDataRowAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span, const Eigen::array<Eigen::Index, 2>& reshape) const;

    Eigen::array<Eigen::Index, 2> getCsvDataDimensions(); ///< get the dimensions of the .csv data where dim 0 = axis 0 and dim 1 = axis 1 * axis 2 * ...
    Eigen::array<Eigen::Index, 2> getCsvShardSpans(); ///< get the shard spans of the .csv data where dim 0 = axis 0 and dim 1 = axis 1 * axis 2 * ...

    /**
      @brief Get a string vector representation of the non-primary axis labels
        at a specified row after reshaping the data to 2D

      NOTE: all operations are done on the CPU!
      TODO: add another template for a different DeviceT for use with multi-threading

      @param[in] row_num The row number to fetch

      @returns A map of string vector of data corresonding to each axis
    */
    std::map<std::string, std::vector<std::string>> getCsvAxesLabelsRow(const int& row_num);

    /**
      @brief Insert new axis labels and tensor data from csv strings

      @param[in] labels_new A map of axis name and 2D tensor of axis labels formated as strings
      @param[in] data_new Tensor data formated as a 2D tensor of strings
    */
    void insertIntoTableFromCsv(const std::map<std::string, Eigen::Tensor<std::string, 2>>& labels_new, const Eigen::Tensor<std::string, 2>& data_new, DeviceT& device);
    void insertIntoTableFromCsv(const Eigen::Tensor<std::string, 2>& data_new, DeviceT& device);

    /**
      @brief Make a sparse tensor table from a 2D tensor of csv strings

      @param[out] sparse_table_ptr A Sparse tensor table
      @param[in] data_new Tensor data formated as a 2D tensor of strings
    */
    virtual void makeSparseTensorTableFromCsv(std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& sparse_table_ptr, const Eigen::Tensor<std::string, 2>& data_new, DeviceT& device) = 0;

    // NOTE: IO methods for TensorTable indices components may not be needed because the call to setAxes remakes all of the indices on the fly
    //virtual bool storeTensorTableIndicesBinary(const std::string& dir, DeviceT& device) = 0; ///< Write tensor indices to disk
    //virtual bool loadTensorTableIndicesBinary(const std::string& dir, DeviceT& device) = 0; ///< Read tensor indices from disk

  protected:
    int id_ = -1;
    std::string name_ = "";
    std::string dir_ = "";

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>();
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> axes_; ///< primary axis is dim=0

    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices_; ///< starting at 1
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices_view_; ///< sorted and/or selected indices
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> is_modified_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> not_in_memory_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> shard_id_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> shard_indices_;

    std::map<std::string, int> axes_to_dims_;
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> data_; ///< The actual tensor data

    std::map<std::string, int> shard_spans_; ///< the shard span in each dimension
    
  private:
  	friend class cereal::access;
  	template<class Archive>
  	void serialize(Archive& archive) {
  		archive(id_, name_, dimensions_, axes_, indices_, indices_view_, is_modified_, not_in_memory_, shard_id_, shard_indices_,
        axes_to_dims_, data_, shard_spans_);
  	}
  };

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::addTensorAxis(const std::shared_ptr<T>& tensor_axis)
  {
    auto found = axes_.emplace(tensor_axis->getName(), std::shared_ptr<TensorAxisConcept<DeviceT>>(new TensorAxisWrapper<T, DeviceT>(tensor_axis)));
  }

  template<typename TensorT, typename DeviceT, int TDim>
  void TensorTable<TensorT, DeviceT, TDim>::clear(const bool& clear_shard_spans) {
    axes_.clear();
    for (auto& axis_to_dim: axes_to_dims_) {
      dimensions_.at(axis_to_dim.second) = 0;
    }
    //dimensions_ = Eigen::array<Eigen::Index, TDim>();
    indices_.clear();
    indices_view_.clear();
    is_modified_.clear();
    not_in_memory_.clear();
    shard_id_.clear();
    shard_indices_.clear();
    data_.reset();
    if (clear_shard_spans) shard_spans_.clear();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setData(const Eigen::Tensor<TensorT, TDim>& data)
  {
    data_->setData(data);
    for (auto& in_memory_map : not_in_memory_) {
      in_memory_map.second->getData() = in_memory_map.second->getData().constant(0); // host
    }
    for (auto& is_modified_map : is_modified_) {
      is_modified_map.second->getData() = is_modified_map.second->getData().constant(1); // host
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setData()
  {
    data_->setData();
    for (auto& in_memory_map : not_in_memory_) {
      in_memory_map.second->getData() = in_memory_map.second->getData().constant(1); // host
    }
    for (auto& is_modified_map : is_modified_) {
      is_modified_map.second->getData() = is_modified_map.second->getData().constant(0); // host
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setDataShards(const std::shared_ptr<TensorData<int, DeviceT, 1>>& not_in_memory_shard_ids, DeviceT& device)
  {
    // determine the needed data dimensions
    Eigen::array<Eigen::Index, TDim> data_dimensions;
    for (const auto& axis_to_dim : axes_to_dims_) {
      data_dimensions.at(axis_to_dim.second) = not_in_memory_shard_ids->getTensorSize() * shard_spans_.at(axis_to_dim.first);
    }

    // allocate memory for the data
    initData(data_dimensions, device);
    setData();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::convertDataFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, DeviceT & device)
  {
    data_->convertFromStringToTensorT(data_new, device);
    for (auto& in_memory_map : not_in_memory_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(in_memory_map.second->getDataPointer().get(), (int)in_memory_map.second->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(0); // device
    }
    for (auto& is_modified_map : is_modified_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_map.second->getDataPointer().get(), (int)is_modified_map.second->getTensorSize());
      is_modified.device(device) = is_modified.constant(1); // device
    }
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
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncNotInMemoryHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : not_in_memory_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setNotInMemoryDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : not_in_memory_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getNotInMemoryDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : not_in_memory_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncShardIdHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : shard_id_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setShardIdDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : shard_id_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getShardIdDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : shard_id_) {
      statuses.emplace(index_map.first, index_map.second->getDataStatus());
    }
    return statuses;
  }


  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncShardIndicesHAndDData(DeviceT & device)
  {
    bool synced = true;
    for (auto& index_map : shard_indices_) {
      bool synced_tmp = index_map.second->syncHAndDData(device);
      if (!synced_tmp) synced = false;
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::setShardIndicesDataStatus(const bool & h_data_updated, const bool & d_data_updated)
  {
    for (auto& index_map : shard_indices_) {
      index_map.second->setDataStatus(h_data_updated, d_data_updated);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::pair<bool, bool>> TensorTable<TensorT, DeviceT, TDim>::getShardIndicesDataStatus()
  {
    std::map<std::string, std::pair<bool, bool>> statuses;
    for (auto& index_map : shard_indices_) {
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
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncAxesAndIndicesDData(DeviceT& device)
  {
    bool synced = true;
    // transfer axis data to the device from the host
    for (auto& axis_map : axes_) {
      std::pair<bool, bool> statuses = axis_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = axis_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    // transfer indices data to the device from the host
    for (auto& index_map : indices_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : indices_view_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : is_modified_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : not_in_memory_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : shard_id_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : shard_indices_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.second) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncAxesAndIndicesHData(DeviceT& device)
  {
    bool synced = true;
    // transfer axis data to the device from the host
    for (auto& axis_map : axes_) {
      std::pair<bool, bool> statuses = axis_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = axis_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    // transfer indices data to the device from the host
    for (auto& index_map : indices_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : indices_view_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : is_modified_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : not_in_memory_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : shard_id_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    for (auto& index_map : shard_indices_) {
      std::pair<bool, bool> statuses = index_map.second->getDataStatus();
      if (!statuses.first) {
        bool synced_tmp = index_map.second->syncHAndDData(device);
        if (!synced_tmp) synced = false;
      }
    }
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncDData(DeviceT& device)
  {
    bool synced = true;
    std::pair<bool, bool> statuses = getDataStatus();
    if (!statuses.second)
      synced = syncHAndDData(device);
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::syncHData(DeviceT& device)
  {
    bool synced = true;
    std::pair<bool, bool> statuses = getDataStatus();
    if (!statuses.first)
      synced = syncHAndDData(device);
    return synced;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::reShardIndices(DeviceT& device)
  {
    for (const auto& shard_span_map : shard_spans_) {
      // Determine the max shard id value
      int shard_id = 0;
      int max_shard_id = ceil(float(axes_.at(shard_span_map.first)->getNLabels()) / float(shard_span_map.second));
      for (; shard_id < max_shard_id; ++shard_id) {
        // Determine the offsets and span for slicing
        Eigen::array<Eigen::Index, 1> offset, span;
        offset.at(0) = shard_id * shard_span_map.second;
        int remaining_length = axes_.at(shard_span_map.first)->getNLabels() - shard_id * shard_span_map.second;
        span.at(0) = (shard_span_map.second <= remaining_length) ? shard_span_map.second: remaining_length;

        // Update the shard id
        Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_id_values(shard_id_.at(shard_span_map.first)->getDataPointer().get(), (int)shard_id_.at(shard_span_map.first)->getTensorSize());
        shard_id_values.slice(offset, span).device(device) = shard_id_values.slice(offset, span).constant(shard_id + 1);

        // Update the shard indices using the indices as a template
        Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_indices_values(shard_indices_.at(shard_span_map.first)->getDataPointer().get(), (int)shard_indices_.at(shard_span_map.first)->getTensorSize());
        Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_.at(shard_span_map.first)->getDataPointer().get(), (int)indices_.at(shard_span_map.first)->getTensorSize());
        shard_indices_values.slice(offset, span).device(device) = indices_values.slice(Eigen::array<Eigen::Index, 1>({0}), span);
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::reShardIndices()
  {
    for (const auto& shard_span_map : shard_spans_) {
      // Determine the max shard id value
      int shard_id = 0;
      int max_shard_id = ceil(float(axes_.at(shard_span_map.first)->getNLabels()) / float(shard_span_map.second));
      for (; shard_id < max_shard_id; ++shard_id) {
        // Determine the offsets and span for slicing
        Eigen::array<Eigen::Index, 1> offset, span;
        offset.at(0) = shard_id * shard_span_map.second;
        int remaining_length = axes_.at(shard_span_map.first)->getNLabels() - shard_id * shard_span_map.second;
        span.at(0) = (shard_span_map.second <= remaining_length) ? shard_span_map.second : remaining_length;

        // Update the shard id
        Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_id_values(shard_id_.at(shard_span_map.first)->getHDataPointer().get(), (int)shard_id_.at(shard_span_map.first)->getTensorSize());
        shard_id_values.slice(offset, span) = shard_id_values.slice(offset, span).constant(shard_id + 1);

        // Update the shard indices using the indices as a template
        Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_indices_values(shard_indices_.at(shard_span_map.first)->getHDataPointer().get(), (int)shard_indices_.at(shard_span_map.first)->getTensorSize());
        Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_.at(shard_span_map.first)->getHDataPointer().get(), (int)indices_.at(shard_span_map.first)->getTensorSize());
        shard_indices_values.slice(offset, span) = indices_values.slice(Eigen::array<Eigen::Index, 1>({ 0 }), span);
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::getDataPointer(std::shared_ptr<T[]>& data_copy)
  {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T[]>(data_->getDataPointer()); // required for compilation: no conversion should be done
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::selectIndicesView(const std::string & axis_name, const int& dimension_index, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& select_labels, DeviceT & device)
  {
    // reshape to match the axis labels shape and broadcast the length of the labels
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_names_selected_reshape(select_labels->getDataPointer().get(), 1, (int)select_labels->getData().size());
 
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<Eigen::Index, 2>({ (int)axes_.at(axis_name)->getNLabels(), 1 }));
    // broadcast the axis labels the size of the labels queried
    std::shared_ptr<LabelsT[]> labels_data;
    axes_.at(axis_name)->getLabelsDataPointer(labels_data);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(labels_data.get(), (int)axes_.at(axis_name)->getNDimensions(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = (labels_reshape.chip(dimension_index, 0)).broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)select_labels->getData().size() }));

    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)select_labels->getData().size() }));

    // select the indices and reduce back to a 1D Tensor
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
    auto selected_sum = selected.clip(0, 1).sum(Eigen::array<Eigen::Index, 1>({ 1 })).clip(0, 1);

    // update the indices view based on the selection
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels());
    indices_view.device(device) = indices_view * selected_sum;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::selectIndicesView(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, DeviceT & device)
  {
    assert(axes_.at(axis_name)->getNDimensions() == select_labels->getDimensions().at(0));
    // reshape to match the axis labels shape and broadcast the length of the labels
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_names_selected_reshape(select_labels->getDataPointer().get(), select_labels->getDimensions().at(0), 1, select_labels->getDimensions().at(1));
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<Eigen::Index, 3>({ 1, (int)axes_.at(axis_name)->getNLabels(), 1 }));
    
    // broadcast the axis labels the size of the labels queried
    std::shared_ptr<LabelsT[]> labels_data;
    axes_.at(axis_name)->getLabelsDataPointer(labels_data);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(labels_data.get(), (int)axes_.at(axis_name)->getNDimensions(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = labels_reshape.broadcast(Eigen::array<Eigen::Index, 3>({ 1, 1, select_labels->getDimensions().at(1) }));

    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 3>> indices_reshape(indices_.at(axis_name)->getDataPointer().get(), 1, (int)axes_.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<Eigen::Index, 3>({ (int)axes_.at(axis_name)->getNDimensions(), 1, select_labels->getDimensions().at(1) }));

    // select the indices and reduce back to a 1D Tensor
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0)).clip(0, 1).sum(
      Eigen::array<Eigen::Index, 1>({ 2 })).clip(0, 1).prod(Eigen::array<Eigen::Index, 1>({ 0 })).clip(0, 1); // matched = 1, not-matched = 0;

    // update the indices view based on the selection
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels());
    indices_view.device(device) = indices_view * selected;
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
    sortIndicesView_(axis_name, indices_view_copy, order_by, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortIndicesView(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels, const sortOrder::order& order_by, DeviceT& device)
  {
    // create a copy of the indices view
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
    assert(indices_view_copy->syncHAndDData(device));

    // select the `labels` indices from the axis labels and store in the current indices view
    selectIndicesView(axis_name, select_labels, device);

    // sort the indices view
    sortIndicesView_(axis_name, indices_view_copy, order_by, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortIndicesView_(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices_view_copy, const sortOrder::order& order_by, DeviceT& device)
  {
    // sort the indices view
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    auto indices_view_selected = (indices_view_values != indices_view_values.constant(0)).select(indices_view_values, indices_view_values.constant(MAX_INT));
    indices_view_values.device(device) = indices_view_selected;
    indices_view_.at(axis_name)->sort("ASC", device);
    indices_view_.at(axis_name)->syncHAndDData(device);

    // extract out the label
    int label_index = getFirstIndexFromIndicesView(axis_name, device) - 1;
    indices_view_.at(axis_name)->syncHAndDData(device);

    // revert back to the origin indices view
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_copy_values(indices_view_copy->getDataPointer().get(), indices_view_copy->getDimensions());
    indices_view_values.device(device) = indices_view_copy_values;

    // Check that the needed data is in memory
    loadTensorTableBinary(dir_, device);

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

    // check that the needed data is in memory
    loadTensorTableBinary(dir_, device);

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
      Eigen::array<Eigen::Index, 1> reduction_dims;
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
      Eigen::array<Eigen::Index, 1> reduction_dims;
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
      Eigen::array<Eigen::Index, TDim - 1> reduction_dims;
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
        Eigen::array<Eigen::Index, TDim - 2> reduction_dims_sum;
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
    makeSelectIndicesFromTensorIndicesComponent(this->indices_view_, indices_select, device);

    // select the tensor data based on the selection indices and update
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor_select;
    getSelectTensorDataFromIndicesView(tensor_select, indices_select, device);

    // resize each axis based on the indices view
    for (const auto& axis_to_name : axes_to_dims_) {
      axes_.at(axis_to_name.first)->deleteFromAxis(indices_view_.at(axis_to_name.first), device);
    }

    // remake the axes and move over the tensor data
    setAxes(device);
    data_ = tensor_select;

    // TODO: does it make sense to move this over to `setAxes()` ?
    // Sync all of the axes and reShard
    syncIndicesHAndDData(device);
    syncIndicesViewHAndDData(device);
    syncIsModifiedHAndDData(device);
    syncNotInMemoryHAndDData(device);
    syncShardIdHAndDData(device);
    syncShardIndicesHAndDData(device);
    reShardIndices(device);
    syncAxesHAndDData(device);

    // update the not_in_memory and is_modified attributes
    for (auto& in_memory_map : not_in_memory_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(in_memory_map.second->getDataPointer().get(), (int)in_memory_map.second->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(0); // device
    }
    for (auto& is_modified_map : is_modified_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_map.second->getDataPointer().get(), (int)is_modified_map.second->getTensorSize());
      is_modified.device(device) = is_modified.constant(1); // device
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortTensorData(DeviceT & device)
  {
    // make the sort index tensor from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_sort;
    makeSortIndicesFromTensorIndicesComponent(indices_view_, indices_sort, device);

    // check that the data is in memory and then apply the sort indices to the tensor data
    loadTensorTableBinary(dir_, device);
    data_->sort(indices_sort, device);

    //sort each of the axis labels then reset the indices view
    for (const auto& axis_to_index: axes_to_dims_) {
      axes_.at(axis_to_index.first)->sortLabels(indices_view_.at(axis_to_index.first), device);
      resetIndicesView(axis_to_index.first, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateSelectTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, std::shared_ptr<T[]>& values_old, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_new_copy = std::reinterpret_pointer_cast<TensorT[]>(values_new);
      auto values_old_copy = std::reinterpret_pointer_cast<TensorT[]>(values_old);
      updateSelectTensorDataValues(values_new_copy, values_old_copy, device);
    }
  }
  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateSelectTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_new_copy = std::reinterpret_pointer_cast<TensorT[]>(values_new);
      updateSelectTensorDataValues(values_new_copy, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateSelectTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_old, DeviceT & device)
  {
    assert(values_new->getDimensions() == data_->getDimensions()); // TODO: will need to be relocated

    // Check that the update values are in memory
    loadTensorTableBinary(dir_, device);

    // copy the old values
    values_old = data_->copy(device);

    // assign the new values
    updateSelectTensorDataValues(values_new, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateSelectTensorDataValues(const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& values_new, DeviceT & device)
  {
    assert(values_new->getDimensions() == data_->getDimensions()); // TODO: will need to be relocated

    // assign the new values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values_new->getDataPointer().get(), values_new->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    data_values.device(device) = values_new_values;

    // update the not_in_memory and is_modified attributes
    for (auto& in_memory_map : not_in_memory_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(in_memory_map.second->getDataPointer().get(), (int)in_memory_map.second->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(0); // device
    }
    for (auto& is_modified_map : is_modified_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_map.second->getDataPointer().get(), (int)is_modified_map.second->getTensorSize());
      is_modified.device(device) = is_modified.constant(1); // device
    }

  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateSelectTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, std::shared_ptr<TensorT[]>& values_old, DeviceT & device)
  {
    // Check that the update values are in memory
    loadTensorTableBinary(dir_, device);

    // copy the old values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_old_values(values_old.get(), data_->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    values_old_values.device(device) = data_values;

    // assign the new values
    updateSelectTensorDataValues(values_new, device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateSelectTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, DeviceT & device)
  {
    // assign the new values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values_new.get(), data_->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    data_values.device(device) = values_new_values;

    // update the not_in_memory and is_modified attributes
    for (auto& in_memory_map : not_in_memory_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(in_memory_map.second->getDataPointer().get(), (int)in_memory_map.second->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(0); // device
    }
    for (auto& is_modified_map : is_modified_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_map.second->getDataPointer().get(), (int)is_modified_map.second->getTensorSize());
      is_modified.device(device) = is_modified.constant(1); // device
    }
  }

	template<typename TensorT, typename DeviceT, int TDim>
	template<typename T>
	inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, std::shared_ptr<TensorTable<T, DeviceT, 2>>& values_old, DeviceT& device)
	{
		if (std::is_same<T, TensorT>::value) {
			auto values_new_copy = std::reinterpret_pointer_cast<TensorT[]>(values_new);
			std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> values_old_copy;
			updateTensorDataValues(values_new_copy, values_old_copy, device);
			values_old = std::reinterpret_pointer_cast<TensorTable<T, DeviceT, 2>>(values_old_copy);
		}
	}

	template<typename TensorT, typename DeviceT, int TDim>
	inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT& device)
	{
		// copy the old values
		getSelectTensorDataAsSparseTensorTable(values_old, device);

		// update the data
		updateTensorDataValues(values_new, device);
	}

	template<typename TensorT, typename DeviceT, int TDim>
	template<typename T>
	inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValuesConcept(const std::shared_ptr<T[]>& values_new, DeviceT& device)
	{
		if (std::is_same<T, TensorT>::value) {
			auto values_new_copy = std::reinterpret_pointer_cast<TensorT[]>(values_new);
			updateSelectTensorDataValues(values_new_copy, device);
		}
	}

	template<typename TensorT, typename DeviceT, int TDim>
	inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataValues(const std::shared_ptr<TensorT[]>& values_new, DeviceT& device)
	{
		// make the sparseTensorTable update
		// TODO: replace with new method `TensorTable::copy` (which does not yet exist...)
		std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> sparse_table_new;
		getSelectTensorDataAsSparseTensorTable(sparse_table_new, device);
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> values_new_values(values_new.get(), sparse_table_new->getDataTensorSize());
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_values(sparse_table_new->getDataPointer().get(), sparse_table_new->getDataTensorSize());
		data_values.device(device) = values_new_values;

		// update the tensorDataValues
		updateTensorDataFromSparseTensorTable(sparse_table_new, device);
	}

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataConstantConcept(const std::shared_ptr<TensorData<T, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<T, DeviceT, 2>>& values_old, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_new_copy = std::reinterpret_pointer_cast<TensorData<TensorT, DeviceT, 1>>(values_new);
      std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> values_old_copy;
      updateTensorDataConstant(values_new_copy, values_old_copy, device);
      values_old = std::reinterpret_pointer_cast<TensorTable<T, DeviceT, 2>>(values_old_copy);
    }
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

    // make the selection indices from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select;
    makeSelectIndicesFromTensorIndicesComponent(this->indices_view_, indices_select, device);

    // Check that the update values are in memory
    loadTensorTableBinary(dir_, device);

    // create the selection
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_new_values(values_new->getDataPointer().get(), reshape_dimensions);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
    auto selection = (indices_select_values > indices_select_values.constant(0)).select(values_new_values.broadcast(data_->getDimensions()), data_values);
      
    // assign the new values
    data_values.device(device) = selection;

    // update the not_in_memory and is_modified attributes
    for (const auto& axis_to_dim : axes_to_dims_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_to_dim.first)->getDataPointer().get(), (int)indices_view_.at(axis_to_dim.first)->getTensorSize());
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(not_in_memory_.at(axis_to_dim.first)->getDataPointer().get(), (int)not_in_memory_.at(axis_to_dim.first)->getTensorSize());
      auto in_memory_selection = (indices_view > indices_view.constant(0)).select(not_in_memory.constant(0), not_in_memory);
      not_in_memory.device(device) = in_memory_selection;
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_.at(axis_to_dim.first)->getDataPointer().get(), (int)is_modified_.at(axis_to_dim.first)->getTensorSize());
      auto is_modified_selection = (indices_view > indices_view.constant(0)).select(is_modified.constant(1), is_modified);
      is_modified.device(device) = is_modified_selection;
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataFromSparseTensorTableConcept(const std::shared_ptr<TensorTable<T, DeviceT, 2>>& values_old, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_old_copy = std::reinterpret_pointer_cast<TensorTable<TensorT, DeviceT, 2>>(values_old);
      updateTensorDataFromSparseTensorTable(values_old_copy, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& values_old, DeviceT & device)
  {
    // make the partition index tensor from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_partition;
    makeSelectIndicesFromTensorIndicesComponent(indices_view_, indices_partition, device);

    // Check that the update values are in memory
    loadTensorTableBinary(this->dir_, device);

    // partition the data in place
    data_->partition(indices_partition, device);

    // update the sorted tensor data "slice" with the old values
    Eigen::array<Eigen::Index, 1> offsets = {0};
    Eigen::array<Eigen::Index, 1> extents = { (int)values_old->getDataTensorSize() };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> values_old_values(values_old->getDataPointer().get(), (int)values_old->getDataTensorSize());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_values(data_->getDataPointer().get(), (int)data_->getTensorSize());
    data_values.slice(offsets, extents).device(device) = values_old_values;

    // update the not_in_memory and is_modified attributes
    for (const auto& axis_to_dim : axes_to_dims_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_to_dim.first)->getDataPointer().get(), (int)indices_view_.at(axis_to_dim.first)->getTensorSize());
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(not_in_memory_.at(axis_to_dim.first)->getDataPointer().get(), (int)not_in_memory_.at(axis_to_dim.first)->getTensorSize());
      auto in_memory_selection = (indices_view > indices_view.constant(0)).select(not_in_memory.constant(0), not_in_memory);
      not_in_memory.device(device) = in_memory_selection;
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_.at(axis_to_dim.first)->getDataPointer().get(), (int)is_modified_.at(axis_to_dim.first)->getTensorSize());
      auto is_modified_selection = (indices_view > indices_view.constant(0)).select(is_modified.constant(1), is_modified);
      is_modified.device(device) = is_modified_selection;
    }

    // make the sort index tensor
    for (const auto& axis_to_index: axes_to_dims_)
      resetIndicesView(axis_to_index.first, device);
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_sort;
    makeSortIndicesFromTensorIndicesComponent(indices_view_, indices_sort, device);

    // partition the indices
    indices_sort->partition(indices_partition, device);

    // re-sort the data back to the original order
    data_->sort(indices_sort, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::getSelectTensorDataAsSparseTensorTable(std::shared_ptr<TensorTable<TensorT, DeviceT, 2>>& sparse_table, DeviceT & device)
  {
    // make the selection indices from the indices view
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select;
    makeSelectIndicesFromTensorIndicesComponent(this->indices_view_, indices_select, device);

    // Check that the needed values are in memory
    loadTensorTableBinary(dir_, device);

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
    std::shared_ptr<TensorData<int, DeviceT, 2>> sparse_labels;
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
  inline void TensorTable<TensorT, DeviceT, TDim>::appendToAxisConcept(const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T[]>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_copy = std::reinterpret_pointer_cast<TensorT[]>(values);
      appendToAxis(axis_name, labels, values_copy, indices, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT[]>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
		if (getDataTensorSize() > 0) {
			// Check that the needed values are in memory
			// TODO [not_in_memory]: only the shards on the "edge" of the insert will be needed
			loadTensorTableBinary(dir_, device);
		}

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
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(data_->getDataPointer().get(), data_->getDimensions());
		if (value_dimensions == new_dimensions) {
			data_values.device(device) = values_new_values;
		}
		else {
			Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy_values(data_copy->getDataPointer().get(), data_copy->getDimensions());
			data_values.device(device) = data_copy_values.concatenate(values_new_values, axes_to_dims_.at(axis_name));
		}
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::appendToIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // copy the current indices
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_copy = indices_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> is_modified_copy = is_modified_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> in_memory_copy = not_in_memory_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> shard_id_copy = shard_id_.at(axis_name)->copy(device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> shard_indices_copy = shard_indices_.at(axis_name)->copy(device);
    indices_copy->syncHAndDData(device);
    indices_view_copy->syncHAndDData(device);
    is_modified_copy->syncHAndDData(device);
    in_memory_copy->syncHAndDData(device);
    shard_id_copy->syncHAndDData(device);
    shard_indices_copy->syncHAndDData(device);

    // resize and reset the indices
    Eigen::array<Eigen::Index, 1> new_dimensions = indices_.at(axis_name)->getDimensions();
    new_dimensions.at(0) += indices->getDimensions().at(0);
    indices_.at(axis_name)->setDimensions(new_dimensions); 
    indices_.at(axis_name)->setData();
    indices_.at(axis_name)->setDataStatus(false, true);
    indices_view_.at(axis_name)->setDimensions(new_dimensions); 
    indices_view_.at(axis_name)->setData();
    indices_view_.at(axis_name)->setDataStatus(false, true);
    is_modified_.at(axis_name)->setDimensions(new_dimensions); 
    is_modified_.at(axis_name)->setData();
    is_modified_.at(axis_name)->setDataStatus(false, true);
    not_in_memory_.at(axis_name)->setDimensions(new_dimensions); 
    not_in_memory_.at(axis_name)->setData();
    not_in_memory_.at(axis_name)->setDataStatus(false, true);
    shard_id_.at(axis_name)->setDimensions(new_dimensions); 
    shard_id_.at(axis_name)->setData();
    shard_id_.at(axis_name)->setDataStatus(false, true);
    shard_indices_.at(axis_name)->setDimensions(new_dimensions);
    shard_indices_.at(axis_name)->setData();
    shard_indices_.at(axis_name)->setDataStatus(false, true);

    // concatenate the new indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_new_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_copy_values(indices_copy->getDataPointer().get(), (int)indices_copy->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_copy_values(indices_view_copy->getDataPointer().get(), (int)indices_view_copy->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified_copy_values(is_modified_copy->getDataPointer().get(), (int)is_modified_copy->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> in_memory_copy_values(in_memory_copy->getDataPointer().get(), (int)in_memory_copy->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_id_copy_values(shard_id_copy->getDataPointer().get(), (int)shard_id_copy->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_indices_copy_values(shard_indices_copy->getDataPointer().get(), (int)shard_indices_copy->getTensorSize());

    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(indices_view_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified_values(is_modified_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> in_memory_values(not_in_memory_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_id_values(shard_id_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_indices_values(shard_indices_.at(axis_name)->getDataPointer().get(), new_dimensions.at(0));

    const Eigen::array<std::pair<int, int>, 1> padding_1({ std::make_pair(0, indices->getDimensions().at(0)) });
    const Eigen::array<std::pair<int, int>, 1> padding_2({ std::make_pair((int)indices_copy->getTensorSize(), 0) });
    indices_values.device(device) = indices_copy_values.pad(padding_1) + indices_new_values.pad(padding_2);
    indices_view_values.device(device) = indices_view_copy_values.pad(padding_1) + indices_new_values.pad(padding_2);
    is_modified_values.device(device) = is_modified_copy_values.pad(padding_1) + indices_new_values.constant(1).pad(padding_2);
    in_memory_values.device(device) = in_memory_copy_values.pad(padding_1) + indices_new_values.constant(0).pad(padding_2);
    // TODO: need to add in logic to check if the shard size has been exceeded, and if so, start a new shard
    shard_id_values.device(device) = shard_id_copy_values.pad(padding_1) + indices_new_values.constant(1).pad(padding_2);
    shard_indices_values.device(device) = shard_indices_copy_values.pad(padding_1) + indices_new_values.constant(0).pad(padding_2);
    reShardIndices(device);

    // update the `in_memory` and `is_modified` for all other dimensions
    for (const auto& axis_to_dim : axes_to_dims_) {
      if (axis_to_dim.first != axis_name) {
        Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified_values(is_modified_.at(axis_to_dim.first)->getDataPointer().get(), dimensions_.at(axis_to_dim.second));
        Eigen::TensorMap<Eigen::Tensor<int, 1>> in_memory_values(not_in_memory_.at(axis_to_dim.first)->getDataPointer().get(), dimensions_.at(axis_to_dim.second));
        is_modified_values.device(device) = is_modified_values.constant(1);
        in_memory_values.device(device) = in_memory_values.constant(0);
      }
    }

    // update the dimensions
    dimensions_.at(axes_to_dims_.at(axis_name)) += indices->getTensorSize();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT, typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromAxisConcept(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T[]>& values, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_copy = std::reinterpret_pointer_cast<TensorT[]>(values);
      deleteFromAxis(axis_name, indices, labels, values_copy, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT[]>& values, DeviceT & device)
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
      auto indices_selected_2d = (indices_view_bcast != indices_bcast).select(indices_view_bcast, indices_view_bcast.constant(0));
      //NOTE: product of many #s can result in overflow which will then be converted to 0 erroneously by clip
      indices_select_values.device(device) = indices_selected_2d.clip(0, 1).prod(Eigen::array<Eigen::Index, 1>({ 1 })).clip(0, 1);
    }
    else {
      auto indices_selected_2d = (indices_view_bcast == indices_bcast).select(indices_view_bcast, indices_view_bcast.constant(0));
      indices_select_values.device(device) = indices_selected_2d.clip(0, 1).sum(Eigen::array<Eigen::Index, 1>({ 1 })).clip(0, 1);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::deleteFromIndices(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // determine the new indices sizes
    Eigen::array<Eigen::Index, 1> new_dimensions = indices_.at(axis_name)->getDimensions();
    new_dimensions.at(0) -= indices->getDimensions().at(0);

		if (new_dimensions.at(0) > 0) {
			// copy and resize the current indices
			std::shared_ptr<TensorData<int, DeviceT, 1>> indices_copy = indices_.at(axis_name)->copy(device);
			std::shared_ptr<TensorData<int, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy(device);
			std::shared_ptr<TensorData<int, DeviceT, 1>> is_modified_copy = is_modified_.at(axis_name)->copy(device);
			std::shared_ptr<TensorData<int, DeviceT, 1>> in_memory_copy = not_in_memory_.at(axis_name)->copy(device);
			std::shared_ptr<TensorData<int, DeviceT, 1>> shard_id_copy = shard_id_.at(axis_name)->copy(device);
			std::shared_ptr<TensorData<int, DeviceT, 1>> shard_indices_copy = shard_indices_.at(axis_name)->copy(device);
 			indices_copy->setDimensions(new_dimensions);
			indices_copy->setData();
			indices_view_copy->setDimensions(new_dimensions);
			indices_view_copy->setData();
			is_modified_copy->setDimensions(new_dimensions);
			is_modified_copy->setData();
			in_memory_copy->setDimensions(new_dimensions);
			in_memory_copy->setData();
			shard_id_copy->setDimensions(new_dimensions);
			shard_id_copy->setData();
			shard_indices_copy->setDimensions(new_dimensions);
			shard_indices_copy->setData();
			indices_copy->syncHAndDData(device);
			indices_view_copy->syncHAndDData(device);
			is_modified_copy->syncHAndDData(device);
			in_memory_copy->syncHAndDData(device);
			shard_id_copy->syncHAndDData(device);
			shard_indices_copy->syncHAndDData(device);

			// make the selection tensor based off of the selection indices
			std::shared_ptr<TensorData<int, DeviceT, 1>> selection_indices;
			makeIndicesViewSelectFromIndices(axis_name, selection_indices, indices, true, device);

			// select the values based on the indices
			indices_.at(axis_name)->select(indices_copy, selection_indices, device);
			indices_view_.at(axis_name)->select(indices_view_copy, selection_indices, device);
			is_modified_.at(axis_name)->select(is_modified_copy, selection_indices, device);
			not_in_memory_.at(axis_name)->select(in_memory_copy, selection_indices, device);
			shard_id_.at(axis_name)->select(shard_id_copy, selection_indices, device);
			shard_indices_.at(axis_name)->select(shard_indices_copy, selection_indices, device);

			// swap the indices
			indices_.at(axis_name) = indices_copy;
			indices_view_.at(axis_name) = indices_view_copy;
			is_modified_.at(axis_name) = is_modified_copy;
			not_in_memory_.at(axis_name) = in_memory_copy;
			shard_id_.at(axis_name) = shard_id_copy;
			shard_indices_.at(axis_name) = shard_indices_copy;
		}
		else {
			// resize the indices and reset the data
			indices_.at(axis_name)->setDimensions(new_dimensions);
			indices_view_.at(axis_name)->setDimensions(new_dimensions);
			is_modified_.at(axis_name)->setDimensions(new_dimensions);
			not_in_memory_.at(axis_name)->setDimensions(new_dimensions);
			shard_id_.at(axis_name)->setDimensions(new_dimensions);
			shard_indices_.at(axis_name)->setDimensions(new_dimensions);
		}

    // update the dimensions
    dimensions_.at(axes_to_dims_.at(axis_name)) -= indices->getTensorSize();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT, typename T>
  inline void TensorTable<TensorT, DeviceT, TDim>::insertIntoAxisConcept(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<T[]>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      auto values_copy = std::reinterpret_pointer_cast<TensorT[]>(values);
      insertIntoAxis(axis_name, labels, values_copy, indices, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels, std::shared_ptr<TensorT[]>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // Append the new labels and values
    std::shared_ptr<TensorData<int, DeviceT, 1>> indices_append;
    appendToAxis(axis_name, labels, values, indices_append, device);

    // Swap the appended indices for the original indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_.at(axis_name)->getDataPointer().get(), indices_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_old_values(indices->getDataPointer().get(), indices->getDimensions());
    Eigen::array<Eigen::Index, 1> offsets = { (int)indices_view_.at(axis_name)->getTensorSize() - (int)indices->getTensorSize() };
    Eigen::array<Eigen::Index, 1> extents = { (int)indices->getTensorSize() };
    indices_view_values.slice(offsets, extents).device(device) = indices_old_values;
    indices_values.slice(offsets, extents).device(device) = indices_old_values;

    // Sort the is_modified values by the indices, and then sort the indices and update the shard indices
    is_modified_.at(axis_name)->sort(indices_.at(axis_name), device);
    indices_.at(axis_name)->sort("ASC", device); // NOTE: this could fail if there are 0's in the index!
    reShardIndices(device);

    // Sort the axis and tensor based on the indices view
    sortTensorData(device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::string TensorTable<TensorT, DeviceT, TDim>::makeTensorTableShardFilename(const std::string& dir, const std::string& tensor_table_name, const int& shard_id) {
    return dir + tensor_table_name + "_" + std::to_string(shard_id) + ".tts";
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::makeModifiedShardIDTensor(std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, DeviceT & device) const
  {
    // Make the selection indices from the modified tensor indices
    std::shared_ptr<TensorData<int, DeviceT, TDim>> select_indices;
    makeSelectIndicesFromTensorIndicesComponent(is_modified_, select_indices, device);

    // Make the sort indices from the `shard_id` values
    std::shared_ptr<TensorData<int, DeviceT, TDim>> shard_indices;
    makeShardIndicesFromShardIDs(shard_indices, device);

    // Select the `shard_id` values to use for writing
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> select_indices_values(select_indices->getDataPointer().get(), select_indices->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> shard_indices_values(shard_indices->getDataPointer().get(), shard_indices->getDimensions());
    shard_indices_values.device(device) = (select_indices_values > select_indices_values.constant(0)).select(shard_indices_values, shard_indices_values.constant(0));

    // Sort and then RunLengthEncode
    shard_indices->sort("ASC", device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> unique, count, num_runs;
    runLengthEncodeIndex(shard_indices, unique, count, num_runs, device);

    // Resize the unique results and remove 0's from the unique
    makeShardIDTensor(modified_shard_ids, unique, num_runs, device);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::makeNotInMemoryShardIDTensor(std::shared_ptr<TensorData<int, DeviceT, 1>>& modified_shard_ids, DeviceT & device) const
  {
    // Make the selection indices from the indices view tensor indices
    std::shared_ptr<TensorData<int, DeviceT, TDim>> select_indices;
    makeSelectIndicesFromTensorIndicesComponent(indices_view_, select_indices, device);

    // Make the selection indices from the in memory tensor indices
    std::shared_ptr<TensorData<int, DeviceT, TDim>> not_in_memory_indices;
    makeSelectIndicesFromTensorIndicesComponent(not_in_memory_, not_in_memory_indices, device);

    // Make the sort indices from the `shard_id` values
    std::shared_ptr<TensorData<int, DeviceT, TDim>> shard_indices;
    makeShardIndicesFromShardIDs(shard_indices, device);

    // Select the `shard_id` values to use for writing
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> not_in_memory_indices_values(not_in_memory_indices->getDataPointer().get(), not_in_memory_indices->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> select_indices_values(select_indices->getDataPointer().get(), select_indices->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> shard_indices_values(shard_indices->getDataPointer().get(), shard_indices->getDimensions());
    shard_indices_values.device(device) = (select_indices_values > select_indices_values.constant(0) &&
      not_in_memory_indices_values > not_in_memory_indices_values.constant(0)).select(shard_indices_values, shard_indices_values.constant(0));

    // Sort and then RunLengthEncode
    shard_indices->sort("ASC", device);
    std::shared_ptr<TensorData<int, DeviceT, 1>> unique, count, num_runs;
    runLengthEncodeIndex(shard_indices, unique, count, num_runs, device);

    // Resize the unique results and remove 0's from the unique
    makeShardIDTensor(modified_shard_ids, unique, num_runs, device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::storeTensorTableAxesBinary(const std::string & dir, DeviceT & device)
  {
    for (auto& axis_map : axes_) {
      std::string filename =  dir + name_ + "_axis_" + axis_map.first;
      axis_map.second->storeLabelsBinary(filename, device);
    }
    return true;
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorTable<TensorT, DeviceT, TDim>::loadTensorTableAxesBinary(const std::string & dir, DeviceT & device)
  {
    for (auto& axis_map : axes_) {
      std::string filename = dir + name_ + "_axis_" + axis_map.first;
      axis_map.second->loadLabelsBinary(filename, device);
    }
    return true;
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline std::vector<std::string> TensorTable<TensorT, DeviceT, TDim>::getCsvDataRow(const int & row_num)
  {
    // Make the slice indices
    Eigen::array<Eigen::Index, 2> offset = {0, row_num};
    Eigen::array<Eigen::Index, 2> span = { getDimensions().at(0), 1 };

    // Make the reshape dimansions
    Eigen::array<Eigen::Index, 2> reshape_dims = getCsvDataDimensions();

    // Return the row as a string
    return getCsvDataRowAsStrings(offset, span, reshape_dims);
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline Eigen::array<Eigen::Index, 2> TensorTable<TensorT, DeviceT, TDim>::getCsvDataDimensions()
  {
    Eigen::array<Eigen::Index, 2> reshape_dims = { 1,1 };
    for (int i = 0; i < TDim; ++i) {
      if (i == 0) {
        reshape_dims.at(0) = getDimensions().at(i);
      }
      else {
        reshape_dims.at(1) *= getDimensions().at(i);
      }
    }
    return reshape_dims;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline Eigen::array<Eigen::Index, 2> TensorTable<TensorT, DeviceT, TDim>::getCsvShardSpans()
  {
    Eigen::array<Eigen::Index, 2> shard_dims = { 1,1 };
    for (auto axis_to_dim: axes_to_dims_) {
      if (axis_to_dim.second == 0) {
        shard_dims.at(0) = shard_spans_.at(axis_to_dim.first);
      }
      else {
        shard_dims.at(1) *= shard_spans_.at(axis_to_dim.first);
      }
    }
    return shard_dims;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, char>::value, int>>
  inline std::vector<std::string> TensorTable<TensorT, DeviceT, TDim>::getCsvDataRowAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span, const Eigen::array<Eigen::Index, 2>& reshape) const
  {
    // Make the slice
    auto row_t = this->getData().reshape(reshape).slice(offset, span);
    Eigen::Tensor<std::string, 2> row = row_t.unaryExpr([](const TensorT& elem) { return std::to_string(static_cast<char>(elem)); });

    // Convert element to a string
    std::vector<std::string> row_vec(row.data(), row.data() + row.size());
    return row_vec;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, bool>::value, int>>
  inline std::vector<std::string> TensorTable<TensorT, DeviceT, TDim>::getCsvDataRowAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span, const Eigen::array<Eigen::Index, 2>& reshape) const
  {
    // Make the slice
    auto row_t = this->getData().reshape(reshape).slice(offset, span);
    Eigen::Tensor<std::string, 2> row = row_t.unaryExpr([](const TensorT& elem) { return std::to_string(elem); });

    // Convert element to a string
    std::vector<std::string> row_vec(row.data(), row.data() + row.size());
    return row_vec;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename T, std::enable_if_t<!std::is_fundamental<T>::value, int>>
  inline std::vector<std::string> TensorTable<TensorT, DeviceT, TDim>::getCsvDataRowAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span, const Eigen::array<Eigen::Index, 2>& reshape) const
  {
    // Make the slice
    auto row_t = this->getData().reshape(reshape).slice(offset, span);
    Eigen::Tensor<std::string, 2> row = row_t.unaryExpr([](const TensorT& elem) { return elem.getTensorArrayAsString(); });

    // Convert element to a string
    std::vector<std::string> row_vec(row.data(), row.data() + row.size());
    return row_vec;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline std::map<std::string, std::vector<std::string>> TensorTable<TensorT, DeviceT, TDim>::getCsvAxesLabelsRow(const int & row_num)
  {
    std::map<std::string, std::vector<std::string>> axes_labels_row;

    // get each labels for the particular row
    int axis_0_size = this->axes_.at(this->axes_to_dims_.begin()->first)->getNLabels();
    int axis_size_cum = axis_0_size;
    for (const auto& axis_to_dim : this->axes_to_dims_) {
      if (axis_to_dim.first != this->axes_to_dims_.begin()->first) {
        // calculate the index corresponding to the 2D row number
        int index = int(floor(float(row_num * axis_0_size) / float(axis_size_cum))) % this->axes_.at(axis_to_dim.first)->getNLabels();

        // Make the slice indices
        Eigen::array<Eigen::Index, 2> offset = { 0, index };
        Eigen::array<Eigen::Index, 2> span = { (int)this->axes_.at(axis_to_dim.first)->getNDimensions(), 1 };

        // Slice out the labels row
        std::vector<std::string> row_vec = this->axes_.at(axis_to_dim.first)->getLabelsAsStrings(offset, span);
        axes_labels_row.emplace(axis_to_dim.first, row_vec);

        // update the accumulative size
        axis_size_cum *= this->axes_.at(axis_to_dim.first)->getNLabels();
      }
    }

    return axes_labels_row;
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::insertIntoTableFromCsv(const std::map<std::string, Eigen::Tensor<std::string, 2>>& labels_new, const Eigen::Tensor<std::string, 2>& data_new, DeviceT & device)
  {
    // add in the new labels
    for (auto& axis_map : axes_) {
      if (axis_map.first != axes_.begin()->first)
        axis_map.second->appendLabelsToAxisFromCsv(labels_new.at(axis_map.first), device);
    }

    insertIntoTableFromCsv(data_new, device);
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::insertIntoTableFromCsv(const Eigen::Tensor<std::string, 2>& data_new, DeviceT & device)
  {
    // Copy the original data
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> data_copy = data_->copy(device);
    data_copy->syncHAndDData(device);

    // Copy the shard indices and set the axes
    std::map<std::string, int> shard_spans_copy = shard_spans_;
    setAxes(device); // NOTE: this will clear the in-memory data
    setShardSpans(shard_spans_copy); // set the shard indices back to what they were

    // Intialize the new data
    setData();

    // TODO: does it make sense to move this over to `setAxes()` ?
    // Sync all of the axes and reShard
    syncIndicesHAndDData(device);
    syncIndicesViewHAndDData(device);
    syncIsModifiedHAndDData(device);
    syncNotInMemoryHAndDData(device);
    syncShardIdHAndDData(device);
    syncShardIndicesHAndDData(device);
    reShardIndices(device);
    syncAxesHAndDData(device);
    syncHAndDData(device);

    // Select the new axis labels
    // NOTE: Due to the non-convex shape of the addition, we are not able to select
    //       the new data without either missing part of the new data or over-writing portions of the previous data
    //for (auto& axis_map : axes_) {
      //// ATTEMPT 1
      //if (axis_map.first != axes_.begin()->first) {
      //  // make the select indices
      //  std::shared_ptr<TensorData<int, DeviceT, 1>> select_indices;
      //  axis_map.second->makeSelectIndicesFromCsv(select_indices, labels_new.at(axis_map.first), device);

      //  // update the indices view based on the selection
      //  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_map.first)->getDataPointer().get(), (int)axes_.at(axis_map.first)->getNLabels());
      //  Eigen::TensorMap<Eigen::Tensor<int, 1>> selected(select_indices->getDataPointer().get(), (int)axes_.at(axis_map.first)->getNLabels());
      //  indices_view.device(device) = indices_view * selected;
      //}
      //// ATTEMPT 1
      //// update the indices view based on the selection
      //Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_map.first)->getDataPointer().get(), (int)axes_.at(axis_map.first)->getNLabels());
      //indices_view.slice(Eigen::array<Eigen::Index, 1>({ 0 }), indices_copy.at(axis_map.first)->getDimensions()).device(device) = indices_view.slice(Eigen::array<Eigen::Index, 1>({ 0 }), indices_copy.at(axis_map.first)->getDimensions()).constant(0);
    //}

    // Reformat into a SparseTensorTable
    std::shared_ptr<TensorTable<TensorT, DeviceT, 2>> sparse_table_ptr;
    makeSparseTensorTableFromCsv(sparse_table_ptr, data_new, device);
    // NOTE: alignment of axes is not enforced in SparseTensorTable so a pre-sort maybe needed...

    // Update the data with the new values as a 1D array
    Eigen::array<Eigen::Index, 1> offsets_new = { (int)data_copy->getTensorSize() };
    Eigen::array<Eigen::Index, 1> extents_new = { (int)sparse_table_ptr->getDataTensorSize() };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_new_values(sparse_table_ptr->getDataPointer().get(), (int)sparse_table_ptr->getDataTensorSize());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_values(data_->getDataPointer().get(), (int)data_->getTensorSize());
    data_values.slice(offsets_new, extents_new).device(device) = data_new_values;

    // Update the data with the original data as a 1D array
    Eigen::array<Eigen::Index, 1> offsets_old = { 0 };
    Eigen::array<Eigen::Index, 1> extents_old = { (int)data_copy->getTensorSize() };
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_old_values(data_copy->getDataPointer().get(), (int)data_copy->getTensorSize());
    data_values.slice(offsets_old, extents_old).device(device) = data_old_values;

    // update the not_in_memory and is_modified attributes
    // NOTE: due to the shape, we are not able to distinguish well what was and what was not modified
    //       so we set everything to modified
    for (const auto& axis_to_dim : axes_to_dims_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(not_in_memory_.at(axis_to_dim.first)->getDataPointer().get(), (int)not_in_memory_.at(axis_to_dim.first)->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(0);
      Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified(is_modified_.at(axis_to_dim.first)->getDataPointer().get(), (int)is_modified_.at(axis_to_dim.first)->getTensorSize());
      is_modified.device(device) = is_modified.constant(1);
    }
  }
};
#endif //TENSORBASE_TENSORTABLE_H