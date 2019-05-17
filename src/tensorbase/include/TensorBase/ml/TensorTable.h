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
    virtual ~TensorTable() = default; ///< Default destructor

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
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndices() { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndicesView() { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsModified() { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getInMemory() { return in_memory_; }; ///< in_memory getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsShardable() { return is_shardable_; }; ///< is_shardable getter
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

    bool syncAxesHAndDData(DeviceT& device); ///< Sync the host and device axes data
    void setAxesDataStatus(const bool& h_data_updated, const bool& d_data_updated);///< Set the status of the host and device axes data
    std::map<std::string, std::pair<bool, bool>> getAxesDataStatus(); ///< Get the status of the host and device axes data

    bool syncHAndDData(DeviceT& device) { return data_->syncHAndDData(device); };  ///< Sync the host and device tensor data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { data_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device tensor data
    std::pair<bool, bool> getDataStatus() { return data_->getDataStatus(); };   ///< Get the status of the host and device tensor data

    /*
    @brief Select Tensor Axis that will be included in the view

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
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const LabelsT& label, const sortOrder::order& order_by, DeviceT& device);

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

    @param[out] indices_view_bcast
    @param[in] axis_name
    */
    virtual void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, const std::string& axis_name, DeviceT& device) = 0;
    
    /*
    @brief Extract data from the Tensor based on a select index tensor

    @param[in] indices_view_bcast The indices (0 or 1) to select from
    @param[out] tensor_select The selected tensor with reduced dimensions according to the indices_view_bcast indices
    @param[in] axis_name The name of the axis to reduce along
    @param[in] n_select The size of the reduced dimensions
    @param[in] device
    */
    virtual void reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, DeviceT& device) = 0;

    /*
    @brief Select indices from the tensor based on a selection criteria

    @param[out] indices_select The indices that passed or did not pass the selection criteria
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
    void applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const std::string & axis_name_select, const std::string& axis_name, const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device);

    /*
    @brief Slice out the 1D Tensor that will be sorted on

    @param[out] tensor_sort The 1D Tensor to sort
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

    @param[out] indices_select Pointer to the indices Tensor
    @param[in] device
    */
    virtual void makeSelectIndicesFromIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, DeviceT& device) = 0;

    /*
    @brief Select the Tensor data and return the selected/reduced data

    @param[out] tensor_select The selected/reduced tensor data
    @param[in] indices_select The indices to perform the selection on
    @param[in] device
    */
    virtual void getSelectTensorData(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, DeviceT& device) = 0;

    /*
    @brief Sort the Tensor Data based on the sort order defined by the indices view
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

    @param[out] indices_sort pointer to the indices sort Tensor
    @param[in] device
    */
    virtual void makeSortIndicesViewFromIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_sort, DeviceT& device) = 0;

  protected:
    int id_ = -1;
    std::string name_ = "";

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>();
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> axes_; ///< primary axis is dim=0
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices_; ///< starting at 1
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> indices_view_; ///< sorted and/or selected indices
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> is_modified_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> in_memory_;
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> is_shardable_;
    std::map<std::string, int> axes_to_dims_;
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> data_; ///< The actual tensor data
    
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
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
    // reshape to match the axis labels shape
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_names_selected_reshape(select_labels->getDataPointer().get(), 1, (int)select_labels->getData().size());
    // broadcast the length of the labels
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<int, 2>({ (int)axes_.at(axis_name)->getNLabels(), 1 }));
    // broadcast the axis labels the size of the labels queried
    std::shared_ptr<LabelsT> labels_data;
    axes_.at(axis_name)->getLabelsDataPointer(labels_data);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(labels_data.get(), (int)axes_.at(axis_name)->getNDimensions(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = (labels_reshape.chip(dimension_index, 0)).broadcast(Eigen::array<int, 2>({ 1, (int)select_labels->getData().size() }));
    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, (int)select_labels->getData().size() }));
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
    auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 }));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels());
    indices_view.device(device) = (indices_view * selected_sum) / indices_view;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::sortIndicesView(const std::string & axis_name, const int & dimension_index, const LabelsT& label, const sortOrder::order& order_by, DeviceT & device)
  {
    // find the index of the label
    int label_index = 0;
    std::shared_ptr<LabelsT> labels_data;
    axes_.at(axis_name)->getLabelsDataPointer(labels_data);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_values(labels_data.get(), (int)axes_.at(axis_name)->getNDimensions(), (int)axes_.at(axis_name)->getNLabels());
    for (int i = 0; i < axes_.at(axis_name)->getNLabels(); ++i) {
      if (labels_values(dimension_index, i) == label) {
        label_index = i;
        break;
      }
    }

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
      if (TDim - 2 > 0) {
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
    getSelectTensorData(tensor_select, indices_select, device);

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

    // apply the sort indices to the tensor data and reset the indices view
    data_->sort(indices_sort, device);
    for (const auto& axis_to_index: axes_to_dims_) {
      resetIndicesView(axis_to_index.first, device);
    }
  }
};
#endif //TENSORBASE_TENSORTABLE_H