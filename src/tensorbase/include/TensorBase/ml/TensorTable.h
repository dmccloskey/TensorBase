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

    std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() { return axes_; }; ///< axes getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndices() { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndicesView() { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsModified() { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getInMemory() { return in_memory_; }; ///< in_memory getter
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsShardable() { return is_shardable_; }; ///< is_shardable getter
    Eigen::array<Eigen::Index, TDim>& getDimensions() { return dimensions_; }  ///< dimensions getter
    int getDimFromAxisName(const std::string& axis_name) { return axes_to_dims_.at(axis_name); }
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& getData() { return data_; }; ///< data getter
    void clear();  ///< clears the axes and all associated data

    /*
    @brief Select Tensor Axis that will be included in the view

    @param[in] axis_name
    @param[in] dimension_index
    @param[in] select_labels_data
    @param[in] n_labels
    @param[in] device
    */
    template<typename LabelsT>
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int& n_labels, const DeviceT& device);
    void resetIndicesView(const std::string& axis_name, const DeviceT& device); ///< copy over the indices values to the indices view
    void zeroIndicesView(const std::string& axis_name, const DeviceT& device); ///< set the indices view to zero

    /*
    @brief Order Tensor Axis View

    @param[in] axis_name
    @param[in] dimension_index
    @param[in] select_labels_data
    @param[in] n_labels
    @param[in] device
    */
    template<typename LabelsT>
    void orderIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int& n_labels, const DeviceT& device);

    /*
    @brief Apply a where selection clause to the Tensor Axis View

    @param[in] axis_name
    @param[in] dimension_index
    @param[in] select_labels_data
    @param[in] n_labels
    @param[in] values
    @param[in] comparitor
    @param[in] modifier
    @param[in] within_continuator
    @param[in] prepend_continuator
    @param[in] device
    */
    template<typename LabelsT>
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int& n_labels, 
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitor& comparitor, const logicalModifier& modifier,
      const logicalContinuator& within_continuator, const logicalContinuator& prepend_continuator, const DeviceT& device);

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
    @param[in] n_select The size of the reduced axis
    @param[in] device
    */
    virtual void extractTensorData(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, DeviceT& device) = 0;

    /*
    @brief Select indices from the tensor based on a selection criteria

    @param[out] indices_select The indices that passed or did not pass the selection criteria
    @param[in] values_select The values to use for comparison
    @param[in] tensor_select The to apply the selection criteria to
    @param[in] axis_name The name of the axis to reduce along
    @param[in] n_select The size of the reduced axis
    @param[in] comparitor The logical comparitor to apply
    @param[in] modifier The logical modifier to apply to the comparitor (i.e., Not; currently not implemented)
    @param[in] device
    */
    virtual void selectTensorIndices(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, const logicalComparitor& comparitor, const logicalModifier& modifier, DeviceT& device) = 0;

    /*
    @brief Apply the indices select to the indices view for the respective axis
      using the logical continuator and prepender

    @param[in] indices_select The indices that passed or did not pass the selection criteria
    @param[in] axis_name_select The name of the axis that the selection was applied to
    @param[in] axis_name The name of the axis to apply the selection on
    @param[in] 
    @param[in] comparitor The logical comparitor to apply
    @param[in] modifier The logical modifier to apply to the comparitor (i.e., Not; currently not implemented)
    @param[in] device
    */
    virtual void applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_select, const std::string & axis_name_select, const std::string& axis_name, const logicalContinuator& within_continuator, const logicalContinuator& prepend_continuator, DeviceT& device) = 0;

    /*
    @brief Broadcast the axis indices view across the entire tensor,
      auto-increment each tensor to preserve order across the entire tensor,
      and allocate to memory

    @param[out] indices_view_bcast
    @param[in] axis_name
    */
    //virtual void broadcastSortIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, const std::string& axis_name) = 0;

    /*
    @brief Sort data from the Tensor based on a sort index tensor
    */
    //virtual void sortTensorData(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_sort, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, DeviceT& device) = 0;

  protected:
    int id_ = -1;
    std::string name_ = "";

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>();
    std::map<std::string, std::shared_ptr<TensorAxisConcept>> axes_; ///< primary axis is dim=0
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
    auto found = axes_.emplace(tensor_axis->getName(), std::shared_ptr<TensorAxisConcept>(new TensorAxisWrapper<T>(tensor_axis)));
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
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::selectIndicesView(const std::string & axis_name, const int& dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int & n_labels, const DeviceT & device)
  {
    // reshape to match the axis labels shape
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_names_selected_reshape(select_labels_data.get(), 1, n_labels);
    // broadcast the length of the labels
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<int, 2>({ (int)axes_.at(axis_name)->getNLabels(), 1 }));
    // broadcast the axis labels the size of the labels queried
    std::shared_ptr<LabelsT> labels_data;
    axes_.at(axis_name)->getLabelsDataPointer(labels_data);
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(labels_data.get(), (int)axes_.at(axis_name)->getNDimensions(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = (labels_reshape.chip(dimension_index, 0)).broadcast(Eigen::array<int, 2>({ 1, n_labels }));
    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, n_labels }));
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
    auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 }));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels());
    indices_view.device(device) += selected_sum;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::orderIndicesView(const std::string & axis_name, const int & dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int & n_labels, const DeviceT & device)
  {
    // TODO extract out the columns
    // TODO sort the columns and update the axes indices according to the sort values
  }

  template<typename TensorT, typename DeviceT, int TDim>
  template<typename LabelsT>
  inline void TensorTable<TensorT, DeviceT, TDim>::whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int& n_labels,
    const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitor& comparitor, const logicalModifier& modifier,
    const logicalContinuator& within_continuator, const logicalContinuator& prepend_continuator, const DeviceT& device) {
    // create a copy of the indices view
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy();

    // select the `labels` indices from the axis labels and store in the current indices view
    selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);

    // Reduce the Tensor to `n_labels` using the `labels` indices as the selection criteria
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_view_bcast;
    broadcastSelectIndicesView(indices_view_bcast, axis_name, device);
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor_select;
    extractTensorData(indices_view_bcast, tensor_select, axis_name, n_labels, device);

    // Determine the indices that pass the selection criteria
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_select;
    selectTensorIndices(indices_select, values, tensor_select, axis_name, n_labels, comparitor, modifier, device);

    // revert back to the origin indices view
    indices_view_.at(axis_name)->getDataPointer() = indices_view_copy->getDataPointer();
    
    // update all other tensor indices view based on the selection criteria tensor
    for (const auto& axis_to_name: axes_to_dims_) {
      if (axis_to_name.first == axis_name) continue;
      applyIndicesSelectToIndicesView(indices_select, axis_name, axis_to_name.first, within_continuator, prepend_continuator, device);
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::resetIndicesView(const std::string& axis_name, const DeviceT& device)
  {
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices(indices_.at(axis_name)->getDataPointer().get(), indices_.at(axis_name)->getDimensions());
    indices_view.device(device) = indices;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::zeroIndicesView(const std::string & axis_name, const DeviceT& device)
  {
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), indices_view_.at(axis_name)->getDimensions());
    indices_view.device(device) = indices_view.constant(0);
  };
};
#endif //TENSORBASE_TENSORTABLE_H