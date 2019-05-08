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

      The method sets the tensor axes and initializes
      the indices, indices_view, is_modified, in_memory, and
      is_shardable attributes after all axes have been added
    */
    virtual void setAxes() = 0; ///< axes setter

    virtual void initData() = 0;  ///< DeviceT specific initializer

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
    @param[in] n_labels
    @param[in] n_labels
    @param[in] n_labels
    @param[in] device
    */
    template<typename LabelsT>
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<LabelsT>& select_labels_data, const int& n_labels, 
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitor& comparitor, const logicalModifier& modifier,
      const logicalContinuator& prepend_continuator, const logicalContinuator& within_continuator, const DeviceT& device);

    /*
    @brief Broadcast the axis indices view across the entire tensor
      and allocate to memory

    @param[out] indices_view_bcast
    @param[in] axis_name
    */
    virtual void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, const std::string& axis_name, DeviceT& device) = 0;

    /*
    @brief Broadcast the axis indices view across the entire tensor,
      auto-increment each tensor to preserve order across the entire tensor,
      and allocate to memory

    @param[out] indices_view_bcast
    @param[in] axis_name
    */
    //virtual void broadcastSortIndicesView(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, const std::string& axis_name) = 0;

    /*
    @brief Select data from the Tensor based on a select index tensor

    @param[in] indices_view_bcast The indices (0 or 1) to select from
    @param[out] tensor_select The selected and reduced tensor
    */
    virtual void selectTensorData(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, DeviceT& device) = 0;

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
    const logicalContinuator& prepend_continuator, const logicalContinuator& within_continuator, const DeviceT& device) {
    // create a copy of the indices view
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> indices_view_copy = indices_view_.at(axis_name)->copy();

    // select the `labels` indices from the axis labels and store in the current indices view
    selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);

    // Reduce the Tensor to `n_labels` using the `labels` indices as the selection criteria
    // TODO: GPU version; see http://nvlabs.github.io/cub/structcub_1_1_device_select.html#details (DeviceSelect::flagged) for selectTensorData
    std::shared_ptr<TensorData<int, DeviceT, TDim>> indices_view_bcast;
    broadcastSelectIndicesView(indices_view_bcast, axis_name, device);
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor_select;
    selectTensorData(indices_view_bcast, tensor_select, axis_name, n_labels, device);

    // determine the dimensions for reshaping and broadcasting the values
    Eigen::array<int, TDim> values_reshape_dimensions;
    Eigen::array<int, TDim> values_bcast_dimensions;
    for (int i = 0; i < TDim; ++i) {
      if (i == axes_to_dims_.at(axis_name)) {
        values_reshape_dimensions.at(i) = n_labels;
        values_bcast_dimensions.at(i) = 1;
      }
      else {
        values_reshape_dimensions.at(i) = 1;
        values_bcast_dimensions.at(i) = dimensions_.at(i);
      }
    }

    // broadcast the comparitor values across the selected tensor dimensions
    Eigen::TensorMap<Eigen::Tensor<LabelsT, TDim>> values_reshape(values.get(), values_reshape_dimensions);
    auto values_bcast = values_reshape.broadcast(values_bcast_dimensions);

    // apply the logical comparitor and modifier as a selection criteria
    if (comparitor == logicalComparitor::EQUAL_TO) {
      auto pass_selection_criteria = (values_bcast == tensor_select).select(tensor_select.constant(1), tensor_select.constant(0));
    }
    // TODO: all other comparators
    
    // update all other tensor indices view based on the selection criteria tensor
    for (int i = 0; i < TDim; ++i) {
      if (i == axes_to_dims_.at(axis_name)) continue;

      // build the continuator reduction indices
      Eigen::array<int, TDim - 1> reduction_dims;
      int index = 0;
      for (int j = 0; j < TDim; ++j) {
        if (j != axes_to_dims_.at(axis_name)) {
          reduction_dims.at(index) = j;
          ++index;
        }
      }

      // apply the continuator reduction
      if (within_continuator == logicalContinuator::OR) {
        auto indicesView_update_tmp = pass_selection_criteria.sum(reduction_dims);
        auto indicesView_update = indicesView_update_tmp / indicesView_update_tmp; //ensure a max value of 1
      }
      else if (within_continuator == logicalContinuator::AND) {
        auto indicesView_update = pass_selection_criteria.prod(reduction_dims);
      }

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuator::OR) {
        Eigen::TensorMap<Eigen::Tensor<int, 1>> indices(indices_.at(axis_name)->getDataPointer().get(), indices_.at(axis_name)->getDimensions());
        auto indicesView_update_select = (indicesView_update > 0).select(indices, indices.constant(0));
        indicesView_copy->getData().device(device) += indicesView_update_select;
      }
      else if (prepend_continuator == logicalContinuator::AND) {
        indicesView_copy->getData().device(device) = indicsView_copy->getData() * indicesView_update;
      }
    }
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::resetIndicesView(const std::string& axis_name, const DeviceT& device)
  {    
    indices_view_.at(axis_name)->getData().device(device) = indices_.at(axis_name)->getData();
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::zeroIndicesView(const std::string & axis_name, const DeviceT& device)
  {
    indices_view_.at(axis_name)->getData().device(device) = indices_view_.at(axis_name)->getData().constant(0);
  };

  template<typename TensorT, int TDim>
  class TensorTableDefaultDevice: public TensorTable<TensorT, Eigen::DefaultDevice, TDim>
  {
  public:
    TensorTableDefaultDevice() = default;
    TensorTableDefaultDevice(const std::string& name) { this->setName(name); };
    ~TensorTableDefaultDevice() = default;
    void setAxes() override;
    void initData() override;
    void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, const std::string& axis_name, Eigen::DefaultDevice& device) override;
    void selectTensorData(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::DefaultDevice& device) override;
  };

  template<typename TensorT, int TDim>
  void TensorTableDefaultDevice<TensorT, TDim>::setAxes() {
    assert(TDim == axes_.size()); // "The number of tensor_axes and the template TDim do not match.";

    // Determine the overall dimensions of the tensor
    int axis_cnt = 0;
    for (auto& axis : axes_) {
      dimensions_.at(axis_cnt) = axis.second->getNLabels();
      Eigen::array<Eigen::Index, 1> axis_dimensions = { (int)axis.second->getNLabels() };

      // Set the axes name to dim map
      axes_to_dims_.emplace(axis.second->getName(), axis_cnt);

      // Set the indices
      Eigen::Tensor<int, 1> indices_values(axis.second->getNLabels());
      for (int i = 0; i < axis.second->getNLabels(); ++i) {
        indices_values(i) = i + 1;
      }
      TensorDataDefaultDevice<int, 1> indices(axis_dimensions);
      indices.setData(indices_values);
      indices_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(indices));

      // Set the indices view
      TensorDataDefaultDevice<int, 1> indices_view(axis_dimensions);
      indices_view.setData(indices_values);
      indices_view_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_view));

      // Set the is_modified defaults
      Eigen::Tensor<int, 1> is_modified_values(axis.second->getNLabels());
      is_modified_values.setZero();
      TensorDataDefaultDevice<int, 1> is_modified(axis_dimensions);
      is_modified.setData(is_modified_values);
      is_modified_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(is_modified));

      // Set the in_memory defaults
      Eigen::Tensor<int, 1> in_memory_values(axis.second->getNLabels());
      in_memory_values.setZero();
      TensorDataDefaultDevice<int, 1> in_memory(axis_dimensions);
      in_memory.setData(in_memory_values);
      in_memory_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(in_memory));

      // Set the in_memory defaults
      Eigen::Tensor<int, 1> is_shardable_values(axis.second->getNLabels());
      if (axis_cnt == 0)
        is_shardable_values.setConstant(1);
      else
        is_shardable_values.setZero();
      TensorDataDefaultDevice<int, 1> is_shardable(axis_dimensions);
      is_shardable.setData(is_shardable_values);
      is_shardable_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(is_shardable));

      // Next iteration
      ++axis_cnt;
    }

    // Allocate memory for the tensor
    initData();
  };

  template<typename TensorT, int TDim>
  void TensorTableDefaultDevice<TensorT, TDim>::initData() {
    this->getData().reset(new TensorDataDefaultDevice<TensorT, TDim>(this->getDimensions()));
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, const std::string & axis_name, Eigen::DefaultDevice& device)
  {
    // determine the dimensions for reshaping and broadcasting the indices
    Eigen::array<int, TDim> indices_reshape_dimensions;
    Eigen::array<int, TDim> indices_bcast_dimensions;
    for (int i = 0; i < TDim; ++i) {
      if (i == axes_to_dims_.at(axis_name)) {
        indices_reshape_dimensions.at(i) = (int)axes_.at(axis_name)->getNLabels();
        indices_bcast_dimensions.at(i) = 1;
      }
      else {
        indices_reshape_dimensions.at(i) = 1;
        indices_bcast_dimensions.at(i) = dimensions_.at(i);
      }
    }

    // broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(indices_view_.at(axis_name)->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.broadcast(indices_bcast_dimensions);

    // allocate to memory
    TensorDataDefaultDevice<int, 3> indices_view_bcast_tmp(dimensions_);
    indices_view_bcast_tmp.setData();
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_bcast_map(indices_view_bcast_tmp.getDataPointer().get(), indices_reshape_dimensions);
    indices_view_bcast_map.device(device) = indices_view_bcast_values;
    
    // move over the results
    indices_view_bcast = std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_view_bcast_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::selectTensorData(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, 
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::DefaultDevice& device)
  {
    // determine the dimensions for the making the selected tensor
    Eigen::array<Eigen::Index, TDim> tensor_select_dimensions;
    Eigen::array<Eigen::Index, 1> tensor_select_1d_dimensions; tensor_select_1d_dimensions.at(0) = 1;
    Eigen::array<Eigen::Index, 1> tensor_1d_dimensions; tensor_1d_dimensions.at(0) = 1;
    for (int i = 0; i < TDim; ++i) {
      if (i == axes_to_dims_.at(axis_name)) {
        tensor_select_dimensions.at(i) = n_select;
        tensor_select_1d_dimensions.at(0) *= n_select;
        tensor_1d_dimensions.at(0) *= dimensions_.at(i);
      }
      else {
        tensor_select_dimensions.at(i) = dimensions_.at(i);
        tensor_select_1d_dimensions.at(0) *= dimensions_.at(i);
        tensor_1d_dimensions.at(0) *= dimensions_.at(i);
      }
    }

    // allocate memory for the selected tensor
    TensorDataDefaultDevice<TensorT, TDim> tensor_select_tmp(tensor_select_dimensions);
    Eigen::Tensor<TensorT, TDim> tensor_select_data(tensor_select_dimensions);
    tensor_select_data.setZero();
    tensor_select_tmp.setData(tensor_select_data);

    // apply the device specific select algorithm
    int iter_select = 0;
    int iter_tensor = 0;
    std::for_each(indices_view_bcast->getDataPointer().get(), indices_view_bcast->getDataPointer().get() + indices_view_bcast->getData().size(),
      [&](const int& index) {
      if (index > 0) {
        tensor_select_tmp.getData().data()[iter_select] = this->data_->getData().data()[iter_tensor];
        ++iter_select;
      }
      ++iter_tensor;
    });

    // move over the results
    tensor_select = std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(tensor_select_tmp);
  }

  template<typename TensorT, int TDim>
  class TensorTableCpu : public TensorTable<TensorT, Eigen::ThreadPoolDevice, TDim>
  {
  public:
    TensorTableCpu() = default;
    TensorTableCpu(const std::string& name) { this->setName(name); };
    ~TensorTableCpu() = default;
    void setAxes() override;
    void initData() override;
  };

  template<typename TensorT, int TDim>
  void TensorTableCpu<TensorT, TDim>::initData() {
    this->getData().reset(new TensorDataCpu<TensorT, TDim>(this->getDimensions()));
  }

#if COMPILE_WITH_CUDA
  template<typename TensorT, int TDim>
  class TensorTableGpu : public TensorTable<TensorT, Eigen::GpuDevice, TDim>
  {
  public:
    TensorTableGpu() = default;
    TensorTableGpu(const std::string& name) { this->setName(name); };
    ~TensorTableGpu() = default;
    void setAxes() override;
    void initData() override;
  };

  template<typename TensorT, int TDim>
  void TensorTableGpu<TensorT, TDim>::initData() {
    this->getData().reset(new TensorDataGpu<TensorT, TDim>(this->getDimensions()));
  }
#endif
};
#endif //TENSORBASE_TENSORTABLE_H