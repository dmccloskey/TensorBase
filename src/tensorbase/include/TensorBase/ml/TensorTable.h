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
    // #1 Select the `labels` indices from the tensor

    // create a copy of the indices view
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> indicesView_copy(indices_view_.at(axis_name)->getDimensions());
    indicesView_copy.setData(indices_view_.at(axis_name)->getData());

    // update the indices view
    selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);

    // #2 Reduce the Tensor to `n_labels` using the `labels` indices as the selection criteria

    Eigen::TensorMap<Eigen::Tensor<int, TDim>> tensor(data_->getDataPointer().get(), dimensions_);
    if (std::is_arithmetic<TensorT>::value) {
      // convert the indices to the an identity tensor
      Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels(), 1);
      auto indices_identity_1 = indices_identity / indices_identity;
      auto indices_identity = indices_identity_1.contract(indices_identity_1, Eigen::array<Eigen::IndexPair<int>, 1>({ Eigen::IndexPair(1, 0) }));

      // zero all non-selected indices using `.cast<T>` and `.contract()`
      auto tensor_select = tensor.contract(indices_identity.cast<TensorT>(), Eigen::array<Eigen::IndexPair<int>, 1>({ Eigen::IndexPair(axes_to_dims_.at(axis_name), 0) }));
    }
    else {
    // broadcast the indices across the tensor 
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_reshape(indices_view_.at(axis_name)->getDataPointer().get(), (int)axes_.at(axis_name)->getNLabels(), 1, 1);
      auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, n_labels }));

      // zero all non-selected indices using `.select` if char/string type
    }

    // #3 Apply the logical comparitor and modifier as a selection criteria

    // broadcast the values across the tensor

    // apply the comparitor 

    // #4 Reduce the selected indices using the within continuator (i.e., Sum or Prod)
    // auto indicesView_update = .Prod(...);
    // if (OR) indicesView_update = .Sum(...);

    // #5 Update the indices view using the prepend continuator (i.e., += or *= )
    // if (AND) indicsView_copy->getData().device(device) = indicsView_copy->getData() * indicesView_update;
    // else indicsView_copy->getData().device(device) += indicesView_update;
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