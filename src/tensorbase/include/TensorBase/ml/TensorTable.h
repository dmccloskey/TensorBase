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
#include <TensorBase/ml/TensorAxis.h>

namespace TensorBase
{
  /// The erasure interface for TensorAxis
  class TensorAxisConcept {
  public:
    virtual std::string getName() const = 0;
    virtual size_t getNLabels() const = 0;
    virtual size_t getNDimensions() const = 0;
    virtual Eigen::Tensor<std::string, 1>& getDimensions() = 0;
    virtual std::shared_ptr<void> getLabelsDataPointer() = 0;
  };

  /// The erasure wrapper around the Tensor Axis interface
  template<typename T>
  class TensorAxisWrapper : public TensorAxisConcept {
    std::shared_ptr<T> tensor_axis_;
  public:
    TensorAxisWrapper(const std::shared_ptr<T>& tensor_axis) : tensor_axis_(tensor_axis) {};
    std::string getName() const { return tensor_axis_->getName(); };
    size_t getNLabels() const { return tensor_axis_->getNLabels(); };
    size_t getNDimensions() const { return tensor_axis_->getNDimensions(); };
    Eigen::Tensor<std::string, 1>& getDimensions() { return tensor_axis_->getDimensions(); };
    std::shared_ptr<void> getLabelsDataPointer() { return tensor_axis_->getLabelsDataPointer(); }
  };

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
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<void>& select_labels_data, const int& n_labels, const std::shared_ptr<void>& device);
    void resetIndicesView(const std::string& axis_name, const DeviceT& device); ///< copy over the indices values to the indices view
    void zeroIndicesView(const std::string& axis_name, const DeviceT& device); ///< set the indices view to zero

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
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_reshape(select_labels_data.get(), 1, n_labels);
    // broadcast the length of the labels
    auto labels_names_selected_bcast = labels_reshape.broadcast(Eigen::array<int, 2>({ (int)getAxes.at(axis_name)->getNLabels(), 1 }));
    // broadcast the axis labels the size of the labels queried
    // TODO: GPU sync the data pointer
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(static_pointer_cast<LabelsT>(getAxes.at(axis_name)->getLabels()->getDataPointer().get()), (int)getAxes.at(axis_name)->getNDimensions(), (int)getAxes.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = (labels_reshape.chip(dimension_index, 0)).broadcast(Eigen::array<int, 2>({ 1, n_labels }));
    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_.at(axis_name)->getDataPointer()->get(), (int)getAxes.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, n_labels }));
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
    auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 }));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer()->get(), (int)getAxes.at(axis_name)->getNLabels());
    indices_view.device(device) += selected_sum;
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorTable<TensorT, DeviceT, TDim>::selectIndicesView(const std::string & axis_name, const int& dimension_index, const std::shared_ptr<void>& select_labels_data, const int & n_labels, const std::shared_ptr<void>& device)
  {
    typedef decltype(*select_labels_data.get()) LabelsT;
    typedef decltype(*device.get()) DeviceT;
    // reshape to match the axis labels shape
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 2>> labels_reshape(reinterpret_cast<LabelsT*>(select_labels_data.get()), 1, n_labels);
    // broadcast the length of the labels
    auto labels_names_selected_bcast = labels_reshape.broadcast(Eigen::array<int, 2>({ (int)getAxes.at(axis_name)->getNLabels(), 1 }));
    // broadcast the axis labels the size of the labels queried
    // TODO: GPU sync the data pointer
    Eigen::TensorMap<Eigen::Tensor<LabelsT, 3>> labels_reshape(std::static_pointer_cast<LabelsT>(getAxes.at(axis_name)->getLabels()->getDataPointer().get()), (int)getAxes.at(axis_name)->getNDimensions(), (int)getAxes.at(axis_name)->getNLabels(), 1);
    auto labels_bcast = (labels_reshape.chip(dimension_index, 0)).broadcast(Eigen::array<int, 2>({ 1, n_labels }));
    // broadcast the tensor indices the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_.at(axis_name)->getDataPointer()->get(), (int)getAxes.at(axis_name)->getNLabels(), 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, n_labels }));
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
    auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 }));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(indices_view_.at(axis_name)->getDataPointer()->get(), (int)getAxes.at(axis_name)->getNLabels());
    indices_view.device(*reinterpret_cast<LabelsT*>(device.get())) += selected_sum;
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