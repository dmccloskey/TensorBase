/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLEDEFAULTDEVICE_H
#define TENSORBASE_TENSORTABLEDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>

namespace TensorBase
{
  template<typename TensorT, int TDim>
  class TensorTableDefaultDevice: public TensorTable<TensorT, Eigen::DefaultDevice, TDim>
  {
  public:
    TensorTableDefaultDevice() = default;
    TensorTableDefaultDevice(const std::string& name) { this->setName(name); };
    ~TensorTableDefaultDevice() = default;
    // Initialization methods
    void setAxes() override;
    void initData() override;
    // Select methods
    void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, const std::string& axis_name, Eigen::DefaultDevice& device) override;
    void reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::DefaultDevice& device) override;
    void selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, Eigen::DefaultDevice& device) override;
    void makeSelectIndicesFromIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, Eigen::DefaultDevice& device) override;
    void getSelectTensorData(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, Eigen::DefaultDevice& device) override;
    // Sort methods
    void sliceTensorDataForSort(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& tensor_sort, const std::string& axis_name_sort, const int& label_index_sort, const std::string& axis_name_apply, Eigen::DefaultDevice& device) override;
    void makeSortIndicesViewFromIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_sort, Eigen::DefaultDevice& device) override;
    int getFirstIndexFromIndicesView(const std::string& axis_name, Eigen::DefaultDevice& device) override;
  };

  template<typename TensorT, int TDim>
  void TensorTableDefaultDevice<TensorT, TDim>::setAxes() {
    assert(TDim == this->axes_.size()); // "The number of tensor_axes and the template TDim do not match.";
    // Clear existing data
    dimensions_ = Eigen::array<Eigen::Index, TDim>();
    indices_.clear();
    indices_view_.clear();
    is_modified_.clear();
    in_memory_.clear();
    is_shardable_.clear();

    // Determine the overall dimensions of the tensor
    int axis_cnt = 0;
    for (auto& axis : axes_) {
      this->dimensions_.at(axis_cnt) = axis.second->getNLabels();
      Eigen::array<Eigen::Index, 1> axis_dimensions = { (int)axis.second->getNLabels() };

      // Set the axes name to dim map
      this->axes_to_dims_.emplace(axis.second->getName(), axis_cnt);

      // Set the indices
      Eigen::Tensor<int, 1> indices_values(axis.second->getNLabels());
      for (int i = 0; i < axis.second->getNLabels(); ++i) {
        indices_values(i) = i + 1;
      }
      TensorDataDefaultDevice<int, 1> indices(axis_dimensions);
      indices.setData(indices_values);
      this->indices_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(indices));

      // Set the indices view
      TensorDataDefaultDevice<int, 1> indices_view(axis_dimensions);
      indices_view.setData(indices_values);
      this->indices_view_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_view));

      // Set the is_modified defaults
      Eigen::Tensor<int, 1> is_modified_values(axis.second->getNLabels());
      is_modified_values.setZero();
      TensorDataDefaultDevice<int, 1> is_modified(axis_dimensions);
      is_modified.setData(is_modified_values);
      this->is_modified_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(is_modified));

      // Set the in_memory defaults
      Eigen::Tensor<int, 1> in_memory_values(axis.second->getNLabels());
      in_memory_values.setZero();
      TensorDataDefaultDevice<int, 1> in_memory(axis_dimensions);
      in_memory.setData(in_memory_values);
      this->in_memory_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(in_memory));

      // Set the in_memory defaults
      Eigen::Tensor<int, 1> is_shardable_values(axis.second->getNLabels());
      if (axis_cnt == 0)
        is_shardable_values.setConstant(1);
      else
        is_shardable_values.setZero();
      TensorDataDefaultDevice<int, 1> is_shardable(axis_dimensions);
      is_shardable.setData(is_shardable_values);
      this->is_shardable_.emplace(axis.second->getName(), std::make_shared<TensorDataDefaultDevice<int, 1>>(is_shardable));

      // Next iteration
      ++axis_cnt;
    }

    // Allocate memory for the tensor
    this->initData();
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
      if (i == this->axes_to_dims_.at(axis_name)) {
        indices_reshape_dimensions.at(i) = (int)this->axes_.at(axis_name)->getNLabels();
        indices_bcast_dimensions.at(i) = 1;
      }
      else {
        indices_reshape_dimensions.at(i) = 1;
        indices_bcast_dimensions.at(i) = this->dimensions_.at(i);
      }
    }

    // allocate to memory
    TensorDataDefaultDevice<int, TDim> indices_view_bcast_tmp(this->dimensions_);
    indices_view_bcast_tmp.setData();

    // broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(this->indices_view_.at(axis_name)->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.broadcast(indices_bcast_dimensions);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_bcast_map(indices_view_bcast_tmp.getDataPointer().get(), this->dimensions_);
    indices_view_bcast_map.device(device) = indices_view_bcast_values;
    
    // move over the results
    indices_view_bcast = std::make_shared<TensorDataDefaultDevice<int, TDim>>(indices_view_bcast_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, 
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::DefaultDevice& device)
  {
    // determine the dimensions for making the selected tensor
    Eigen::array<Eigen::Index, TDim> tensor_select_dimensions;
    for (int i = 0; i < TDim; ++i) {
      if (i == axes_to_dims_.at(axis_name)) {
        tensor_select_dimensions.at(i) = n_select;
      }
      else {
        tensor_select_dimensions.at(i) = dimensions_.at(i);
      }
    }

    // allocate memory for the selected tensor
    TensorDataDefaultDevice<TensorT, TDim> tensor_select_tmp(tensor_select_dimensions);
    Eigen::Tensor<TensorT, TDim> tensor_select_data(tensor_select_dimensions);
    tensor_select_data.setZero();
    tensor_select_tmp.setData(tensor_select_data);

    // move over the results
    tensor_select = std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(tensor_select_tmp);

    // apply the device specific select algorithm
    this->data_->select(tensor_select, indices_view_bcast, device);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string & axis_name, const int & n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, Eigen::DefaultDevice & device)
  {
    // determine the dimensions for reshaping and broadcasting the values
    Eigen::array<int, TDim> values_reshape_dimensions;
    Eigen::array<int, TDim> values_bcast_dimensions;
    for (int i = 0; i < TDim; ++i) {
      if (i == axes_to_dims_.at(axis_name)) {
        values_reshape_dimensions.at(i) = n_select;
        values_bcast_dimensions.at(i) = 1;
      }
      else {
        values_reshape_dimensions.at(i) = 1;
        values_bcast_dimensions.at(i) = dimensions_.at(i);
      }
    }

    // broadcast the comparitor values across the selected tensor dimensions
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> values_reshape(values_select->getDataPointer().get(), values_reshape_dimensions);
    auto values_bcast = values_reshape.broadcast(values_bcast_dimensions);

    // allocate memory for the indices
    TensorDataDefaultDevice<int, TDim> indices_select_tmp(tensor_select->getDimensions());
    indices_select_tmp.setData();

    // apply the logical comparitor and modifier as a selection criteria
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_select_values(tensor_select->getDataPointer().get(), tensor_select->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select_tmp.getDataPointer().get(), indices_select_tmp.getDimensions());
    if (comparitor == logicalComparitors::logicalComparitor::NOT_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values != values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitors::logicalComparitor::EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values == values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitors::logicalComparitor::LESS_THAN) {
      indices_select_values.device(device) = ((tensor_select_values < values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitors::logicalComparitor::LESS_THAN_OR_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values <= values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitors::logicalComparitor::GREATER_THAN) {
      indices_select_values.device(device) = ((tensor_select_values > values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitors::logicalComparitor::GREATER_THAN_OR_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values >= values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else {
      std::cout << "The comparitor was not recognized.  No comparison will be performed." << std::endl;
    }

    // move over the results
    indices_select = std::make_shared<TensorDataDefaultDevice<int, TDim>>(indices_select_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::makeSelectIndicesFromIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, Eigen::DefaultDevice & device)
  {
    // allocate memory for the indices
    TensorDataDefaultDevice<int, TDim> indices_select_tmp(this->getDimensions());
    Eigen::Tensor<int, TDim> ones(this->getDimensions());
    ones.setConstant(1);
    indices_select_tmp.setData(ones);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select_tmp.getDataPointer().get(), indices_select_tmp.getDimensions());

    // [PERFORMANCE: Can this be replaced with contractions?]
    for (const auto& axis_to_index : this->axes_to_dims_) {
      // determine the dimensions for reshaping and broadcasting the indices
      Eigen::array<int, TDim> indices_reshape_dimensions;
      Eigen::array<int, TDim> indices_bcast_dimensions;
      for (int i = 0; i < TDim; ++i) {
        if (i == this->axes_to_dims_.at(axis_to_index.first)) {
          indices_reshape_dimensions.at(i) = (int)this->axes_.at(axis_to_index.first)->getNLabels();
          indices_bcast_dimensions.at(i) = 1;
        }
        else {
          indices_reshape_dimensions.at(i) = 1;
          indices_bcast_dimensions.at(i) = this->dimensions_.at(i);
        }
      }
      
      // normalize and broadcast the indices across the tensor
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(this->indices_view_.at(axis_to_index.first)->getDataPointer().get(), indices_reshape_dimensions);
      auto indices_view_norm = (indices_view_reshape.cast<float>() / (indices_view_reshape.cast<float>() + indices_view_reshape.cast<float>().constant(1e-12))).cast<int>();
      auto indices_view_bcast_values = indices_view_norm.broadcast(indices_bcast_dimensions);

      // update the indices_select_values
      indices_select_values.device(device) = indices_select_values * indices_view_bcast_values;
    }

    // move over the results
    indices_select = std::make_shared<TensorDataDefaultDevice<int, TDim>>(indices_select_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::getSelectTensorData(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, Eigen::DefaultDevice & device)
  {
    // temporary memory for calculating the sum of each axis
    TensorDataDefaultDevice<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    dim_size.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 0>> dim_size_value(dim_size.getDataPointer().get());

    // determine the new dimensions
    Eigen::array<Eigen::Index, TDim> select_dimensions;
    for (const auto& axis_to_name : this->axes_to_dims_) {
      // calculate the sum
      Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(this->indices_view_.at(axis_to_name.first)->getDataPointer().get(), this->indices_view_.at(axis_to_name.first)->getDimensions());
      dim_size_value.device(device) = indices_view_values.clip(0,1).sum();

      // update the dimensions
      select_dimensions.at(axis_to_name.second) = dim_size.getData()(0);
    }

    // allocate memory for the selected tensor
    TensorDataDefaultDevice<TensorT, TDim> tensor_select_tmp(select_dimensions);
    tensor_select_tmp.setData();
    tensor_select = std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(tensor_select_tmp);

    // select the tensor
    this->data_->select(tensor_select, indices_select, device);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::sliceTensorDataForSort(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& tensor_sort, 
    const std::string & axis_name_sort, const int & label_index_sort, const std::string & axis_name_apply, Eigen::DefaultDevice & device)
  {
    // determine the offsets and extents for the slice operation
    Eigen::array<int, TDim> extents;
    Eigen::array<int, TDim> offsets;
    for (const auto& axis_to_name_slice : this->axes_to_dims_) {
      if (axis_to_name_slice.first == axis_name_sort) {
        extents.at(axis_to_name_slice.second) = 1;
        offsets.at(axis_to_name_slice.second) = label_index_sort;
      }
      else if (axis_to_name_slice.first == axis_name_apply) {
        extents.at(axis_to_name_slice.second) = this->axes_.at(axis_to_name_slice.first)->getNLabels();
        offsets.at(axis_to_name_slice.second) = 0;
      }
      else {
        extents.at(axis_to_name_slice.second) = 1;
        offsets.at(axis_to_name_slice.second) = 0;
      }
    }

    // slice out the 1D tensor
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_values(this->data_->getDataPointer().get(), this->data_->getDimensions());
    auto tensor_1d = tensor_values.slice(offsets, extents).reshape(Eigen::array<Eigen::Index, 1>({ (int)this->axes_.at(axis_name_apply)->getNLabels() }));

    // allocate memory for the slice
    TensorDataDefaultDevice<TensorT, 1> tensor_sort_tmp(Eigen::array<Eigen::Index, 1>({ (int)this->axes_.at(axis_name_apply)->getNLabels() }));
    tensor_sort_tmp.setData();
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_sort_values(tensor_sort_tmp.getDataPointer().get(), tensor_sort_tmp.getDimensions());
    tensor_sort_values.device(device) = tensor_1d;

    // move over the tensor sort data
    tensor_sort = std::make_shared<TensorDataDefaultDevice<TensorT, 1>>(tensor_sort_tmp);
  }
  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::makeSortIndicesViewFromIndicesView(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_sort, Eigen::DefaultDevice & device)
  {
    // allocate memory for the indices
    TensorDataDefaultDevice<int, TDim> indices_sort_tmp(this->getDimensions());
    Eigen::Tensor<int, TDim> zeros(this->getDimensions());
    zeros.setZero();
    indices_sort_tmp.setData(zeros);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());

    // [PERFORMANCE: Can this be replaced with contractions?]
    int accumulative_size = 1;
    for (const auto& axis_to_index : this->axes_to_dims_) {
      // determine the dimensions for reshaping and broadcasting the indices
      Eigen::array<int, TDim> indices_reshape_dimensions;
      Eigen::array<int, TDim> indices_bcast_dimensions;
      for (int i = 0; i < TDim; ++i) {
        if (i == this->axes_to_dims_.at(axis_to_index.first)) {
          indices_reshape_dimensions.at(i) = (int)this->axes_.at(axis_to_index.first)->getNLabels();
          indices_bcast_dimensions.at(i) = 1;
        }
        else {
          indices_reshape_dimensions.at(i) = 1;
          indices_bcast_dimensions.at(i) = this->dimensions_.at(i);
        }
      }

      // normalize and broadcast the indices across the tensor
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(this->indices_view_.at(axis_to_index.first)->getDataPointer().get(), indices_reshape_dimensions);
      auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(accumulative_size);
      auto indices_view_bcast_values = indices_view_norm.broadcast(indices_bcast_dimensions);

      // update the indices_sort_values
      indices_sort_values.device(device) += indices_view_bcast_values;

      // update the accumulative size
      accumulative_size *= (int)this->axes_.at(axis_to_index.first)->getNLabels();
    }
    indices_sort_values.device(device) += indices_sort_values.constant(1);
    // move over the results
    indices_sort = std::make_shared<TensorDataDefaultDevice<int, TDim>>(indices_sort_tmp);
  }
  template<typename TensorT, int TDim>
  inline int TensorTableDefaultDevice<TensorT, TDim>::getFirstIndexFromIndicesView(const std::string& axis_name, Eigen::DefaultDevice & device)
  {
    return this->indices_view_.at(axis_name)->getData()(0); // the first occurance of the label
  }
};
#endif //TENSORBASE_TENSORTABLEDEFAULTDEVICE_H