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
};
#endif //TENSORBASE_TENSORTABLEDEFAULTDEVICE_H