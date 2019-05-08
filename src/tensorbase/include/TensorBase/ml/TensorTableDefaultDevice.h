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
    void extractTensorData(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::DefaultDevice& device) override;
    void selectTensorIndices(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, const logicalComparitor& comparitor, const logicalModifier& modifier, Eigen::DefaultDevice& device) override;
    void applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, const std::string & axis_name_select, const std::string& axis_name, const logicalContinuator& within_continuator, const logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) override;
  };

  template<typename TensorT, int TDim>
  void TensorTableDefaultDevice<TensorT, TDim>::setAxes() {
    assert(TDim == this->axes_.size()); // "The number of tensor_axes and the template TDim do not match.";

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
    TensorDataDefaultDevice<int, 3> indices_view_bcast_tmp(this->dimensions_);
    indices_view_bcast_tmp.setData();

    // broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(this->indices_view_.at(axis_name)->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.broadcast(indices_bcast_dimensions);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_bcast_map(indices_view_bcast_tmp.getDataPointer().get(), this->dimensions_);
    indices_view_bcast_map.device(device) = indices_view_bcast_values;
    
    // move over the results
    indices_view_bcast = std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_view_bcast_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::extractTensorData(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_view_bcast, 
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
  inline void TensorTableDefaultDevice<TensorT, TDim>::selectTensorIndices(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::string & axis_name, const int & n_select, const logicalComparitor& comparitor, const logicalModifier& modifier, Eigen::DefaultDevice & device)
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
    if (comparitor == logicalComparitor::NOT_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values != values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitor::EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values == values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitor::LESS_THAN) {
      indices_select_values.device(device) = ((tensor_select_values < values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitor::LESS_THAN_OR_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values <= values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitor::GREATER_THAN) {
      indices_select_values.device(device) = ((tensor_select_values > values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else if (comparitor == logicalComparitor::GREATER_THAN_OR_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values >= values_bcast).select(tensor_select_values.constant(1), tensor_select_values.constant(0))).cast<int>();
    }
    else {
      std::cout << "The comparitor was not recognized.  No comparison will be performed." << std::endl;
    }

    // move over the results
    indices_select = std::make_shared<TensorDataDefaultDevice<int, TDim>>(indices_select_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableDefaultDevice<TensorT, TDim>::applyIndicesSelectToIndicesView(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices_select, const std::string & axis_name_select, const std::string & axis_name, const logicalContinuator & within_continuator, const logicalContinuator & prepend_continuator, Eigen::DefaultDevice & device)
  {
    // build the continuator reduction indices
    Eigen::array<int, TDim - 1> reduction_dims;
    int index = 0;
    for (const auto& axis_to_name_red : this->axes_to_dims_) {
      if (axis_to_name_red.first != axis_name_select) {
        reduction_dims.at(index) = axis_to_name_red.second;
        ++index;
      }
    }

    // apply the continuator reduction, then...
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());
    if (within_continuator == logicalContinuator::OR) {
      auto indices_view_update_tmp = indices_select_values.sum(reduction_dims);
      //ensure a max value of 1 (Note: + 1e-12 is to prevent division by 0; the cast back to "int" rounds down to 0)
      auto indices_view_update = (indices_view_update_tmp.cast<float>() / (indices_view_update_tmp.cast<float>() + indices_view_update_tmp.cast<float>().constant(1e-12) )).cast<int>();

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
    else if (within_continuator == logicalContinuator::AND) {
      auto indices_view_update = indices_select_values.prod(reduction_dims);

      // update the indices view based on the prepend_continuator
      if (prepend_continuator == logicalContinuator::OR) {
        indices_view.device(device) = (indices_view_update > indices_view_update.constant(0) || indices_view > indices_view.constant(0)).select(indices_view, indices_view.constant(0));
      }
      else if (prepend_continuator == logicalContinuator::AND) {
        indices_view.device(device) = indices_view * indices_view_update;
      }
    }
  }
};
#endif //TENSORBASE_TENSORTABLEDEFAULTDEVICE_H