/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLEGPU_H
#define TENSORBASE_TENSORTABLEGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorAxisGpu.h>
#include <TensorBase/ml/TensorAxisConceptGpu.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT, int TDim>
  class TensorTableGpu : public TensorTable<TensorT, Eigen::GpuDevice, TDim>
  {
  public:
    TensorTableGpu() = default;
    TensorTableGpu(const std::string& name) { this->setName(name); };
    ~TensorTableGpu() = default;
    // Initialization methods
    void setAxes() override;
    void initData() override;
    // Select methods
    void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast, const std::string& axis_name, Eigen::GpuDevice& device) override;
    void reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::GpuDevice& device) override;
    void selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, Eigen::GpuDevice& device) override;
    void makeSelectIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice& device) override;
    void getSelectTensorDataFromIndicesView(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice& device) override;
    // Sort methods
    void sliceTensorDataForSort(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& tensor_sort, const std::string& axis_name_sort, const int& label_index_sort, const std::string& axis_name_apply, Eigen::GpuDevice& device) override;
    void makeSortIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_sort, Eigen::GpuDevice& device) override;
    int getFirstIndexFromIndicesView(const std::string& axis_name, Eigen::GpuDevice& device) override;
    // Append to Axis methods
    void makeAppendIndices(const std::string& axis_name, const int& n_labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice& device) override;
    // Delete from Axis methods
    void makeSelectIndicesFromIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice& device) override;
    void getSelectTensorDataFromIndices(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const Eigen::array<Eigen::Index, TDim>& dimensions_select, Eigen::GpuDevice& device) override;
    void makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice& device) override;
    // Update methods
    void makeSparseAxisLabelsFromIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_select, Eigen::GpuDevice& device) override;
    void makeSparseTensorTable(const Eigen::Tensor<std::string, 1>& sparse_dimensions, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_labels, const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& sparse_data, std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 2>>& sparse_table, Eigen::GpuDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorTable<TensorT, Eigen::GpuDevice, TDim>>(this));
    }
  };

  template<typename TensorT, int TDim>
  void TensorTableGpu<TensorT, TDim>::setAxes() {
    assert(TDim == this->axes_.size()); // "The number of tensor_axes and the template TDim do not match.";
    // Clear existing data
    this->dimensions_ = Eigen::array<Eigen::Index, TDim>();
    this->indices_.clear();
    this->indices_view_.clear();
    this->is_modified_.clear();
    this->in_memory_.clear();
    this->shard_id_.clear();
    this->shard_spans_.clear();

    // Determine the overall dimensions of the tensor
    int axis_cnt = 0;
    for (auto& axis : axes_) {
      this->dimensions_.at(axis_cnt) = axis.second->getNLabels();
      Eigen::array<Eigen::Index, 1> axis_dimensions = { (int)axis.second->getNLabels() };

      // Set the axes name to dim map
      this->axes_to_dims_.emplace(axis.second->getName(), axis_cnt);

      // Set the initial shard size
      this->shard_spans_.emplace(axis.second->getName(), axis.second->getNLabels());

      // Set the indices
      Eigen::Tensor<int, 1> indices_values(axis.second->getNLabels());
      for (int i = 0; i < axis.second->getNLabels(); ++i) {
        indices_values(i) = i + 1;
      }
      TensorDataGpu<int, 1> indices(axis_dimensions);
      indices.setData(indices_values);
      this->indices_.emplace(axis.second->getName(), std::make_shared<TensorDataGpu<int, 1>>(indices));

      // Set the indices view
      TensorDataGpu<int, 1> indices_view(axis_dimensions);
      indices_view.setData(indices_values);
      this->indices_view_.emplace(axis.second->getName(), std::make_shared<TensorDataGpu<int, 1>>(indices_view));

      // Set the is_modified defaults
      Eigen::Tensor<int, 1> is_modified_values(axis.second->getNLabels());
      is_modified_values.setZero();
      TensorDataGpu<int, 1> is_modified(axis_dimensions);
      is_modified.setData(is_modified_values);
      this->is_modified_.emplace(axis.second->getName(), std::make_shared<TensorDataGpu<int, 1>>(is_modified));

      // Set the in_memory defaults
      TensorDataGpu<int, 1> in_memory(axis_dimensions);
      in_memory.setData(is_modified_values.constant(1));
      this->in_memory_.emplace(axis.second->getName(), std::make_shared<TensorDataGpu<int, 1>>(in_memory));

      // Set the shard_id defaults
      TensorDataGpu<int, 1> shard_id(axis_dimensions);
      shard_id.setData(is_modified_values);
      this->shard_id_.emplace(axis.second->getName(), std::make_shared<TensorDataGpu<int, 1>>(shard_id));

      // Set the shard_indices defaults
      TensorDataGpu<int, 1> shard_indices(axis_dimensions);
      shard_indices.setData(indices_values);
      this->shard_indices_.emplace(axis.second->getName(), std::make_shared<TensorDataGpu<int, 1>>(shard_indices));

      // Next iteration
      ++axis_cnt;
    }

    // Allocate memory for the tensor
    this->initData();
  };

  template<typename TensorT, int TDim>
  void TensorTableGpu<TensorT, TDim>::initData() {
    this->data_.reset(new TensorDataGpu<TensorT, TDim>(this->getDimensions()));
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast, const std::string & axis_name, Eigen::GpuDevice& device)
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
    TensorDataGpu<int, TDim> indices_view_bcast_tmp(this->dimensions_);
    indices_view_bcast_tmp.setData();
    indices_view_bcast_tmp.syncHAndDData(device);

    // broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(this->indices_view_.at(axis_name)->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.broadcast(indices_bcast_dimensions);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_bcast_map(indices_view_bcast_tmp.getDataPointer().get(), this->dimensions_);
    indices_view_bcast_map.device(device) = indices_view_bcast_values;

    // move over the results
    indices_view_bcast = std::make_shared<TensorDataGpu<int, TDim>>(indices_view_bcast_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast,
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::GpuDevice& device)
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
    TensorDataGpu<TensorT, TDim> tensor_select_tmp(tensor_select_dimensions);
    Eigen::Tensor<TensorT, TDim> tensor_select_data(tensor_select_dimensions);
    tensor_select_data.setZero();
    tensor_select_tmp.setData(tensor_select_data);
    tensor_select_tmp.syncHAndDData(device);

    // move over the results
    tensor_select = std::make_shared<TensorDataGpu<TensorT, TDim>>(tensor_select_tmp);

    // apply the device specific select algorithm
    this->data_->select(tensor_select, indices_view_bcast, device);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& values_select, const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::string & axis_name, const int & n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, Eigen::GpuDevice & device)
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
    TensorDataGpu<int, TDim> indices_select_tmp(tensor_select->getDimensions());
    indices_select_tmp.setData();
    indices_select_tmp.syncHAndDData(device);

    // apply the logical comparitor and modifier as a selection criteria
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_select_values(tensor_select->getDataPointer().get(), tensor_select->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select_tmp.getDataPointer().get(), indices_select_tmp.getDimensions());
    if (comparitor == logicalComparitors::logicalComparitor::NOT_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values != values_bcast).select(indices_select_values.constant(1), indices_select_values.constant(0)));
    }
    else if (comparitor == logicalComparitors::logicalComparitor::EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values == values_bcast).select(indices_select_values.constant(1), indices_select_values.constant(0)));
    }
    else if (comparitor == logicalComparitors::logicalComparitor::LESS_THAN) {
      indices_select_values.device(device) = ((tensor_select_values < values_bcast).select(indices_select_values.constant(1), indices_select_values.constant(0)));
    }
    else if (comparitor == logicalComparitors::logicalComparitor::LESS_THAN_OR_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values <= values_bcast).select(indices_select_values.constant(1), indices_select_values.constant(0)));
    }
    else if (comparitor == logicalComparitors::logicalComparitor::GREATER_THAN) {
      indices_select_values.device(device) = ((tensor_select_values > values_bcast).select(indices_select_values.constant(1), indices_select_values.constant(0)));
    }
    else if (comparitor == logicalComparitors::logicalComparitor::GREATER_THAN_OR_EQUAL_TO) {
      indices_select_values.device(device) = ((tensor_select_values >= values_bcast).select(indices_select_values.constant(1), indices_select_values.constant(0)));
    }
    else {
      std::cout << "The comparitor was not recognized.  No comparison will be performed." << std::endl;
    }

    // move over the results
    indices_select = std::make_shared<TensorDataGpu<int, TDim>>(indices_select_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeSelectIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices
    TensorDataGpu<int, TDim> indices_select_tmp(this->getDimensions());
    Eigen::Tensor<int, TDim> ones(this->getDimensions());
    ones.setConstant(1);
    indices_select_tmp.setData(ones);
    indices_select_tmp.syncHAndDData(device);
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
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(indices_component.at(axis_to_index.first)->getDataPointer().get(), indices_reshape_dimensions);
      auto indices_view_bcast_values = indices_view_reshape.clip(0,1).broadcast(indices_bcast_dimensions);

      // update the indices_select_values
      indices_select_values.device(device) = indices_select_values * indices_view_bcast_values;
    }

    // move over the results
    indices_select = std::make_shared<TensorDataGpu<int, TDim>>(indices_select_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::getSelectTensorDataFromIndicesView(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the sum of each axis
    TensorDataGpu<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    dim_size.setData();
    dim_size.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> dim_size_value(dim_size.getDataPointer().get());

    // determine the new dimensions
    Eigen::array<Eigen::Index, TDim> select_dimensions;
    for (const auto& axis_to_name : this->axes_to_dims_) {
      dim_size.setDataStatus(false, true);

      // calculate the sum
      Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(this->indices_view_.at(axis_to_name.first)->getDataPointer().get(), this->indices_view_.at(axis_to_name.first)->getDimensions());
      dim_size_value.device(device) = indices_view_values.clip(0,1).sum();

      // update the dimensions
      dim_size.syncHAndDData(device);
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      select_dimensions.at(axis_to_name.second) = dim_size.getData()(0);
    }

    // allocate memory for the selected tensor
    TensorDataGpu<TensorT, TDim> tensor_select_tmp(select_dimensions);
    tensor_select_tmp.setData();
    tensor_select = std::make_shared<TensorDataGpu<TensorT, TDim>>(tensor_select_tmp);
    tensor_select->syncHAndDData(device);

    // select the tensor
    this->data_->select(tensor_select, indices_select, device);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::sliceTensorDataForSort(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& tensor_sort,
    const std::string & axis_name_sort, const int & label_index_sort, const std::string & axis_name_apply, Eigen::GpuDevice & device)
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
    TensorDataGpu<TensorT, 1> tensor_sort_tmp(Eigen::array<Eigen::Index, 1>({ (int)this->axes_.at(axis_name_apply)->getNLabels() }));
    tensor_sort_tmp.setData();
    tensor_sort_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_sort_values(tensor_sort_tmp.getDataPointer().get(), tensor_sort_tmp.getDimensions());
    tensor_sort_values.device(device) = tensor_1d;

    // move over the tensor sort data
    tensor_sort = std::make_shared<TensorDataGpu<TensorT, 1>>(tensor_sort_tmp);
  }
  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeSortIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_sort, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices
    TensorDataGpu<int, TDim> indices_sort_tmp(this->getDimensions());
    Eigen::Tensor<int, TDim> zeros(this->getDimensions());
    zeros.setZero();
    indices_sort_tmp.setData(zeros);
    indices_sort_tmp.syncHAndDData(device);
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
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(indices_component.at(axis_to_index.first)->getDataPointer().get(), indices_reshape_dimensions);
      auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(accumulative_size);
      auto indices_view_bcast_values = indices_view_norm.broadcast(indices_bcast_dimensions);

      // update the indices_sort_values
      indices_sort_values.device(device) += indices_view_bcast_values;

      // update the accumulative size
      accumulative_size *= (int)this->axes_.at(axis_to_index.first)->getNLabels();
    }
    indices_sort_values.device(device) += indices_sort_values.constant(1);
    // move over the results
    indices_sort = std::make_shared<TensorDataGpu<int, TDim>>(indices_sort_tmp);
  }
  template<typename TensorT, int TDim>
  inline int TensorTableGpu<TensorT, TDim>::getFirstIndexFromIndicesView(const std::string & axis_name, Eigen::GpuDevice & device)
  {
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    return this->indices_view_.at(axis_name)->getData()(0); // the first occurance of the label
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeAppendIndices(const std::string & axis_name, const int & n_labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice & device)
  {
    // Allocate memory for the extend axis indices
    TensorDataGpu<int, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ n_labels }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);

    // Determine the maximum index value
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_values(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getTensorSize(), 1);
    auto max_bcast = indices_view_values.maximum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 1>({ n_labels }));

    // Make the extended axis indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_tmp.getDataPointer().get(), n_labels);
    auto tmp = indices_values.constant(1).cumsum(0, false);
    indices_values.device(device) = max_bcast + tmp;

    // Move over the indices to the output
    indices = std::make_shared<TensorDataGpu<int, 1>>(indices_tmp);
  }
  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeSelectIndicesFromIndices(const std::string & axis_name, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices
    TensorDataGpu<int, TDim> indices_select_tmp(this->getDimensions());
    indices_select_tmp.setData();
    indices_select_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select_tmp.getDataPointer().get(), indices_select_tmp.getDimensions());

    // Determine the dimensions for reshaping and broadcasting
    Eigen::array<int, TDim> indices_reshape_dimensions;
    Eigen::array<int, TDim> indices_bcast_dimensions;
    for (const auto& axis_to_index : this->axes_to_dims_) {
      if (axis_to_index.first == axis_name) {
        indices_reshape_dimensions.at(axis_to_index.second) = this->dimensions_.at(axis_to_index.second);
        indices_bcast_dimensions.at(axis_to_index.second) = 1;
      }
      else {
        indices_reshape_dimensions.at(axis_to_index.second) = 1;
        indices_bcast_dimensions.at(axis_to_index.second) = this->dimensions_.at(axis_to_index.second);
      }
    }

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(indices->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.clip(0, 1).broadcast(indices_bcast_dimensions);

    // update the indices_select_values
    indices_select_values.device(device) = indices_view_bcast_values;

    // move over the results
    indices_select = std::make_shared<TensorDataGpu<int, TDim>>(indices_select_tmp);
  }

  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::getSelectTensorDataFromIndices(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const Eigen::array<Eigen::Index, TDim>& dimensions_select, Eigen::GpuDevice & device)
  {
    // allocate memory for the selected tensor
    TensorDataGpu<TensorT, TDim> tensor_select_tmp(dimensions_select);
    tensor_select_tmp.setData();
    tensor_select_tmp.syncHAndDData(device);
    tensor_select = std::make_shared<TensorDataGpu<TensorT, TDim>>(tensor_select_tmp);

    // select the tensor
    this->data_->select(tensor_select, indices_select, device);
  }
  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice & device)
  {
    // Normalize the indices view
    auto indices_view_copy = this->indices_view_.at(axis_name)->copy(device);
    indices_view_copy->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_copy_values(indices_view_copy->getDataPointer().get(), indices_view_copy->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getDimensions());
    indices_view_copy_values.device(device) = indices_view_values.clip(0, 1);

    // Determine the size of the indices
    TensorDataGpu<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    dim_size.setData();
    dim_size.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> dim_size_value(dim_size.getDataPointer().get());
    dim_size_value.device(device) = indices_view_copy_values.sum();
    dim_size.syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // Allocate memory for the indices
    TensorDataGpu<int, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ dim_size.getData()(0) }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataGpu<int, 1>>(indices_tmp);

    // Select out the non zero indices
    this->indices_view_.at(axis_name)->select(indices, indices_view_copy, device);
  }
  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeSparseAxisLabelsFromIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the total sum of each axis
    TensorDataGpu<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    dim_size.setData();
    dim_size.getData()(0) = 0;

    // Determine the total number of labels and create the selected `indices_view`
    std::vector<std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>> indices_selected_vec;
    int labels_size = 1;
    for (const auto& axis_to_name : this->axes_to_dims_) {
      // calculate the sum
      dim_size.syncHAndDData(device); // h to d
      Eigen::TensorMap<Eigen::Tensor<int, 0>> dim_size_value(dim_size.getDataPointer().get());
      Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(this->indices_view_.at(axis_to_name.first)->getDataPointer().get(), this->indices_view_.at(axis_to_name.first)->getDimensions());
      dim_size_value.device(device) = indices_view_values.clip(0, 1).sum();

      // update the dimensions
      dim_size.syncHAndDData(device); // d to h
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      labels_size *= dim_size.getData()(0);

      // create the selection for the indices view
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_select = this->indices_view_.at(axis_to_name.first)->copy(device);
      indices_select->syncHAndDData(device);
      Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_select_values(indices_select->getDataPointer().get(), indices_select->getDimensions());
      indices_select_values.device(device) = indices_view_values.clip(0, 1);

      // select out the indices
      TensorDataGpu<int, 1> indices_selected(Eigen::array<Eigen::Index, 1>({ dim_size.getData()(0) }));
      indices_selected.setData();
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_selected_ptr = std::make_shared<TensorDataGpu<int, 1>>(indices_selected);
      indices_selected_ptr->syncHAndDData(device);
      this->indices_view_.at(axis_to_name.first)->select(indices_selected_ptr, indices_select, device);
      indices_selected_vec.push_back(indices_selected_ptr);
    }

    // allocate memory for the labels
    TensorDataGpu<int, 2> sparse_labels(Eigen::array<Eigen::Index, 2>({ (int)this->axes_to_dims_.size(), labels_size }));
    sparse_labels.setData();
    sparse_labels.syncHAndDData(device);

    // iterate through each of the axes and assign the labels
    for (int i = 0; i < TDim; ++i) {
      // determine the padding and repeats for each dimension
      int n_padding = 1;
      for (int j = 0; j < i; ++j) {
        n_padding *= indices_selected_vec.at(j)->getTensorSize();
      }
      int slice_size = n_padding * indices_selected_vec.at(i)->getTensorSize();
      int n_repeats = labels_size / slice_size;

      // create the repeating "slice"
      Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_selected_values(indices_selected_vec.at(i)->getDataPointer().get(), 1, indices_selected_vec.at(i)->getTensorSize());
      auto indices_bcast = indices_selected_values.broadcast(Eigen::array<Eigen::Index, 2>({ n_padding, 1 })).reshape(Eigen::array<Eigen::Index, 2>({ 1, slice_size }));

      // repeatedly assign the slice
      Eigen::TensorMap<Eigen::Tensor<int, 2>> sparse_labels_values(sparse_labels.getDataPointer().get(), sparse_labels.getDimensions());
      for (int j = 0; j < n_repeats; ++j) {
        Eigen::array<int, 2> offsets = { i, j * slice_size };
        Eigen::array<int, 2> extents = { 1, slice_size };
        sparse_labels_values.slice(offsets, extents).device(device) = indices_bcast;
      }
    }

    // move over the output
    sparse_select = std::make_shared<TensorDataGpu<int, 2>>(sparse_labels);
  }
  template<typename TensorT, int TDim>
  inline void TensorTableGpu<TensorT, TDim>::makeSparseTensorTable(const Eigen::Tensor<std::string, 1>& sparse_dimensions, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_labels, const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& sparse_data, std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 2>>& sparse_table, Eigen::GpuDevice & device)
  {
    sparse_labels->syncHAndDData(device); // d to h
    sparse_data->syncHAndDData(device); // d to h
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // make the sparse axis
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("Indices", sparse_dimensions, sparse_labels->getData()));

    // make the values axis
    Eigen::Tensor<std::string, 1> values_dimension(1);
    values_dimension.setValues({ "Values" });
    Eigen::Tensor<int, 2> values_labels(1, 1);
    values_labels.setZero();
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("Values", values_dimension, values_labels));

    // add the axes to the tensorTable
    TensorTableGpu<TensorT, 2> tensorTable;
    tensorTable.addTensorAxis(axis_1_ptr);
    tensorTable.addTensorAxis(axis_2_ptr);
    tensorTable.setAxes();

    // set the data
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sparse_data_values(sparse_data->getData().data(), sparse_data->getTensorSize(), 1);
    tensorTable.setData(sparse_data_values);

    // sync the data
    tensorTable.syncIndicesHAndDData(device);
    tensorTable.syncIndicesViewHAndDData(device);
    tensorTable.syncInMemoryHAndDData(device);
    tensorTable.syncIsModifiedHAndDData(device);
    tensorTable.syncShardIdHAndDData(device);
    tensorTable.syncShardIndicesHAndDData(device);
    tensorTable.syncAxesHAndDData(device);
    tensorTable.syncHAndDData(device);

    // move over the table
    sparse_table = std::make_shared<TensorTableGpu<TensorT, 2>>(tensorTable);
  }
};

// Cereal registration of TensorTs: float, int, char, double and TDims: 1, 2, 3, 4
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<int, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<float, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<double, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<int, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<float, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<double, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<int, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<float, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<double, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<int, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<float, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<double, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpu<char, 4>);
#endif
#endif //TENSORBASE_TENSORTABLEGPU_H