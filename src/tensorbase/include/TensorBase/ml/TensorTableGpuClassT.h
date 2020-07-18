/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLEGPUCLASST_H
#define TENSORBASE_TENSORTABLEGPUCLASST_H

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
  template<template<class> class ArrayT, class TensorT, int TDim>
  class TensorTableGpuClassT : public TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, TDim>
  {
  public:
    TensorTableGpuClassT() = default;
    TensorTableGpuClassT(const std::string& name) : TensorTable(name) {};
    TensorTableGpuClassT(const std::string& name, const std::string& dir) : TensorTable(name, dir) {};
    ~TensorTableGpuClassT() = default;
    // Initialization methods
    void setAxes(Eigen::GpuDevice& device) override;
    void initData(Eigen::GpuDevice& device) override;
    void initData(const Eigen::array<Eigen::Index, TDim>& new_dimensions, Eigen::GpuDevice& device) override;
    std::shared_ptr<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, TDim>> copy(Eigen::GpuDevice& device) override;
    // Select methods
    void broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast, const std::string& axis_name, Eigen::GpuDevice& device) override;
    void reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::GpuDevice& device) override;
    void selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& values_select, const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, Eigen::GpuDevice& device) override;
    void makeSelectIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice& device) const override;
    void getSelectTensorDataFromIndicesView(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice& device) override;
    // Sort methods
    void sliceTensorDataForSort(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& tensor_sort, const std::string& axis_name_sort, const int& label_index_sort, const std::string& axis_name_apply, Eigen::GpuDevice& device) override;
    void makeSortIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_sort, Eigen::GpuDevice& device) const override;
    int getFirstIndexFromIndicesView(const std::string& axis_name, Eigen::GpuDevice& device) override;
    // Append to Axis methods
    void makeAppendIndices(const std::string& axis_name, const int& n_labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice& device) override;
    // Delete from Axis methods
    void makeSelectIndicesFromIndices(const std::string& axis_name, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice& device) override;
    void getSelectTensorDataFromIndices(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const Eigen::array<Eigen::Index, TDim>& dimensions_select, Eigen::GpuDevice& device) override;
    void makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice& device) override;
    // Update methods
    void makeSparseAxisLabelsFromIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_select, Eigen::GpuDevice& device) override;
    void makeSparseTensorTable(const Eigen::Tensor<std::string, 1>& sparse_dimensions, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_labels, const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& sparse_data, std::shared_ptr<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& sparse_table, Eigen::GpuDevice& device) override;
    // IO methods
    void makeShardIndicesFromShardIDs(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_shard, Eigen::GpuDevice& device) const override;
    void runLengthEncodeIndex(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& data, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& n_runs, Eigen::GpuDevice& device) const override;
    int makeSliceIndicesFromShardIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& modified_shard_ids, std::map<int, std::pair<Eigen::array<Eigen::Index, TDim>, Eigen::array<Eigen::Index, TDim>>>& slice_indices, Eigen::array<Eigen::Index, TDim>& shard_data_dimensions, Eigen::GpuDevice& device) const override;
    void makeShardIDTensor(std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& modified_shard_ids, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& num_runs, Eigen::GpuDevice & device) const override;
    bool loadTensorTableBinary(const std::string& dir, Eigen::GpuDevice& device) override;
    bool storeTensorTableBinary(const std::string& dir, Eigen::GpuDevice& device) override;
    // CSV methods
    void makeSparseTensorTableFromCsv(std::shared_ptr<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& sparse_table_ptr, const Eigen::Tensor<std::string, 2>& data_new, Eigen::GpuDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>(this));
    }
  };

  template<template<class> class ArrayT, class TensorT, int TDim>
  void TensorTableGpuClassT<ArrayT, TensorT, TDim>::setAxes(Eigen::GpuDevice& device) {
    assert(TDim == this->axes_.size()); // "The number of tensor_axes and the template TDim do not match.";
    // Clear existing data
    this->dimensions_ = Eigen::array<Eigen::Index, TDim>();
    this->indices_.clear();
    this->indices_view_.clear();
    this->is_modified_.clear();
    this->not_in_memory_.clear();
    this->shard_id_.clear();
    this->shard_indices_.clear();
    bool update_shard_spans = false;
    if (this->shard_spans_.size() == 0) {
      this->shard_spans_.clear();
      update_shard_spans = true;
    }

    // Determine the overall dimensions of the tensor
    int axis_cnt = 0;
    Eigen::array<Eigen::Index, TDim> dimensions;
    for (auto& axis : axes_) {
      dimensions.at(axis_cnt) = axis.second->getNLabels();
      Eigen::array<Eigen::Index, 1> axis_dimensions = { (int)axis.second->getNLabels() };

      // Set the axes name to dim map
      this->axes_to_dims_.emplace(axis.second->getName(), axis_cnt);

      // Set the initial shard size
      if (update_shard_spans) this->shard_spans_.emplace(axis.second->getName(), axis.second->getNLabels());

      // Set the indices
      Eigen::Tensor<int, 1> indices_values(axis.second->getNLabels());
      for (int i = 0; i < axis.second->getNLabels(); ++i) {
        indices_values(i) = i + 1;
      }
      TensorDataGpuPrimitiveT<int, 1> indices(axis_dimensions);
      indices.setData(indices_values);
      this->indices_.emplace(axis.second->getName(), std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices));

      // Set the indices view
      TensorDataGpuPrimitiveT<int, 1> indices_view(axis_dimensions);
      indices_view.setData(indices_values);
      this->indices_view_.emplace(axis.second->getName(), std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_view));

      // Set the is_modified defaults
      Eigen::Tensor<int, 1> is_modified_values(axis.second->getNLabels());
      is_modified_values.setZero();
      TensorDataGpuPrimitiveT<int, 1> is_modified(axis_dimensions);
      is_modified.setData(is_modified_values);
      this->is_modified_.emplace(axis.second->getName(), std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(is_modified));

      // Set the in_memory defaults
      TensorDataGpuPrimitiveT<int, 1> in_memory(axis_dimensions);
      in_memory.setData(is_modified_values.constant(1));
      this->not_in_memory_.emplace(axis.second->getName(), std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(in_memory));

      // Set the shard_id defaults
      TensorDataGpuPrimitiveT<int, 1> shard_id(axis_dimensions);
      shard_id.setData();
      this->shard_id_.emplace(axis.second->getName(), std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(shard_id));

      // Set the shard_indices defaults
      TensorDataGpuPrimitiveT<int, 1> shard_indices(axis_dimensions);
      shard_indices.setData();
      this->shard_indices_.emplace(axis.second->getName(), std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(shard_indices));

      // Next iteration
      ++axis_cnt;
    }
    // Set the dimensions and tensor size
    this->setDimensions(dimensions);

    // Set the shard_id and shard_indices
    this->reShardIndices();

    // Allocate memory for the tensor
    this->initData(this->getDimensions(), device);
  };

  template<template<class> class ArrayT, class TensorT, int TDim>
  void TensorTableGpuClassT<ArrayT, TensorT, TDim>::initData(Eigen::GpuDevice& device) {
    Eigen::array<Eigen::Index, TDim> zero_dimensions;
    for (int i = 0; i < TDim; ++i) zero_dimensions.at(i) = 0;
    this->data_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, TDim>>(TensorDataGpuClassT<ArrayT, TensorT, TDim>(zero_dimensions));
    // update the not_in_memory
    for (const auto& axis_to_dim : this->axes_to_dims_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(this->not_in_memory_.at(axis_to_dim.first)->getDataPointer().get(), (int)this->not_in_memory_.at(axis_to_dim.first)->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(1);
    }
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  void TensorTableGpuClassT<ArrayT, TensorT, TDim>::initData(const Eigen::array<Eigen::Index, TDim>& new_dimensions, Eigen::GpuDevice& device) {
    this->data_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, TDim>>(TensorDataGpuClassT<ArrayT, TensorT, TDim>(new_dimensions));
    // update the not_in_memory
    for (const auto& axis_to_dim : this->axes_to_dims_) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory(this->not_in_memory_.at(axis_to_dim.first)->getDataPointer().get(), (int)this->not_in_memory_.at(axis_to_dim.first)->getTensorSize());
      not_in_memory.device(device) = not_in_memory.constant(1);
    }
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline std::shared_ptr<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, TDim>> TensorTableGpuClassT<ArrayT, TensorT, TDim>::copy(Eigen::GpuDevice& device)
  {
    TensorTableGpuClassT<ArrayT, TensorT, TDim> tensor_table_copy;
    // copy the metadata
    tensor_table_copy.setId(this->getId());
    tensor_table_copy.setName(this->getName());
    tensor_table_copy.setDir(this->getDir());
    tensor_table_copy.axes_to_dims_ = this->getAxesToDims();
    tensor_table_copy.dimensions_ = this->getDimensions();
    tensor_table_copy.setShardSpans(this->getShardSpans());
    tensor_table_copy.setMaximumDimensions(this->getMaximumDimensions());

    // copy the axes and indices
    for (auto& axis_to_dim : this->getAxesToDims()) {
      tensor_table_copy.getAxes().emplace(axis_to_dim.first, this->getAxes().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getIndices().emplace(axis_to_dim.first, this->getIndices().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getIndicesView().emplace(axis_to_dim.first, this->getIndicesView().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getIsModified().emplace(axis_to_dim.first, this->getIsModified().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getNotInMemory().emplace(axis_to_dim.first, this->getNotInMemory().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getShardId().emplace(axis_to_dim.first, this->getShardId().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getShardIndices().emplace(axis_to_dim.first, this->getShardIndices().at(axis_to_dim.first)->copy(device));
      tensor_table_copy.getDimensions().at(axis_to_dim.second) = this->getDimensions().at(axis_to_dim.second);
    }

    // copy the data
    tensor_table_copy.setData(data_->copy(device));
    return std::make_shared<TensorTableGpuClassT<ArrayT, TensorT, TDim>>(tensor_table_copy);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::broadcastSelectIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast, const std::string & axis_name, Eigen::GpuDevice& device)
  {
    // determine the dimensions for reshaping and broadcasting the indices
    Eigen::array<Eigen::Index, TDim> indices_reshape_dimensions;
    Eigen::array<Eigen::Index, TDim> indices_bcast_dimensions;
    for (int i = 0; i < TDim; ++i) {
      if (i == this->axes_to_dims_.at(axis_name)) {
        indices_reshape_dimensions.at(i) = (int)this->axes_.at(axis_name)->getNLabels();
        indices_bcast_dimensions.at(i) = 1;
      }
      else {
        indices_reshape_dimensions.at(i) = 1;
        indices_bcast_dimensions.at(i) = this->getDimensions().at(i);
      }
    }

    // allocate to memory
    TensorDataGpuPrimitiveT<int, TDim> indices_view_bcast_tmp(this->getDimensions());
    indices_view_bcast_tmp.setData();
    indices_view_bcast_tmp.syncHAndDData(device);

    // broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(this->indices_view_.at(axis_name)->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.broadcast(indices_bcast_dimensions);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_bcast_map(indices_view_bcast_tmp.getDataPointer().get(), this->getDimensions());
    indices_view_bcast_map.device(device) = indices_view_bcast_values;

    // move over the results
    indices_view_bcast = std::make_shared<TensorDataGpuPrimitiveT<int, TDim>>(indices_view_bcast_tmp);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::reduceTensorDataToSelectIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_view_bcast,
    std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::string& axis_name, const int& n_select, Eigen::GpuDevice& device)
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
    TensorDataGpuClassT<ArrayT, TensorT, TDim> tensor_select_tmp(tensor_select_dimensions);
    tensor_select_tmp.setData();
    tensor_select_tmp.syncHAndDData(device);

    // move over the results
    tensor_select = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, TDim>>(tensor_select_tmp);

    // apply the device specific select algorithm
    assert(this->getDataTensorSize() == 0 || this->getDataDimensions() == indices_view_bcast->getDimensions());
    this->data_->select(tensor_select, indices_view_bcast, device);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::selectTensorIndicesOnReducedTensorData(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& values_select, const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::string & axis_name, const int & n_select, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier, Eigen::GpuDevice & device)
  {
    // determine the dimensions for reshaping and broadcasting the values
    Eigen::array<Eigen::Index, TDim> values_reshape_dimensions;
    Eigen::array<Eigen::Index, TDim> values_bcast_dimensions;
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
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, TDim>> values_reshape(values_select->getDataPointer().get(), values_reshape_dimensions);
    auto values_bcast = values_reshape.broadcast(values_bcast_dimensions);

    // allocate memory for the indices
    TensorDataGpuPrimitiveT<int, TDim> indices_select_tmp(tensor_select->getDimensions());
    indices_select_tmp.setData();
    indices_select_tmp.syncHAndDData(device);

    // apply the logical comparitor and modifier as a selection criteria
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, TDim>> tensor_select_values(tensor_select->getDataPointer().get(), tensor_select->getDimensions());
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
    indices_select = std::make_shared<TensorDataGpuPrimitiveT<int, TDim>>(indices_select_tmp);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSelectIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice & device) const
  {
    // allocate memory for the indices
    TensorDataGpuPrimitiveT<int, TDim> indices_select_tmp(this->getDimensions());
    Eigen::Tensor<int, TDim> ones(this->getDimensions());
    ones.setConstant(1);
    indices_select_tmp.setData(ones);
    indices_select_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select_tmp.getDataPointer().get(), indices_select_tmp.getDimensions());

    // [PERFORMANCE: Can this be replaced with contractions?]
    for (const auto& axis_to_index : this->axes_to_dims_) {
      // determine the dimensions for reshaping and broadcasting the indices
      Eigen::array<Eigen::Index, TDim> indices_reshape_dimensions;
      Eigen::array<Eigen::Index, TDim> indices_bcast_dimensions;
      for (int i = 0; i < TDim; ++i) {
        if (i == this->axes_to_dims_.at(axis_to_index.first)) {
          indices_reshape_dimensions.at(i) = (int)this->axes_.at(axis_to_index.first)->getNLabels();
          indices_bcast_dimensions.at(i) = 1;
        }
        else {
          indices_reshape_dimensions.at(i) = 1;
          indices_bcast_dimensions.at(i) = this->getDimensions().at(i);
        }
      }

      // normalize and broadcast the indices across the tensor
      Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(indices_component.at(axis_to_index.first)->getDataPointer().get(), indices_reshape_dimensions);
      auto indices_view_bcast_values = indices_view_reshape.clip(0,1).broadcast(indices_bcast_dimensions);

      // update the indices_select_values
      indices_select_values.device(device) = indices_select_values * indices_view_bcast_values;
    }

    // move over the results
    indices_select = std::make_shared<TensorDataGpuPrimitiveT<int, TDim>>(indices_select_tmp);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::getSelectTensorDataFromIndicesView(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the sum of each axis
    TensorDataGpuPrimitiveT<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
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

    this->getSelectTensorDataFromIndices(tensor_select, indices_select, select_dimensions, device);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::sliceTensorDataForSort(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& tensor_sort,
    const std::string & axis_name_sort, const int & label_index_sort, const std::string & axis_name_apply, Eigen::GpuDevice & device)
  {
    // determine the offsets and extents for the slice operation
    Eigen::array<Eigen::Index, TDim> extents;
    Eigen::array<Eigen::Index, TDim> offsets;
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
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, TDim>> tensor_values(this->data_->getDataPointer().get(), this->data_->getDimensions());
    auto tensor_1d = tensor_values.slice(offsets, extents).reshape(Eigen::array<Eigen::Index, 1>({ (int)this->axes_.at(axis_name_apply)->getNLabels() }));

    // allocate memory for the slice
    TensorDataGpuClassT<ArrayT, TensorT, 1> tensor_sort_tmp(Eigen::array<Eigen::Index, 1>({ (int)this->axes_.at(axis_name_apply)->getNLabels() }));
    tensor_sort_tmp.setData();
    tensor_sort_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 1>> tensor_sort_values(tensor_sort_tmp.getDataPointer().get(), tensor_sort_tmp.getDimensions());
    tensor_sort_values.device(device) = tensor_1d;

    // move over the tensor sort data
    tensor_sort = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 1>>(tensor_sort_tmp);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSortIndicesFromTensorIndicesComponent(const std::map<std::string, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>>& indices_component, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_sort, Eigen::GpuDevice & device) const
  {
    // allocate memory for the indices
    TensorDataGpuPrimitiveT<int, TDim> indices_sort_tmp(this->getDimensions());
    Eigen::Tensor<int, TDim> zeros(this->getDimensions());
    zeros.setZero();
    indices_sort_tmp.setData(zeros);
    indices_sort_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());

    // [PERFORMANCE: Can this be replaced with contractions?]
    int accumulative_size = 1;
    for (const auto& axis_to_index : this->axes_to_dims_) {
      // determine the dimensions for reshaping and broadcasting the indices
      Eigen::array<Eigen::Index, TDim> indices_reshape_dimensions;
      Eigen::array<Eigen::Index, TDim> indices_bcast_dimensions;
      for (int i = 0; i < TDim; ++i) {
        if (i == this->axes_to_dims_.at(axis_to_index.first)) {
          indices_reshape_dimensions.at(i) = (int)this->axes_.at(axis_to_index.first)->getNLabels();
          indices_bcast_dimensions.at(i) = 1;
        }
        else {
          indices_reshape_dimensions.at(i) = 1;
          indices_bcast_dimensions.at(i) = this->getDimensions().at(i);
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
    indices_sort = std::make_shared<TensorDataGpuPrimitiveT<int, TDim>>(indices_sort_tmp);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline int TensorTableGpuClassT<ArrayT, TensorT, TDim>::getFirstIndexFromIndicesView(const std::string & axis_name, Eigen::GpuDevice & device)
  {
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    return this->indices_view_.at(axis_name)->getData()(0); // the first occurance of the label
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeAppendIndices(const std::string & axis_name, const int & n_labels, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice & device)
  {
    // Allocate memory for the extend axis indices
    TensorDataGpuPrimitiveT<int, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ n_labels }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);

		Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices_tmp.getDataPointer().get(), n_labels);
		if (this->indices_view_.at(axis_name)->getTensorSize() > 0) {
			// Determine the maximum index value
      Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_values(this->indices_view_.at(axis_name)->getDataPointer().get(), 1, this->indices_view_.at(axis_name)->getTensorSize());
      auto max_bcast = indices_view_values.maximum(Eigen::array<Eigen::Index, 1>({ 1 })).eval().broadcast(Eigen::array<Eigen::Index, 1>({ n_labels })).eval();

			// Make the extended axis indices
			auto tmp = indices_values.constant(1).cumsum(0, false);
			indices_values.device(device) = max_bcast + tmp;
		}
		else {
			// Make the extended axis indices
			auto tmp = indices_values.constant(1).cumsum(0, false);
			indices_values.device(device) = tmp;
		}

    // Move over the indices to the output
    indices = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_tmp);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSelectIndicesFromIndices(const std::string & axis_name, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices
    TensorDataGpuPrimitiveT<int, TDim> indices_select_tmp(this->getDimensions());
    indices_select_tmp.setData();
    indices_select_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_select_values(indices_select_tmp.getDataPointer().get(), indices_select_tmp.getDimensions());

    // Determine the dimensions for reshaping and broadcasting
    Eigen::array<Eigen::Index, TDim> indices_reshape_dimensions;
    Eigen::array<Eigen::Index, TDim> indices_bcast_dimensions;
    for (const auto& axis_to_index : this->axes_to_dims_) {
      if (axis_to_index.first == axis_name) {
        indices_reshape_dimensions.at(axis_to_index.second) = this->getDimensions().at(axis_to_index.second);
        indices_bcast_dimensions.at(axis_to_index.second) = 1;
      }
      else {
        indices_reshape_dimensions.at(axis_to_index.second) = 1;
        indices_bcast_dimensions.at(axis_to_index.second) = this->getDimensions().at(axis_to_index.second);
      }
    }

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, TDim>> indices_view_reshape(indices->getDataPointer().get(), indices_reshape_dimensions);
    auto indices_view_bcast_values = indices_view_reshape.clip(0, 1).broadcast(indices_bcast_dimensions);

    // update the indices_select_values
    indices_select_values.device(device) = indices_view_bcast_values;

    // move over the results
    indices_select = std::make_shared<TensorDataGpuPrimitiveT<int, TDim>>(indices_select_tmp);
  }

  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::getSelectTensorDataFromIndices(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_select, const Eigen::array<Eigen::Index, TDim>& dimensions_select, Eigen::GpuDevice & device)
  {
    // allocate memory for the selected tensor
    TensorDataGpuClassT<ArrayT, TensorT, TDim> tensor_select_tmp(dimensions_select);
    tensor_select_tmp.setData();
    tensor_select_tmp.syncHAndDData(device);
    tensor_select = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, TDim>>(tensor_select_tmp);

    // select the tensor
    assert(this->getDataTensorSize() == 0 || this->getDataDimensions() == indices_select->getDimensions());
    this->data_->select(tensor_select, indices_select, device);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice & device)
  {
    // Normalize the indices view
    auto indices_view_copy = this->indices_view_.at(axis_name)->copy(device);
    indices_view_copy->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_copy_values(indices_view_copy->getDataPointer().get(), indices_view_copy->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_values(this->indices_view_.at(axis_name)->getDataPointer().get(), this->indices_view_.at(axis_name)->getDimensions());
    indices_view_copy_values.device(device) = indices_view_values.clip(0, 1);

    // Determine the size of the indices
    TensorDataGpuPrimitiveT<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    dim_size.setData();
    dim_size.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> dim_size_value(dim_size.getDataPointer().get());
    dim_size_value.device(device) = indices_view_copy_values.sum();
    dim_size.syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // Allocate memory for the indices
    TensorDataGpuPrimitiveT<int, 1> indices_tmp(Eigen::array<Eigen::Index, 1>({ dim_size.getData()(0) }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_tmp);

    // Select out the non zero indices
    this->indices_view_.at(axis_name)->select(indices, indices_view_copy, device);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSparseAxisLabelsFromIndicesView(std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the total sum of each axis
    TensorDataGpuPrimitiveT<int, 1> dim_size(Eigen::array<Eigen::Index, 1>({ 1 }));
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
      TensorDataGpuPrimitiveT<int, 1> indices_selected(Eigen::array<Eigen::Index, 1>({ dim_size.getData()(0) }));
      indices_selected.setData();
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_selected_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_selected);
      indices_selected_ptr->syncHAndDData(device);
      this->indices_view_.at(axis_to_name.first)->select(indices_selected_ptr, indices_select, device);
      indices_selected_vec.push_back(indices_selected_ptr);
    }

    // allocate memory for the labels
    TensorDataGpuPrimitiveT<int, 2> sparse_labels(Eigen::array<Eigen::Index, 2>({ (int)this->axes_to_dims_.size(), labels_size }));
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
        Eigen::array<Eigen::Index, 2> offsets = { i, j * slice_size };
        Eigen::array<Eigen::Index, 2> extents = { 1, slice_size };
        sparse_labels_values.slice(offsets, extents).device(device) = indices_bcast;
      }
    }

    // move over the output
    sparse_select = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(sparse_labels);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSparseTensorTable(const Eigen::Tensor<std::string, 1>& sparse_dimensions, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& sparse_labels, const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& sparse_data, std::shared_ptr<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& sparse_table, Eigen::GpuDevice & device)
  {
    sparse_labels->syncHAndDData(device); // d to h
    sparse_data->syncHAndDData(device); // d to h
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // make the sparse axis
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("Indices", sparse_dimensions, sparse_labels->getData()));

    // make the values axis
    Eigen::Tensor<std::string, 1> values_dimension(1);
    values_dimension.setValues({ "Values" });
    Eigen::Tensor<int, 2> values_labels(1, 1);
    values_labels.setZero();
    std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("Values", values_dimension, values_labels));

    // add the axes to the tensorTable
    TensorTableGpuClassT<ArrayT, TensorT, 2> tensorTable;
    tensorTable.addTensorAxis(axis_1_ptr);
    tensorTable.addTensorAxis(axis_2_ptr);
    tensorTable.setAxes(device);

    // set the data
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> sparse_data_values(sparse_data->getData().data(), sparse_data->getTensorSize(), 1);
    tensorTable.setData(sparse_data_values);

    // sync the data
    tensorTable.syncIndicesHAndDData(device);
    tensorTable.syncIndicesViewHAndDData(device);
    tensorTable.syncNotInMemoryHAndDData(device);
    tensorTable.syncIsModifiedHAndDData(device);
    tensorTable.syncShardIdHAndDData(device);
    tensorTable.syncShardIndicesHAndDData(device);
    tensorTable.syncAxesHAndDData(device);
    tensorTable.syncHAndDData(device);

    // move over the table
    sparse_table = std::make_shared<TensorTableGpuClassT<ArrayT, TensorT, 2>>(tensorTable);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeShardIndicesFromShardIDs(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices_shard, Eigen::GpuDevice & device) const
  {
    // allocate memory for the indices
    TensorDataGpuPrimitiveT<int, TDim> indices_shard_tmp(this->getDimensions());
    indices_shard_tmp.setData();
    indices_shard_tmp.syncHAndDData(device);
    indices_shard = std::make_shared<TensorDataGpuPrimitiveT<int, TDim>>(indices_shard_tmp);

    // make the shard indices
    TensorShard::makeShardIndicesFromShardIDs(this->getAxesToDims(), this->getShardSpans(), this->getDimensions(), this->getMaximumDimensions(), this->getShardId(), indices_shard, device);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::runLengthEncodeIndex(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& data, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& n_runs, Eigen::GpuDevice &device) const
  {
    // Allocate memory
    TensorDataGpuPrimitiveT<int, 1> unique_tmp(Eigen::array<Eigen::Index, 1>({ (int)data->getTensorSize() }));
    unique_tmp.setData();
    unique_tmp.syncHAndDData(device);
    TensorDataGpuPrimitiveT<int, 1> count_tmp(Eigen::array<Eigen::Index, 1>({ (int)data->getTensorSize() }));
    count_tmp.setData();
    count_tmp.syncHAndDData(device);
    TensorDataGpuPrimitiveT<int, 1> n_runs_tmp(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_runs_tmp.setData();
    n_runs_tmp.syncHAndDData(device);

    // Move over the memory
    unique = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(unique_tmp);
    count = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(count_tmp);
    n_runs = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(n_runs_tmp);

    // Run the algorithm
    data->runLengthEncode(unique, count, n_runs, device);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline int TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSliceIndicesFromShardIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& modified_shard_ids, std::map<int, std::pair<Eigen::array<Eigen::Index, TDim>, Eigen::array<Eigen::Index, TDim>>>& slice_indices, Eigen::array<Eigen::Index, TDim>& shard_data_dimensions, Eigen::GpuDevice & device) const
  {
    if (modified_shard_ids->getTensorSize() == 0) return 0;

    // broadcast the indices view to select the indices for each modified shard
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>> indices_sort;
    this->makeSortIndicesFromTensorIndicesComponent(indices_, indices_sort, device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(indices_sort->getDataPointer().get(), 1, (int)indices_sort->getTensorSize());
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<Eigen::Index, 2>({ (int)modified_shard_ids->getTensorSize(), 1 }));

    // broadcast the modified_shard_ids
    Eigen::TensorMap<Eigen::Tensor<int, 2>> modified_shard_ids_reshape(modified_shard_ids->getDataPointer().get(), (int)modified_shard_ids->getTensorSize(), 1);
    auto modified_shard_ids_bcast = modified_shard_ids_reshape.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)indices_sort->getTensorSize() }));

    // broadcast the shard_ids
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>> shard_id_indices;
    this->makeShardIndicesFromShardIDs(shard_id_indices, device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> shard_ids_reshape(shard_id_indices->getDataPointer().get(), 1, (int)shard_id_indices->getTensorSize());
    auto shard_ids_bcast = shard_ids_reshape.broadcast(Eigen::array<Eigen::Index, 2>({ (int)modified_shard_ids->getTensorSize(), 1 }));

    // select the indices that correspond to the matching shard ids, and normalize to 0-based indexing
    auto shard_ids_slice_indices = (modified_shard_ids_bcast == shard_ids_bcast).select(indices_bcast, indices_bcast.constant(0)) - indices_bcast.constant(1);

    // Allocate temporary memory for the min/max indices (Device-specific code block)
    TensorDataGpuPrimitiveT<int, 1> shard_slice_min(Eigen::array<Eigen::Index, 1>({ (int)modified_shard_ids->getTensorSize() }));
    shard_slice_min.setData();
    TensorDataGpuPrimitiveT<int, 1> shard_slice_max(Eigen::array<Eigen::Index, 1>({ (int)modified_shard_ids->getTensorSize() }));
    shard_slice_max.setData();

    // find the min and max indices values (along Dim=1) 
    shard_slice_min.syncHAndDData(device); // H to D
    shard_slice_max.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_ids_slice_max(shard_slice_max.getDataPointer().get(), (int)shard_slice_max.getTensorSize());
    shard_ids_slice_max.device(device) = shard_ids_slice_indices.maximum(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::TensorMap<Eigen::Tensor<int, 1>> shard_ids_slice_min(shard_slice_min.getDataPointer().get(), (int)shard_slice_min.getTensorSize());
    auto shard_ids_slice_indices_min = (shard_ids_slice_indices >= shard_ids_slice_indices.constant(0)).select(shard_ids_slice_indices, shard_ids_slice_indices.constant(this->getMaxInt())); // substitute -1 with a large number prior to calling minimum
    shard_ids_slice_min.device(device) = shard_ids_slice_indices_min.minimum(Eigen::array<Eigen::Index, 1>({ 1 }));
    shard_slice_min.syncHAndDData(device); // D to H
    shard_slice_max.syncHAndDData(device);

    // initialize the slice indices
    modified_shard_ids->syncHAndDData(device);// D to H
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    for (int i = 0; i < modified_shard_ids->getTensorSize(); ++i) {
      slice_indices.emplace(modified_shard_ids->getData()(i), std::make_pair(Eigen::array<Eigen::Index, TDim>(), Eigen::array<Eigen::Index, TDim>()));
    }

    // assign the slice indices based on determining the individual axes indices from the linearized indices_sort value 
    int axis_size_cum = 1;
    shard_data_dimensions = Eigen::array<Eigen::Index, TDim>();
    int shard_data_size = 1;
    for (const auto& axis_to_dim : this->axes_to_dims_) {
      // NOTE: not sure if this part can be done on the GPU using a % b = a - (b * int(a/b)) as the modulo operator
      int minimum_index = this->getMaxInt();
      int maximum_index = 0;
      // PARALLEL: could execute this code using multiple Threads though
      for (int i = 0; i < modified_shard_ids->getTensorSize(); ++i) {
        int min_index = int(floor(float(shard_slice_min.getData()(i)) / float(axis_size_cum))) % this->axes_.at(axis_to_dim.first)->getNLabels();
        slice_indices.at(modified_shard_ids->getData()(i)).first.at(axis_to_dim.second) = min_index;
        int max_index = int(floor(float(shard_slice_max.getData()(i)) / float(axis_size_cum))) % this->axes_.at(axis_to_dim.first)->getNLabels();
        int span = max_index - min_index + 1;
        if (min_index < 0 || max_index < 0 || span < 0) {
          std::cout << "Check that the maximum_dimensions are set correctly!" << std::endl;
        }
        slice_indices.at(modified_shard_ids->getData()(i)).second.at(axis_to_dim.second) = span;
        minimum_index = std::min(minimum_index, min_index);
        maximum_index = std::max(maximum_index, max_index);
      }
      // Estimate the dimensions based off the the unique indices
      int shard_data_size_dim_estimate = maximum_index + 1; // NOTE: assumes starting from 0
      //int shard_data_size_dim_estimate = maximum_index - minimum_index + 1; // NOTE: assumes starting from minimum_index (but we need to adjust the slice indices accordingly)
      shard_data_dimensions.at(axis_to_dim.second) = shard_data_size_dim_estimate;
      shard_data_size *= shard_data_size_dim_estimate;

      //// Re-adjust the offset slice index to the minimium_index
      //// NOTE: assumes starting from minimum_index
      //for (int i = 0; i < modified_shard_ids->getTensorSize(); ++i) {
      //  int min_index = int(floor(float(shard_slice_min.getData()(i)) / float(axis_size_cum))) % this->axes_.at(axis_to_dim.first)->getNLabels();
      //  slice_indices.at(modified_shard_ids->getData()(i)).first.at(axis_to_dim.second) = min_index - minimum_index;
      //}

      // update the accumulative size
      axis_size_cum *= this->axes_.at(axis_to_dim.first)->getNLabels();
    }

    return shard_data_size;
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeShardIDTensor(std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& modified_shard_ids, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& num_runs, Eigen::GpuDevice& device) const
  {
    // Resize the unique results and remove 0's from the unique
    unique->syncHAndDData(device); // d to h
    num_runs->syncHAndDData(device); // d to h

    if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    }

    if (num_runs->getData()(0) == 1 && unique->getData()(0) == 0) {
      unique->setDimensions(Eigen::array<Eigen::Index, 1>({ 0 }));
    }
    else if (unique->getData()(0) == 0) {
      Eigen::TensorMap<Eigen::Tensor<int, 1>> unqiue_values(unique->getDataPointer().get(), unique->getDimensions());
      unqiue_values.slice(Eigen::array<Eigen::Index, 1>({ 0 }), Eigen::array<Eigen::Index, 1>({ num_runs->getData()(0) - 1 })).device(device) = unqiue_values.slice(Eigen::array<Eigen::Index, 1>({ 1 }), Eigen::array<Eigen::Index, 1>({ num_runs->getData()(0) - 1 }));
      unique->setDimensions(Eigen::array<Eigen::Index, 1>({ num_runs->getData()(0) - 1 }));
    }
    else {
      unique->setDimensions(Eigen::array<Eigen::Index, 1>({ num_runs->getData()(0) }));
    }
    unique->setDataStatus(false, true);
    num_runs->setDataStatus(false, true);
    modified_shard_ids = unique;
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline bool TensorTableGpuClassT<ArrayT, TensorT, TDim>::loadTensorTableBinary(const std::string & dir, Eigen::GpuDevice & device)
  {
    // determine the shards to read from disk
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> not_in_memory_shard_ids;
    makeNotInMemoryShardIDTensor(not_in_memory_shard_ids, device);
    if (not_in_memory_shard_ids->getTensorSize() == 0) {
      //std::cout << "No shards have been modified." << std::endl; // TODO: Move to logging
      return false;
    }

    // make the slices for the shards
    std::map<int, std::pair<Eigen::array<Eigen::Index, TDim>, Eigen::array<Eigen::Index, TDim>>> slice_indices;
    Eigen::array<Eigen::Index, TDim> shard_dimensions;
    const int data_size = this->makeSliceIndicesFromShardIndices(not_in_memory_shard_ids, slice_indices, shard_dimensions, device);

    // check if enough data is allocated for the slices
    bool data_dims_too_small = false;
    for (int i = 0; i < TDim; ++i) if (this->getDataDimensions().at(i) < shard_dimensions.at(i)) data_dims_too_small = true;
    if (data_dims_too_small) {
      this->setDataShards(this->getDimensions(), device);
      //this->setDataShards(shard_dimensions, device);
    }
    else {
      this->syncHData(device); // D to H
      if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      }
    }

    // read in the shards and update the TensorTable data asyncronously
    for (const auto slice_index : slice_indices) {
      // read in the shard
      const std::string filename = makeTensorTableShardFilename(dir, getName(), slice_index.first);
      Eigen::Tensor<ArrayT<TensorT>, TDim> shard_data(slice_index.second.second);
      DataFile::loadDataBinary<ArrayT<TensorT>, TDim>(filename, shard_data);
      assert(slice_index.second.second == shard_data.dimensions());

      // slice and update the data with the shard data
      this->getData().slice(slice_index.second.first, slice_index.second.second) = shard_data;

      // update the `not_in_memory` tensor table attribute
      for (auto& not_in_memory_map : not_in_memory_) {
        Eigen::array<Eigen::Index, 1> offset;
        offset.at(0) = slice_index.second.first.at(this->getDimFromAxisName(not_in_memory_map.first));
        Eigen::array<Eigen::Index, 1> span;
        span.at(0) = slice_index.second.second.at(this->getDimFromAxisName(not_in_memory_map.first));
        Eigen::TensorMap<Eigen::Tensor<int, 1>> not_in_memory_values(not_in_memory_map.second->getDataPointer().get(), (int)not_in_memory_map.second->getTensorSize());
        not_in_memory_values.slice(offset, span).device(device) = not_in_memory_values.slice(offset, span).constant(0);
      }
    }
    this->syncDData(device); // H to D

    return true;
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline bool TensorTableGpuClassT<ArrayT, TensorT, TDim>::storeTensorTableBinary(const std::string & dir, Eigen::GpuDevice & device)
  {
    // determine the shards to write to disk
    if (this->getDataTensorSize()) {
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> modified_shard_ids;
      makeModifiedShardIDTensor(modified_shard_ids, device);
      if (modified_shard_ids->getTensorSize() == 0) {
        //std::cout << "No shards have been modified." << std::endl; // TODO: Move to logging
        return false;
      }
      std::map<int, std::pair<Eigen::array<Eigen::Index, TDim>, Eigen::array<Eigen::Index, TDim>>> slice_indices;
      Eigen::array<Eigen::Index, TDim> shard_dimensions;
      const int data_size = this->makeSliceIndicesFromShardIndices(modified_shard_ids, slice_indices, shard_dimensions, device);

      // write the TensorTable shards to disk asyncronously
      this->syncHData(device); // D to H
      if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      }
      for (const auto slice_index : slice_indices) {
        const std::string filename = makeTensorTableShardFilename(dir, getName(), slice_index.first);
        Eigen::Tensor<ArrayT<TensorT>, TDim> shard_data = getData().slice(slice_index.second.first, slice_index.second.second);
        DataFile::storeDataBinary<ArrayT<TensorT>, TDim>(filename, shard_data);

        // update the `is_modified` tensor table attribute
        for (auto& is_modified_map : is_modified_) {
          Eigen::array<Eigen::Index, 1> offset;
          offset.at(0) = slice_index.second.first.at(this->getDimFromAxisName(is_modified_map.first));
          Eigen::array<Eigen::Index, 1> span;
          span.at(0) = slice_index.second.second.at(this->getDimFromAxisName(is_modified_map.first));
          Eigen::TensorMap<Eigen::Tensor<int, 1>> is_modified_values(is_modified_map.second->getDataPointer().get(), (int)is_modified_map.second->getTensorSize());
          is_modified_values.slice(offset, span).device(device) = is_modified_values.slice(offset, span).constant(0);
        }
      }
      this->setDataStatus(false, true);
    }
    return true;
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorTableGpuClassT<ArrayT, TensorT, TDim>::makeSparseTensorTableFromCsv(std::shared_ptr<TensorTable<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& sparse_table_ptr, const Eigen::Tensor<std::string, 2>& data_new, Eigen::GpuDevice & device)
  {
    // Convert from string to TensorT and reshape to n_data x 1
    TensorTableGpuClassT<ArrayT, TensorT, 2> sparse_table;
    Eigen::array<Eigen::Index, 2> new_dimensions = { int(data_new.size()), 1 };
    sparse_table.setDimensions(new_dimensions);
    sparse_table.initData(new_dimensions, device);
    sparse_table.setData();
    sparse_table.syncHAndDData(device);
    sparse_table.convertDataFromStringToTensorT(data_new, device);
    sparse_table_ptr = std::make_shared<TensorTableGpuClassT<ArrayT, TensorT, 2>>(sparse_table);
  }
};

// Cereal registration of TensorTs: charArray8 and TDims: 1, 2, 3, 4
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu8, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu32, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu128, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu512, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableGpuClassT<TensorBase::TensorArrayGpu2048, char, 4>);
#endif
#endif //TENSORBASE_TENSORTABLEGPUCLASST_H