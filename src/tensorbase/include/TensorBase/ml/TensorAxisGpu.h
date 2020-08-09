/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISGPU_H
#define TENSORBASE_TENSORAXISGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorDataGpu.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxis.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT>
  class TensorAxisGpuPrimitiveT : public TensorAxis<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorAxisGpuPrimitiveT() = default;  ///< Default constructor
    TensorAxisGpuPrimitiveT(const std::string& name, const size_t& n_dimensions, const size_t& n_labels) : TensorAxis(name, n_dimensions, n_labels) { this->setLabels(); };
    TensorAxisGpuPrimitiveT(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) : TensorAxis(name) { this->setDimensionsAndLabels(dimensions, labels); };
    ~TensorAxisGpuPrimitiveT() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 2>& labels) override;
    void setLabels() override;
    std::shared_ptr<TensorAxis<TensorT, Eigen::GpuDevice>> copyToHost(Eigen::GpuDevice& device) override;
    std::shared_ptr<TensorAxis<TensorT, Eigen::GpuDevice>> copyToDevice(Eigen::GpuDevice& device) override;
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice& device) override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::GpuDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxis<TensorT, Eigen::GpuDevice>>(this));
    }
  };
  template<typename TensorT>
  void TensorAxisGpuPrimitiveT<TensorT>::setLabels(const Eigen::Tensor<TensorT, 2>& labels) {
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(TensorDataGpuPrimitiveT<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(TensorDataGpuPrimitiveT<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<typename TensorT>
  inline std::shared_ptr<TensorAxis<TensorT, Eigen::GpuDevice>> TensorAxisGpuPrimitiveT<TensorT>::copyToHost(Eigen::GpuDevice& device)
  {
    TensorAxisGpuPrimitiveT<TensorT> tensor_axis_copy;
    // copy the metadata
    tensor_axis_copy.setId(this->getId());
    tensor_axis_copy.setName(this->getName());

    // copy the dimensions and labels
    tensor_axis_copy.setNDimensions(this->getNDimensions());
    tensor_axis_copy.setNLabels(this->getNLabels());
    tensor_axis_copy.tensor_dimension_names_ = this->tensor_dimension_names_;
    tensor_axis_copy.tensor_dimension_labels_ = this->tensor_dimension_labels_->copyToHost(device);

    return std::make_shared<TensorAxisGpuPrimitiveT<TensorT>>(tensor_axis_copy);
  }
  template<typename TensorT>
  inline std::shared_ptr<TensorAxis<TensorT, Eigen::GpuDevice>> TensorAxisGpuPrimitiveT<TensorT>::copyToDevice(Eigen::GpuDevice& device)
  {
    TensorAxisGpuPrimitiveT<TensorT> tensor_axis_copy;
    // copy the metadata
    tensor_axis_copy.setId(this->getId());
    tensor_axis_copy.setName(this->getName());

    // copy the dimensions and labels
    tensor_axis_copy.setNDimensions(this->getNDimensions());
    tensor_axis_copy.setNLabels(this->getNLabels());
    tensor_axis_copy.tensor_dimension_names_ = this->tensor_dimension_names_;
    tensor_axis_copy.tensor_dimension_labels_ = this->tensor_dimension_labels_->copyToDevice(device);

    return std::make_shared<TensorAxisGpuPrimitiveT<TensorT>>(tensor_axis_copy);
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataGpuPrimitiveT<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    axis_size.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    auto indices_view_norm = (indices_values.cast<float>() / (indices_values.cast<float>() + indices_values.cast<float>().constant(1e-12))).cast<int>();
    axis_size_value.device(device) = indices_view_norm.sum();
    axis_size.syncHData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // allocate memory for the new labels
    TensorDataGpuPrimitiveT<TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> new_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(new_labels);
    new_labels_ptr->syncDData(device);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataGpuPrimitiveT<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_select);
    indices_select_ptr->syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select_ptr->getDataPointer().get(), indices_select_ptr->getDimensions());
    indices_select_values.device(device) = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, 1 }));

    // perform the reduction on the labels and move over the results
    this->tensor_dimension_labels_->select(new_labels_ptr, indices_select_ptr, device);
    labels_select = new_labels_ptr;
  };

  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device)
  {
    assert(labels->getDimensions().at(0) == this->n_dimensions_);

    // copy the original number of labels
    size_t n_labels_copy = this->n_labels_;

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataGpuPrimitiveT<TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    std::shared_ptr<TensorDataGpuPrimitiveT<TensorT, 2>> labels_concat_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(labels_concat);
    labels_concat_ptr->syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_concat_values(labels_concat_ptr->getDataPointer().get(), labels_concat_ptr->getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    if (n_labels_copy > 0) {
      labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);
    }
    else {
      labels_concat_values.device(device) = new_labels_values;
    }

    // Move over the new labels
    this->tensor_dimension_labels_ = labels_concat_ptr;
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataGpuPrimitiveT<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), (int)this->getNLabels() }));
    indices_sort_tmp.setData();
    indices_sort_tmp.syncDData(device);

    // create a dummy index along the dimension
    TensorDataGpuPrimitiveT<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    indices_dimension.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_dimension_reshape(indices_dimension.getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // normalize and broadcast the dummy indices across the tensor    
    auto indices_dimension_norm = indices_dimension_reshape - indices_dimension_reshape.constant(1);
    auto indices_dimension_bcast_values = indices_dimension_norm.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_reshape(indices->getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));
    auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(this->getNDimensions());
    auto indices_view_bcast_values = indices_view_norm.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // update the indices_sort_values
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());
    indices_sort_values.device(device) = indices_view_bcast_values + indices_dimension_bcast_values + indices_sort_values.constant(1);

    // move over the results
    indices_sort = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_sort_tmp);
  }
  template<typename TensorT>
  inline bool TensorAxisGpuPrimitiveT<TensorT>::loadLabelsBinary(const std::string & filename, Eigen::GpuDevice & device)
  {
    // Read in the the labels
    this->setDataStatus(true, false);
    Eigen::Tensor<TensorT, 2> labels_data((int)this->n_dimensions_, (int)this->n_labels_);
    DataFile::loadDataBinary<TensorT, 2>(filename + ".ta", labels_data);
    this->getLabels() = labels_data;
    this->syncDData(device); // H to D
    return true;
  }
  template<typename TensorT>
  inline bool TensorAxisGpuPrimitiveT<TensorT>::storeLabelsBinary(const std::string & filename, Eigen::GpuDevice & device)
  {
    // Store the labels
    if (this->getNLabels()*this->getNLabels() > 0) {
      this->syncHData(device); // D to H
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      DataFile::storeDataBinary<TensorT, 2>(filename + ".ta", this->getLabels());
      this->setDataStatus(false, true);
    }
    return true;
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::GpuDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataGpuPrimitiveT<TensorT, 2> labels_converted(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    labels_converted.setData();
    labels_converted.syncDData(device);
    labels_converted.convertFromStringToTensorT(labels, device);

    // Make the labels unique
    TensorDataGpuPrimitiveT<int, 3> labels_unique_tmp(Eigen::array<Eigen::Index, 3>({ 1, (int)labels.dimension(1), (int)labels.dimension(1) }));
    labels_unique_tmp.setData();
    labels_unique_tmp.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 3>> labels_new_v1v2_prod(labels_unique_tmp.getDataPointer().get(), 1, (int)labels.dimension(1), (int)labels.dimension(1));
    // Make the indices unique
    TensorDataGpuPrimitiveT<int, 2> indices_unique(Eigen::array<Eigen::Index, 2>({ 1, (int)labels.dimension(1) }));
    indices_unique.setData();
    indices_unique.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_unique_values(indices_unique.getDataPointer().get(), 1, (int)labels.dimension(1));
    // Make the indices select
    TensorDataGpuPrimitiveT<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    indices_select.setData();
    indices_select.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), (int)labels.dimension(0), (int)labels.dimension(1));

    // Determine the unique input axis labels
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_unique_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_unique_values_bcast = indices_unique_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v1(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_v1_bcast = labels_new_v1.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v2(labels_converted.getDataPointer().get(), 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1));
    auto labels_new_v2_bcast = labels_new_v2.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    // Select the overlapping labels
    auto labels_new_v1v2_select = (labels_new_v1_bcast == labels_new_v2_bcast).select(indices_unique_values_bcast.constant(1), indices_unique_values_bcast.constant(0));
    // Reduct along the axis dimensions and then cum sum along the axis labels
    labels_new_v1v2_prod.device(device) = labels_new_v1v2_select.sum(Eigen::array<Eigen::Index, 2>({ 1, 3 })).clip(0, 1);
    auto labels_new_v1v2_cumsum = (labels_new_v1v2_prod.cumsum(1) * labels_new_v1v2_prod).cumsum(2) * labels_new_v1v2_prod;
    // Select the unique labels marked with a 1
    auto labels_unique_v1v2 = (labels_new_v1v2_cumsum == labels_new_v1v2_cumsum.constant(1)).select(labels_new_v1v2_cumsum.constant(1), labels_new_v1v2_cumsum.constant(0));
    // Collapse back to 1xnlabels by summing along one of the axis labels dimensions
    indices_unique_values.device(device) = labels_unique_v1v2.sum(Eigen::array<Eigen::Index, 1>({ 2 })).clip(0, 1);

    if (this->n_labels_ > 0) {
      // Determine the new labels to add to the axis 
      Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_select_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
      auto indices_select_values_bcast = indices_select_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
      // Broadcast along the labels and along the dimensions and for both the labels and the new labels
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_);
      auto labels_values_bcast = labels_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_values(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
      auto labels_new_values_bcast = labels_new_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
      // Select and sum along the labels and multiple along the dimensions
      auto labels_selected = (labels_values_bcast == labels_new_values_bcast).select(indices_select_values_bcast.constant(1), indices_select_values_bcast.constant(0)).sum(
        Eigen::array<Eigen::Index, 2>({ 3, 4 })).prod(Eigen::array<Eigen::Index, 1>({ 1 })); // not new > 1, new = 0
      // Invert the selection
      auto labels_new = (labels_selected > labels_selected.constant(0)).select(labels_selected.constant(0), labels_selected.constant(1)); // new = 1, not new = 0

      // Determine the new and unique labels to add to the axis
      auto labels_new_unique = labels_new * indices_unique_values;
      // Broadcast back to n_dimensions x labels
      auto labels_new_unique_bcast = labels_new_unique.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

      // Store the selection indices
      indices_select_values.device(device) = labels_new_unique_bcast;
    }
    else {
      // Broadcast back to n_dimensions x labels
      auto labels_new_unique_bcast = indices_unique_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

      // Store the selection indices
      indices_select_values.device(device) = labels_new_unique_bcast;
    }
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_select);

    // Determine the number of new labels
    TensorDataGpuPrimitiveT<int, 1> n_labels_new(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_labels_new.setData();
    n_labels_new.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> n_labels_new_values(n_labels_new.getDataPointer().get());
    n_labels_new_values.device(device) = indices_select_values.sum() / n_labels_new_values.constant((int)labels.dimension(0));
    n_labels_new.syncHData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // Allocate memory for the new labels
    TensorDataGpuPrimitiveT<TensorT, 2> labels_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, n_labels_new.getData()(0) }));
    labels_select.setData();
    labels_select.syncDData(device);
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> labels_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(labels_select);

    // Select the labels
    labels_converted.select(labels_select_ptr, indices_select_ptr, device);

    // Append the selected labels to the axis
    this->appendLabelsToAxis(labels_select_ptr, device);
  }

  template<template<class> class ArrayT, class TensorT>
  class TensorAxisGpuClassT : public TensorAxis<ArrayT<TensorT>, Eigen::GpuDevice>
  {
  public:
    TensorAxisGpuClassT() = default;  ///< Default constructor
    TensorAxisGpuClassT(const std::string& name, const size_t& n_dimensions, const size_t& n_labels) : TensorAxis(name, n_dimensions, n_labels) { this->setLabels(); };
    TensorAxisGpuClassT(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<ArrayT<TensorT>, 2>& labels) : TensorAxis(name) { this->setDimensionsAndLabels(dimensions, labels); };
    ~TensorAxisGpuClassT() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<ArrayT<TensorT>, 2>& labels) override;
    void setLabels() override;
    std::shared_ptr<TensorAxis<ArrayT<TensorT>, Eigen::GpuDevice>> copyToHost(Eigen::GpuDevice& device) override;
    std::shared_ptr<TensorAxis<ArrayT<TensorT>, Eigen::GpuDevice>> copyToDevice(Eigen::GpuDevice& device) override;
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice& device) override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::GpuDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxis<ArrayT<TensorT>, Eigen::GpuDevice>>(this));
    }
  };
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::setLabels(const Eigen::Tensor<ArrayT<TensorT>, 2>& labels) {
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(TensorDataGpuClassT<ArrayT, TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(TensorDataGpuClassT<ArrayT, TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<template<class> class ArrayT, class TensorT>
  inline std::shared_ptr<TensorAxis<ArrayT<TensorT>, Eigen::GpuDevice>> TensorAxisGpuClassT<ArrayT, TensorT>::copyToHost(Eigen::GpuDevice& device)
  {
    TensorAxisGpuClassT<ArrayT, TensorT> tensor_axis_copy;
    // copy the metadata
    tensor_axis_copy.setId(this->getId());
    tensor_axis_copy.setName(this->getName());

    // copy the dimensions and labels
    tensor_axis_copy.setNDimensions(this->getNDimensions());
    tensor_axis_copy.setNLabels(this->getNLabels());
    tensor_axis_copy.tensor_dimension_names_ = this->tensor_dimension_names_;
    tensor_axis_copy.tensor_dimension_labels_ = this->tensor_dimension_labels_->copyToHost(device);

    return std::make_shared<TensorAxisGpuClassT<ArrayT, TensorT>>(tensor_axis_copy);
  }
  template<template<class> class ArrayT, class TensorT>
  inline std::shared_ptr<TensorAxis<ArrayT<TensorT>, Eigen::GpuDevice>> TensorAxisGpuClassT<ArrayT, TensorT>::copyToDevice(Eigen::GpuDevice& device)
  {
    TensorAxisGpuClassT<ArrayT, TensorT> tensor_axis_copy;
    // copy the metadata
    tensor_axis_copy.setId(this->getId());
    tensor_axis_copy.setName(this->getName());

    // copy the dimensions and labels
    tensor_axis_copy.setNDimensions(this->getNDimensions());
    tensor_axis_copy.setNLabels(this->getNLabels());
    tensor_axis_copy.tensor_dimension_names_ = this->tensor_dimension_names_;
    tensor_axis_copy.tensor_dimension_labels_ = this->tensor_dimension_labels_->copyToDevice(device);

    return std::make_shared<TensorAxisGpuClassT<ArrayT, TensorT>>(tensor_axis_copy);
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataGpuPrimitiveT<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    axis_size.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    auto indices_view_norm = (indices_values.cast<float>() / (indices_values.cast<float>() + indices_values.cast<float>().constant(1e-12))).cast<int>();
    axis_size_value.device(device) = indices_view_norm.sum();
    axis_size.syncHData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // allocate memory for the new labels
    TensorDataGpuClassT<ArrayT, TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>> new_labels_ptr = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(new_labels);
    new_labels_ptr->syncDData(device);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataGpuPrimitiveT<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_select);
    indices_select_ptr->syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select_ptr->getDataPointer().get(), indices_select_ptr->getDimensions());
    indices_select_values.device(device) = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, 1 }));

    // perform the reduction on the labels and move over the results
    this->tensor_dimension_labels_->select(new_labels_ptr, indices_select_ptr, device);
    labels_select = new_labels_ptr;
  };

  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::appendLabelsToAxis(const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device)
  {
    assert(labels->getDimensions().at(0) == this->n_dimensions_);

    // copy the original number of labels
    size_t n_labels_copy = this->n_labels_;

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataGpuClassT<ArrayT, TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    std::shared_ptr<TensorDataGpuClassT<ArrayT, TensorT, 2>> labels_concat_ptr = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(labels_concat);
    labels_concat_ptr->syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> labels_concat_values(labels_concat_ptr->getDataPointer().get(), labels_concat_ptr->getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    if (n_labels_copy > 0) {
      labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);
    }
    else {
      labels_concat_values.device(device) = new_labels_values;
    }

    // Move over the new labels
    this->tensor_dimension_labels_ = labels_concat_ptr;
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataGpuPrimitiveT<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), (int)this->getNLabels() }));
    indices_sort_tmp.setData();
    indices_sort_tmp.syncDData(device);

    // create a dummy index along the dimension
    TensorDataGpuPrimitiveT<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    indices_dimension.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_dimension_reshape(indices_dimension.getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // normalize and broadcast the dummy indices across the tensor    
    auto indices_dimension_norm = indices_dimension_reshape - indices_dimension_reshape.constant(1);
    auto indices_dimension_bcast_values = indices_dimension_norm.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_reshape(indices->getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));
    auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(this->getNDimensions());
    auto indices_view_bcast_values = indices_view_norm.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // update the indices_sort_values
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());
    indices_sort_values.device(device) = indices_view_bcast_values + indices_dimension_bcast_values + indices_sort_values.constant(1);

    // move over the results
    indices_sort = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_sort_tmp);
  }
  template<template<class> class ArrayT, class TensorT>
  inline bool TensorAxisGpuClassT<ArrayT, TensorT>::loadLabelsBinary(const std::string & filename, Eigen::GpuDevice & device)
  {
    // Read in the the labels
    this->setDataStatus(true, false);
    Eigen::Tensor<ArrayT<TensorT>, 2> labels_data((int)this->n_dimensions_, (int)this->n_labels_);
    DataFile::loadDataBinary<ArrayT<TensorT>, 2>(filename + ".ta", labels_data);
    this->getLabels() = labels_data;
    this->syncDData(device); // H to D
    return true;
  }
  template<template<class> class ArrayT, class TensorT>
  inline bool TensorAxisGpuClassT<ArrayT, TensorT>::storeLabelsBinary(const std::string & filename, Eigen::GpuDevice & device)
  {
    // Store the labels
    this->syncHData(device); // D to H
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    DataFile::storeDataBinary<ArrayT<TensorT>, 2>(filename + ".ta", this->getLabels());
    this->setDataStatus(false, true);
    return true;
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::GpuDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to ArrayT<TensorT>
    TensorDataGpuClassT<ArrayT, TensorT, 2> labels_converted(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    labels_converted.setData();
    labels_converted.syncDData(device);
    labels_converted.convertFromStringToTensorT(labels, device);

    // Make the labels unique
    TensorDataGpuPrimitiveT<int, 3> labels_unique_tmp(Eigen::array<Eigen::Index, 3>({ 1, (int)labels.dimension(1), (int)labels.dimension(1) }));
    labels_unique_tmp.setData();
    labels_unique_tmp.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 3>> labels_new_v1v2_prod(labels_unique_tmp.getDataPointer().get(), 1, (int)labels.dimension(1), (int)labels.dimension(1));
    // Make the indices unique
    TensorDataGpuPrimitiveT<int, 2> indices_unique(Eigen::array<Eigen::Index, 2>({ 1, (int)labels.dimension(1) }));
    indices_unique.setData();
    indices_unique.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_unique_values(indices_unique.getDataPointer().get(), 1, (int)labels.dimension(1));
    // Make the indices select
    TensorDataGpuPrimitiveT<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    indices_select.setData();
    indices_select.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), (int)labels.dimension(0), (int)labels.dimension(1));

    // Determine the unique input axis labels
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_unique_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_unique_values_bcast = indices_unique_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 5>> labels_new_v1(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_v1_bcast = labels_new_v1.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 5>> labels_new_v2(labels_converted.getDataPointer().get(), 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1));
    auto labels_new_v2_bcast = labels_new_v2.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    // Select the overlapping labels
    auto labels_new_v1v2_select = (labels_new_v1_bcast == labels_new_v2_bcast).select(indices_unique_values_bcast.constant(1), indices_unique_values_bcast.constant(0));
    // Reduct along the axis dimensions and then cum sum along the axis labels
    labels_new_v1v2_prod.device(device) = labels_new_v1v2_select.sum(Eigen::array<Eigen::Index, 2>({ 1, 3 })).clip(0, 1);
    auto labels_new_v1v2_cumsum = (labels_new_v1v2_prod.cumsum(1) * labels_new_v1v2_prod).cumsum(2) * labels_new_v1v2_prod;
    // Select the unique labels marked with a 1
    auto labels_unique_v1v2 = (labels_new_v1v2_cumsum == labels_new_v1v2_cumsum.constant(1)).select(labels_new_v1v2_cumsum.constant(1), labels_new_v1v2_cumsum.constant(0));
    // Collapse back to 1xnlabels by summing along one of the axis labels dimensions
    indices_unique_values.device(device) = labels_unique_v1v2.sum(Eigen::array<Eigen::Index, 1>({ 2 })).clip(0, 1);

    if (this->n_labels_ > 0) {
      // Determine the new labels to add to the axis 
      Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_select_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
      auto indices_select_values_bcast = indices_select_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
      // Broadcast along the labels and along the dimensions and for both the labels and the new labels
      Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 5>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_);
      auto labels_values_bcast = labels_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
      Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 5>> labels_new_values(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
      auto labels_new_values_bcast = labels_new_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
      // Select and sum along the labels and multiple along the dimensions
      auto labels_selected = (labels_values_bcast == labels_new_values_bcast).select(indices_select_values_bcast.constant(1), indices_select_values_bcast.constant(0)).sum(
        Eigen::array<Eigen::Index, 2>({ 3, 4 })).prod(Eigen::array<Eigen::Index, 1>({ 1 })); // not new > 1, new = 0
      // Invert the selection
      auto labels_new = (labels_selected > labels_selected.constant(0)).select(labels_selected.constant(0), labels_selected.constant(1)); // new = 1, not new = 0

      // Determine the new and unique labels to add to the axis
      auto labels_new_unique = labels_new * indices_unique_values;
      // Broadcast back to n_dimensions x labels
      auto labels_new_unique_bcast = labels_new_unique.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

      // Store the selection indices
      indices_select_values.device(device) = labels_new_unique_bcast;
    }
    else {
      // Broadcast back to n_dimensions x labels
      auto labels_new_unique_bcast = indices_unique_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

      // Store the selection indices
      indices_select_values.device(device) = labels_new_unique_bcast;
    }
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_select);

    // Determine the number of new labels
    TensorDataGpuPrimitiveT<int, 1> n_labels_new(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_labels_new.setData();
    n_labels_new.syncDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> n_labels_new_values(n_labels_new.getDataPointer().get());
    n_labels_new_values.device(device) = indices_select_values.sum() / n_labels_new_values.constant((int)labels.dimension(0));
    n_labels_new.syncHData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // Allocate memory for the new labels
    TensorDataGpuClassT<ArrayT, TensorT, 2> labels_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, n_labels_new.getData()(0) }));
    labels_select.setData();
    labels_select.syncDData(device);
    std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>> labels_select_ptr = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(labels_select);

    // Select the labels
    labels_converted.select(labels_select_ptr, indices_select_ptr, device);

    // Append the selected labels to the axis
    this->appendLabelsToAxis(labels_select_ptr, device);
  }
};

// Cereal registration of TensorTs: float, int, char, double, charArray8
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<char>);

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu8, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu32, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu128, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu512, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu2048, char>);
#endif
#endif //TENSORBASE_TENSORAXISGPU_H