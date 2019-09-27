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
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice& device) override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
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
    this->tensor_dimension_labels_.reset(new TensorDataGpuPrimitiveT<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_.reset(new TensorDataGpuPrimitiveT<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataGpuPrimitiveT<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    axis_size.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    auto indices_view_norm = (indices_values.cast<float>() / (indices_values.cast<float>() + indices_values.cast<float>().constant(1e-12))).cast<int>();
    axis_size_value.device(device) = indices_view_norm.sum();
    axis_size.syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // allocate memory for the new labels
    TensorDataGpuPrimitiveT<TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> new_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(new_labels);
    new_labels_ptr->syncHAndDData(device);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataGpuPrimitiveT<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_select);
    indices_select_ptr->syncHAndDData(device);
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

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataGpuPrimitiveT<TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    std::shared_ptr<TensorDataGpuPrimitiveT<TensorT, 2>> labels_concat_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(labels_concat);
    labels_concat_ptr->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_concat_values(labels_concat_ptr->getDataPointer().get(), labels_concat_ptr->getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);

    // Move over the new labels
    this->tensor_dimension_labels_ = labels_concat_ptr;
  }
  template<typename TensorT>
  inline void TensorAxisGpuPrimitiveT<TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataGpuPrimitiveT<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), (int)this->getNLabels() }));
    indices_sort_tmp.setData();
    indices_sort_tmp.syncHAndDData(device);

    // create a dummy index along the dimension
    TensorDataGpuPrimitiveT<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    indices_dimension.syncHAndDData(device);
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
    this->syncHAndDData(device); // H to D
    return true;
  }
  template<typename TensorT>
  inline bool TensorAxisGpuPrimitiveT<TensorT>::storeLabelsBinary(const std::string & filename, Eigen::GpuDevice & device)
  {
    // Store the labels
    this->syncHAndDData(device); // D to H
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    DataFile::storeDataBinary<TensorT, 2>(filename + ".ta", this->getLabels());
    this->setDataStatus(false, true);
    return true;
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
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice& device) override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxis<TensorT, Eigen::GpuDevice>>(this));
    }
  };
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::setLabels(const Eigen::Tensor<ArrayT<TensorT>, 2>& labels) {
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_.reset(new TensorDataGpuClassT<ArrayT, TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_.reset(new TensorDataGpuClassT<ArrayT, TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>>& labels_select, Eigen::GpuDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataGpuPrimitiveT<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    axis_size.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    auto indices_view_norm = (indices_values.cast<float>() / (indices_values.cast<float>() + indices_values.cast<float>().constant(1e-12))).cast<int>();
    axis_size_value.device(device) = indices_view_norm.sum();
    axis_size.syncHAndDData(device);
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);

    // allocate memory for the new labels
    TensorDataGpuClassT<ArrayT, TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 2>> new_labels_ptr = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(new_labels);
    new_labels_ptr->syncHAndDData(device);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataGpuPrimitiveT<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(indices_select);
    indices_select_ptr->syncHAndDData(device);
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

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataGpuClassT<ArrayT, TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    std::shared_ptr<TensorDataGpuClassT<ArrayT, TensorT, 2>> labels_concat_ptr = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 2>>(labels_concat);
    labels_concat_ptr->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> labels_concat_values(labels_concat_ptr->getDataPointer().get(), labels_concat_ptr->getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);

    // Move over the new labels
    this->tensor_dimension_labels_ = labels_concat_ptr;
  }
  template<template<class> class ArrayT, class TensorT>
  inline void TensorAxisGpuClassT<ArrayT, TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>>& indices_sort, Eigen::GpuDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataGpuPrimitiveT<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), (int)this->getNLabels() }));
    indices_sort_tmp.setData();
    indices_sort_tmp.syncHAndDData(device);

    // create a dummy index along the dimension
    TensorDataGpuPrimitiveT<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    indices_dimension.syncHAndDData(device);
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
    this->syncHAndDData(device); // H to D
    return true;
  }
  template<template<class> class ArrayT, class TensorT>
  inline bool TensorAxisGpuClassT<ArrayT, TensorT>::storeLabelsBinary(const std::string & filename, Eigen::GpuDevice & device)
  {
    // Store the labels
    this->syncHAndDData(device); // D to H
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    DataFile::storeDataBinary<ArrayT<TensorT>, 2>(filename + ".ta", this->getLabels());
    this->setDataStatus(false, true);
    return true;
  }
};

// Cereal registration of TensorTs: float, int, char, double, charArray8
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuPrimitiveT<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisGpuClassT<TensorBase::TensorArrayGpu8, char>);
#endif
#endif //TENSORBASE_TENSORAXISGPU_H