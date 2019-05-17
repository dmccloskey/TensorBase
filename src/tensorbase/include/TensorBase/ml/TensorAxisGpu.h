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

namespace TensorBase
{
  template<typename TensorT>
  class TensorAxisGpu : public TensorAxis<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorAxisGpu() = default;  ///< Default constructor
    TensorAxisGpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
    ~TensorAxisGpu() = default; ///< Default destructor
    void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) override;
    void deleteFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice& device) override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device) override;
  };
  template<typename TensorT>
  TensorAxisGpu<TensorT>::TensorAxisGpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    setName(name);
    setDimensionsAndLabels(dimensions, labels);
  }
  template<typename TensorT>
  void TensorAxisGpu<TensorT>::setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    assert(labels.dimension(0) == dimensions.dimension(0));
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_.reset(new TensorDataGpu<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->tensor_dimension_names_ = dimensions;
    this->setNDimensions(labels.dimension(0));
    this->setNLabels(labels.dimension(1));
  };

  template<typename TensorT>
  inline void TensorAxisGpu<TensorT>::deleteFromAxis(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& indices, Eigen::GpuDevice& device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataGpu<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
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
    TensorDataGpu<TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> new_labels_ptr = std::make_shared<TensorDataGpu<TensorT, 2>>(new_labels);
    new_labels_ptr->syncHAndDData(device);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataGpu<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> indices_select_ptr = std::make_shared<TensorDataGpu<int, 2>>(indices_select);
    indices_select_ptr->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select_ptr->getDataPointer().get(), indices_select_ptr->getDimensions());
    indices_select_values.device(device) = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, 1 }));

    // perform the reduction on the labels and update the axis attributes
    this->tensor_dimension_labels_->select(new_labels_ptr, indices_select_ptr, device);
    this->tensor_dimension_labels_ = new_labels_ptr;
    this->setNLabels(axis_size.getData()(0));
  }

  template<typename TensorT>
  inline void TensorAxisGpu<TensorT>::appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& labels, Eigen::GpuDevice & device)
  {
    assert(labels->getDimensions().at(0) == this->n_dimensions_);

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataGpu<TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    std::shared_ptr<TensorDataGpu<TensorT, 2>> labels_concat_ptr = std::make_shared<TensorDataGpu<TensorT, 2>>(labels_concat);
    labels_concat_ptr->syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_concat_values(labels_concat_ptr->getDataPointer().get(), labels_concat_ptr->getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);

    // Move over the new labels
    this->tensor_dimension_labels_ = labels_concat_ptr;
  }
};
#endif
#endif //TENSORBASE_TENSORAXISGPU_H