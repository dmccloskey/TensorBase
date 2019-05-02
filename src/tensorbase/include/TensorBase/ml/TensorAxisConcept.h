/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPT_H
#define TENSORBASE_TENSORAXISCONCEPT_H

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

    // All TensorT combos of `getLabelsDatapointer`
    virtual void getLabelsDataPointer(std::shared_ptr<int>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<float>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<double>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<char>& data_copy) = 0;
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

    void getLabelsDataPointer(std::shared_ptr<int>& data_copy) {
      tensor_axis_->getLabelsDataPointer<int>(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<float>& data_copy) {
      tensor_axis_->getLabelsDataPointer<float>(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<double>& data_copy) {
      tensor_axis_->getLabelsDataPointer<double>(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<char>& data_copy) {
      tensor_axis_->getLabelsDataPointer<char>(data_copy);
    };
  };
};
#endif //TENSORBASE_TENSORAXISCONCEPT_H