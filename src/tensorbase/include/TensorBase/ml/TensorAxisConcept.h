/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPT_H
#define TENSORBASE_TENSORAXISCONCEPT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxis.h>

namespace TensorBase
{
  /// The erasure interface for TensorAxis
  template<typename DeviceT>
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

    // All DeviceT combos of `deleteFromAxis`
    virtual void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;

    // All TensorT and DeviceT combos of `appendLabelsToAxis`
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, DeviceT& device) = 0;

    // All DeviceT combos of tensorDataWrappers
    virtual bool syncHAndDData(DeviceT& device) = 0;
    virtual void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) = 0;
    virtual std::pair<bool, bool> getDataStatus() = 0;

    // All DeviceT combos of `sortLabels`
    virtual void sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;
  };

  /// The erasure wrapper around the Tensor Axis interface
  template<typename T, typename DeviceT>
  class TensorAxisWrapper : public TensorAxisConcept<DeviceT> {
    std::shared_ptr<T> tensor_axis_;
  public:
    TensorAxisWrapper(const std::shared_ptr<T>& tensor_axis) : tensor_axis_(tensor_axis) {};
    std::string getName() const { return tensor_axis_->getName(); };
    size_t getNLabels() const { return tensor_axis_->getNLabels(); };
    size_t getNDimensions() const { return tensor_axis_->getNDimensions(); };
    Eigen::Tensor<std::string, 1>& getDimensions() { return tensor_axis_->getDimensions(); };

    void getLabelsDataPointer(std::shared_ptr<int>& data_copy) {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<float>& data_copy) {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<double>& data_copy) {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<char>& data_copy) {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };

    void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) {
      tensor_axis_->deleteFromAxis(indices, device);
    };

    void appendLabelsToAxis(const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };

    bool syncHAndDData(DeviceT& device) { return  tensor_axis_->syncHAndDData(device); };  
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_axis_->setDataStatus(h_data_updated, d_data_updated); } 
    std::pair<bool, bool> getDataStatus() { return  tensor_axis_->getDataStatus(); };  

    void sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) {
      tensor_axis_->sortLabels(indices, device);
    };
  };
};
#endif //TENSORBASE_TENSORAXISCONCEPT_H