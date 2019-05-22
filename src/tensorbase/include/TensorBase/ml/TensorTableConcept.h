/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPT_H
#define TENSORBASE_TENSORTABLECONCEPT_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>

namespace TensorBase
{
  /// The erasure interface to Tensor Table
  template<typename DeviceT>
  class TensorTableConcept {
  public:
    virtual std::string getName() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>>& getAxes() = 0;

    /*
    All LabelT and DeviceT combos of `selectIndicesView`
    */ 
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels, DeviceT& device) = 0;

    /*
    All DeviceT combos of `zeroIndicesView`
    */
    virtual void zeroIndicesView(const std::string& axis_name, DeviceT& device) = 0;

    /*
    All LabelT, TensorT, and DeviceT combos of `whereIndicesView`
    */
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;

    /*
    All LabelT, TensorT, and DeviceT combos of `sortIndicesView`
    */ 
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;

    /*
    All DeviceT combos of `selectTensorData`
    */
    virtual void selectTensorData(DeviceT& device) = 0;

    /*
    All DeviceT combos of `sortTensorData`
    */
    virtual void sortTensorData(DeviceT& device) = 0;

    /*
    All TensorT and DeviceT combos of `updateTensorData`
    */
    virtual void updateTensorDataConcept(const std::shared_ptr<int>& values_new, std::shared_ptr<int>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<float>& values_new, std::shared_ptr<float>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<double>& values_new, std::shared_ptr<double>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<char>& values_new, std::shared_ptr<char>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<int>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<float>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<double>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataConcept(const std::shared_ptr<char>& values_new, DeviceT& device) = 0;
  };

  /// The erasure wrapper around the Tensor Table interface
  template<typename T, typename DeviceT>
  class TensorTableWrapper : public TensorTableConcept<DeviceT> {
    std::shared_ptr<T> tensor_table_;
  public:
    TensorTableWrapper(const std::shared_ptr<T>& tensor_table) : tensor_table_(tensor_table) {};
    std::string getName() const { return tensor_table_->getName(); };
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>>& getAxes() { return tensor_table_->getAxes(); };

    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels, DeviceT& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, DeviceT, 1>>& select_labels, DeviceT& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, DeviceT, 1>>& select_labels, DeviceT& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels, DeviceT& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };

    void zeroIndicesView(const std::string& axis_name, DeviceT& device) { tensor_table_->zeroIndicesView(axis_name, device); }

    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device){ 
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };

    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, select_labels, order_by, device);
    };
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, select_labels, order_by, device);
    };
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, select_labels, order_by, device);
    };
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, select_labels, order_by, device);
    };

    void selectTensorData(DeviceT& device) {
      tensor_table_->selectTensorData(device);
    };

    void sortTensorData(DeviceT& device) {
      tensor_table_->sortTensorData(device);
    };

    void updateTensorDataConcept(const std::shared_ptr<int>& values_new, std::shared_ptr<int>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, values_old, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<float>& values_new, std::shared_ptr<float>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, values_old, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<double>& values_new, std::shared_ptr<double>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, values_old, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<char>& values_new, std::shared_ptr<char>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, values_old, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<int>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<float>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<double>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, device);
    };
    void updateTensorDataConcept(const std::shared_ptr<char>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataConcept(values_new, device);
    };
  };
};
#endif //TENSORBASE_TENSORTABLECONCEPT_H