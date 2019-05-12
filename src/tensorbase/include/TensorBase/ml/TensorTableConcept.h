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
  class TensorTableConcept {
  public:
    virtual std::string getName() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() = 0;

    /*
    All LabelT and DeviceT combos of `selectIndicesView`
    */ 
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) = 0;
    // TODO: all other DeviceT combos of `selectIndicesView`

    /*
    All DeviceT combos of `zeroIndicesView`
    */
    virtual void zeroIndicesView(const std::string& axis_name, Eigen::DefaultDevice& device) = 0;
//    virtual void zeroIndicesView(const std::string& axis_name, const Eigen::ThreadPoolDevice& device) = 0;
//#if COMPILE_WITH_CUDA
//    virtual void zeroIndicesView(const std::string& axis_name, const Eigen::GpuDevice& device) = 0;
//#endif
    // TODO:: all DeviceT combos of `resetIndicesView`

    /*
    All LabelT, TensorT, and DeviceT combos of `whereIndicesView`
    */
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) = 0;
    // TODO: all other LabelsT, TensorT, and DeviceT combos of `whereIndicesView` 

    /*
    All LabelT, TensorT, and DeviceT combos of `sortIndicesView`
    */ 
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const int& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const char& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const float& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const double& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device) = 0;
    // TODO: all other LabelsT and DeviceT combos of `sortIndicesView` 

    /*
    All DeviceT combos of `selectTensorData`
    */
    virtual void selectTensorData(Eigen::DefaultDevice& device) = 0;
    // TODO: all other LabelsT and DeviceT combos of `selectTensorData` 

    /*
    All DeviceT combos of `sortTensorData`
    */
    virtual void sortTensorData(Eigen::DefaultDevice& device) = 0;
    // TODO: all other LabelsT and DeviceT combos of `sortTensorData` 
  };

  /// The erasure wrapper around the Tensor Table interface
  template<typename T>
  class TensorTableWrapper : public TensorTableConcept {
    std::shared_ptr<T> tensor_table_;
  public:
    TensorTableWrapper(const std::shared_ptr<T>& tensor_table) : tensor_table_(tensor_table) {};
    std::string getName() const { return tensor_table_->getName(); };
    std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() { return tensor_table_->getAxes(); };

    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels, Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };

    void zeroIndicesView(const std::string& axis_name, Eigen::DefaultDevice& device) { tensor_table_->zeroIndicesView(axis_name, device); }
//    void zeroIndicesView(const std::string& axis_name, const Eigen::ThreadPoolDevice& device) { tensor_table_->zeroIndicesView(axis_name, device); }
//#if COMPILE_WITH_CUDA
//    void zeroIndicesView(const std::string& axis_name, const Eigen::GpuDevice& device) { tensor_table_->zeroIndicesView(axis_name, device); }
//#endif

    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device){ 
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, Eigen::DefaultDevice, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, Eigen::DefaultDevice, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, Eigen::DefaultDevice& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };

    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const int& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, label, order_by, device);
    };
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const char& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, label, order_by, device);
    };
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const float& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, label, order_by, device);
    };
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const double& label, const sortOrder::order& order_by, Eigen::DefaultDevice& device){
      tensor_table_->sortIndicesView(axis_name, dimension_index, label, order_by, device);
    };

    void selectTensorData(Eigen::DefaultDevice& device) {
      tensor_table_->selectTensorData(device);
    };

    void sortTensorData(Eigen::DefaultDevice& device) {
      tensor_table_->sortTensorData(device);
    };
  };
};
#endif //TENSORBASE_TENSORTABLECONCEPT_H