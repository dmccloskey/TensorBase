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
#include <map>

namespace TensorBase
{
  /// The erasure interface to Tensor Table
  class TensorTableConcept {
  public:
    virtual std::string getName() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() = 0;

    // All LabelT and DeviceT combos of `selectIndicesView`
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<int>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<float>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<double>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<char>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) = 0;
    
    // All DeviceT combos of `zeroIndicesView`
    virtual void zeroIndicesView(const std::string& axis_name, const Eigen::DefaultDevice& device) = 0;
//    virtual void zeroIndicesView(const std::string& axis_name, const Eigen::ThreadPoolDevice& device) = 0;
//#if COMPILE_WITH_CUDA
//    virtual void zeroIndicesView(const std::string& axis_name, const Eigen::GpuDevice& device) = 0;
//#endif

    // TODO:: all DeviceT combos of `resetIndicesView`
  };

  /// The erasure wrapper around the Tensor Table interface
  template<typename T>
  class TensorTableWrapper : public TensorTableConcept {
    std::shared_ptr<T> tensor_table_;
  public:
    TensorTableWrapper(const std::shared_ptr<T>& tensor_table) : tensor_table_(tensor_table) {};
    std::string getName() const { return tensor_table_->getName(); };
    std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() { return tensor_table_->getAxes(); };

    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<int>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<float>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<double>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);
    };
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<char>& select_labels_data, const int& n_labels, const Eigen::DefaultDevice& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels_data, n_labels, device);
    };

    void zeroIndicesView(const std::string& axis_name, const Eigen::DefaultDevice& device) { tensor_table_->zeroIndicesView(axis_name, device); }
//    void zeroIndicesView(const std::string& axis_name, const Eigen::ThreadPoolDevice& device) { tensor_table_->zeroIndicesView(axis_name, device); }
//#if COMPILE_WITH_CUDA
//    void zeroIndicesView(const std::string& axis_name, const Eigen::GpuDevice& device) { tensor_table_->zeroIndicesView(axis_name, device); }
//#endif
  };
};
#endif //TENSORBASE_TENSORTABLECONCEPT_H