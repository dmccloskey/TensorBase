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
#include <TensorBase/ml/TensorTableDefaultDevice.h>
#include <TensorBase/ml/TensorTableCpu.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /// The erasure interface to Tensor Table
  template<typename DeviceT>
  class TensorTableConcept {
  public:
    TensorTableConcept() = default;
    virtual ~TensorTableConcept() = default;

    template<typename DeviceTOther>
    inline bool operator==(const TensorTableConcept<DeviceTOther>& other) const
    {
      bool meta_equal = (this->getId() == other.getId() && this->getName() == other.getName());
      auto compare_maps = [](auto lhs, auto rhs) {return *(lhs.second.get()) == *(rhs.second.get()); };
      auto axes = this->getAxes();
      if (axes.size() != other.getAxes().size()) return false;
      bool axes_equal = std::equal(axes.begin(), axes.end(), other.getAxes().begin(), compare_maps);
      auto indices = this->getIndices();
      if (indices.size() != other.getIndices().size()) return false;
      bool indices_equal = std::equal(indices.begin(), indices.end(), other.getIndices().begin(), compare_maps);
      auto indices_view = this->getIndicesView();
      if (indices_view.size() != other.getIndicesView().size()) return false;
      bool indices_view_equal = std::equal(indices_view.begin(), indices_view.end(), other.getIndicesView().begin(), compare_maps);
      auto is_shardable = this->getShardId();
      if (indices_view.size() != other.getIndicesView().size()) return false;
      bool is_shardable_equal = std::equal(is_shardable.begin(), is_shardable.end(), other.getShardId().begin(), compare_maps);
      auto in_memory = this->getNotInMemory();
      if (in_memory.size() != other.getNotInMemory().size()) return false;
      bool in_memory_equal = std::equal(in_memory.begin(), in_memory.end(), other.getNotInMemory().begin(), compare_maps);
      auto is_modified = this->getIsModified();
      if (is_modified.size() != other.getIsModified().size()) return false;
      bool is_modified_equal = std::equal(is_modified.begin(), is_modified.end(), other.getIsModified().begin(), compare_maps);
      return meta_equal && axes_equal && indices_equal && indices_view_equal && is_shardable_equal
        && in_memory_equal && is_modified_equal;
    }

    inline bool operator!=(const TensorTableConcept& other) const
    {
      return !(*this == other);
    }

    virtual int getId() const = 0;
    virtual std::string getName() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>>& getAxes() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndices() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndicesView() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsModified() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getNotInMemory() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getShardId() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getShardIndices() = 0;
    virtual std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> getAxes() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndices() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndicesView() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIsModified() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getNotInMemory() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getShardId() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getShardIndices() const = 0;
    virtual void resetIndicesView(const std::string& axis_name, DeviceT& device) = 0;
    virtual void makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;
    virtual int getDimFromAxisName(const std::string& axis_name) = 0;

    /*
    All LabelT and DeviceT combos of `selectIndicesView`
    */ 
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels, DeviceT& device) = 0;
#endif

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
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
    virtual void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) = 0;
#endif

    /*
    All LabelT, TensorT, and DeviceT combos of `sortIndicesView`
    */ 
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<char, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<float, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<double, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) = 0;
#endif

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
    virtual void updateTensorDataValues(const std::shared_ptr<int>& values_new, std::shared_ptr<int>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<float>& values_new, std::shared_ptr<float>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<double>& values_new, std::shared_ptr<double>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<char>& values_new, std::shared_ptr<char>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<TensorArray<char>>& values_new, std::shared_ptr<TensorArray<char>>& values_old, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void updateTensorDataValues(const std::shared_ptr<TensorArrayGpu8<char>>& values_new, std::shared_ptr<TensorArrayGpu8<char>>& values_old, DeviceT& device) = 0;
#endif
    virtual void updateTensorDataValues(const std::shared_ptr<int>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<float>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<double>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<char>& values_new, DeviceT& device) = 0;
    virtual void updateTensorDataValues(const std::shared_ptr<TensorArray<char>>& values_new, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void updateTensorDataValues(const std::shared_ptr<TensorArrayGpu8<char>>& values_new, DeviceT& device) = 0;
#endif

    /*
    All LabelsT, TensorT, and DeviceT combos of `appendToAxis`
    */
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
#if COMPILE_WITH_CUDA
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
#endif

    /*
    All LabelsT, TensorT, and DeviceT combos of `deleteFromAxis`
    */
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) = 0;
    virtual void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) = 0;
#endif

    /*
    All LabelsT, TensorT, and DeviceT combos of `insertIntoAxis`
    */
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
#if COMPILE_WITH_CUDA
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
    virtual void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) = 0;
#endif

    /*
    All TensorT and DeviceT combos of `updateTensorDataConstant`
    */
    virtual void updateTensorDataConstant(const std::shared_ptr<TensorData<int, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<int, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConstant(const std::shared_ptr<TensorData<float, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<float, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConstant(const std::shared_ptr<TensorData<double, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<double, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConstant(const std::shared_ptr<TensorData<char, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<char, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataConstant(const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorArray<char>, DeviceT, 2>>& values_old, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void updateTensorDataConstant(const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorArrayGpu8<char>, DeviceT, 2>>& values_old, DeviceT& device) = 0;
#endif

    /*
    All TensorT and DeviceT combos of `updateTensorDataConstant`
    */
    virtual void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<int, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<float, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<double, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<char, DeviceT, 2>>& values_old, DeviceT& device) = 0;
    virtual void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorArray<char>, DeviceT, 2>>& values_old, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorArrayGpu8<char>, DeviceT, 2>>& values_old, DeviceT& device) = 0;
#endif

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) { }
  };

  /// The erasure wrapper around the Tensor Table interface
  template<typename T, typename DeviceT>
  class TensorTableWrapper : public TensorTableConcept<DeviceT> {
    std::shared_ptr<T> tensor_table_;
  public:
    TensorTableWrapper(const std::shared_ptr<T>& tensor_table) : tensor_table_(tensor_table) {};
    TensorTableWrapper() = default;
    ~TensorTableWrapper() = default;
    int getId() const { return tensor_table_->getId(); }
    std::string getName() const { return tensor_table_->getName(); };
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>>& getAxes() { return tensor_table_->getAxes(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndices() { return tensor_table_->getIndices(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIndicesView() { return tensor_table_->getIndicesView(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getIsModified() { return tensor_table_->getIsModified(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getNotInMemory() { return tensor_table_->getNotInMemory(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getShardId() { return tensor_table_->getShardId(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>>& getShardIndices() { return tensor_table_->getShardIndices(); };
    std::map<std::string, std::shared_ptr<TensorAxisConcept<DeviceT>>> getAxes() const { return tensor_table_->getAxes(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndices() const { return tensor_table_->getIndices(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIndicesView() const { return tensor_table_->getIndicesView(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getIsModified() const { return tensor_table_->getIsModified(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getNotInMemory() const { return tensor_table_->getNotInMemory(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getShardId() const { return tensor_table_->getShardId(); };
    std::map<std::string, std::shared_ptr<TensorData<int, DeviceT, 1>>> getShardIndices() const { return tensor_table_->getShardIndices(); };
    void resetIndicesView(const std::string& axis_name, DeviceT& device) { tensor_table_->resetIndicesView(axis_name, device); };
    void makeIndicesFromIndicesView(const std::string & axis_name, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) { 
      tensor_table_->makeIndicesFromIndicesView(axis_name, indices, device);
    };
    int getDimFromAxisName(const std::string& axis_name) { return tensor_table_->getDimFromAxisName(axis_name); };

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
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels, DeviceT& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
#if COMPILE_WITH_CUDA
    void selectIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels, DeviceT& device) {
      tensor_table_->selectIndicesView(axis_name, dimension_index, select_labels, device);
    };
#endif

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
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
#if COMPILE_WITH_CUDA
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<int, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<int, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<float, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<double, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
    void whereIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels,
      const std::shared_ptr<TensorData<char, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor, const logicalModifiers::logicalModifier& modifier,
      const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator, DeviceT& device) {
      tensor_table_->whereIndicesViewConcept(axis_name, dimension_index, select_labels, values, comparitor, modifier, within_continuator, prepend_continuator, device);
    };
#endif

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
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) {
      tensor_table_->sortIndicesView(axis_name, dimension_index, select_labels, order_by, device);
    };
#if COMPILE_WITH_CUDA
    void sortIndicesView(const std::string& axis_name, const int& dimension_index, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& select_labels, const sortOrder::order& order_by, DeviceT& device) {
      tensor_table_->sortIndicesView(axis_name, dimension_index, select_labels, order_by, device);
    };
#endif

    void selectTensorData(DeviceT& device) {
      tensor_table_->selectTensorData(device);
    };

    void sortTensorData(DeviceT& device) {
      tensor_table_->sortTensorData(device);
    };

    void updateTensorDataValues(const std::shared_ptr<int>& values_new, std::shared_ptr<int>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, values_old, device);
    };
    void updateTensorDataValues(const std::shared_ptr<float>& values_new, std::shared_ptr<float>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, values_old, device);
    };
    void updateTensorDataValues(const std::shared_ptr<double>& values_new, std::shared_ptr<double>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, values_old, device);
    };
    void updateTensorDataValues(const std::shared_ptr<char>& values_new, std::shared_ptr<char>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, values_old, device);
    };
    void updateTensorDataValues(const std::shared_ptr<TensorArray<char>>& values_new, std::shared_ptr<TensorArray<char>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, values_old, device);
    };
#if COMPILE_WITH_CUDA
    void updateTensorDataValues(const std::shared_ptr<TensorArrayGpu8<char>>& values_new, std::shared_ptr<TensorArrayGpu8<char>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, values_old, device);
    };
#endif
    void updateTensorDataValues(const std::shared_ptr<int>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, device);
    };
    void updateTensorDataValues(const std::shared_ptr<float>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, device);
    };
    void updateTensorDataValues(const std::shared_ptr<double>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, device);
    };
    void updateTensorDataValues(const std::shared_ptr<char>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, device);
    };
    void updateTensorDataValues(const std::shared_ptr<TensorArray<char>>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, device);
    };
#if COMPILE_WITH_CUDA
    void updateTensorDataValues(const std::shared_ptr<TensorArrayGpu8<char>>& values_new, DeviceT& device) {
      tensor_table_->updateTensorDataValuesConcept(values_new, device);
    };
#endif

    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
#if COMPILE_WITH_CUDA
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
    void appendToAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->appendToAxisConcept(axis_name, labels, values, indices, device);
    };
#endif

    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) {
      tensor_table_->deleteFromAxis(axis_name, indices, device);
    }
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
#if COMPILE_WITH_CUDA
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
    void deleteFromAxis(const std::string& axis_name, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, DeviceT& device) {
      tensor_table_->deleteFromAxisConcept(axis_name, indices, labels, values, device);
    };
#endif

    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArray<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
#if COMPILE_WITH_CUDA
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<int>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<float>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<double>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<char>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
    void insertIntoAxis(const std::string & axis_name, const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, std::shared_ptr<TensorArrayGpu8<char>>& values, const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device) {
      tensor_table_->insertIntoAxisConcept(axis_name, labels, values, indices, device);
    };
#endif

    void updateTensorDataConstant(const std::shared_ptr<TensorData<int, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<int, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConstantConcept(values_new, values_old, device);
    };
    void updateTensorDataConstant(const std::shared_ptr<TensorData<float, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<float, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConstantConcept(values_new, values_old, device);
    };
    void updateTensorDataConstant(const std::shared_ptr<TensorData<double, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<double, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConstantConcept(values_new, values_old, device);
    };
    void updateTensorDataConstant(const std::shared_ptr<TensorData<char, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<char, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConstantConcept(values_new, values_old, device);
    };
    void updateTensorDataConstant(const std::shared_ptr<TensorData<TensorArray<char>, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorArray<char>, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConstantConcept(values_new, values_old, device);
    };
#if COMPILE_WITH_CUDA
    void updateTensorDataConstant(const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 1>>& values_new, std::shared_ptr<TensorTable<TensorArrayGpu8<char>, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataConstantConcept(values_new, values_old, device);
    };
#endif

    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<int, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataFromSparseTensorTableConcept(values_old, device);
    };
    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<float, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataFromSparseTensorTableConcept(values_old, device);
    };
    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<double, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataFromSparseTensorTableConcept(values_old, device);
    };
    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<char, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataFromSparseTensorTableConcept(values_old, device);
    };
    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorArray<char>, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataFromSparseTensorTableConcept(values_old, device);
    };
#if COMPILE_WITH_CUDA
    void updateTensorDataFromSparseTensorTable(const std::shared_ptr<TensorTable<TensorArrayGpu8<char>, DeviceT, 2>>& values_old, DeviceT& device) {
      tensor_table_->updateTensorDataFromSparseTensorTableConcept(values_old, device);
    };
#endif

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorTableConcept<DeviceT>>(this), tensor_table_);
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<int, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<float, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<double, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<char, 4>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 1>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 2>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 3>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableDefaultDevice<TensorBase::TensorArray8<char>, 4>, Eigen::DefaultDevice>);

CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<int, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<float, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<double, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<char, 4>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 1>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 2>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 3>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorTableWrapper<TensorBase::TensorTableCpu<TensorBase::TensorArray8<char>, 4>, Eigen::ThreadPoolDevice>);
#endif //TENSORBASE_TENSORTABLECONCEPT_H