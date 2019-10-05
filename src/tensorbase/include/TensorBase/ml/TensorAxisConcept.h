/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCONCEPT_H
#define TENSORBASE_TENSORAXISCONCEPT_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorArrayGpu.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorAxis.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /// The erasure interface for TensorAxis
  template<typename DeviceT>
  class TensorAxisConcept {
  public:
    TensorAxisConcept() = default;
    virtual ~TensorAxisConcept() = default;

    inline bool operator==(const TensorAxisConcept& other) const
    {
      bool meta_equal = (this->getId() == other.getId() && this->getName() == other.getName() &&
        this->getNLabels() == other.getNLabels(), this->getNDimensions() == other.getNDimensions());
      // bool dimension_names_equal = (this->getDimensions() == other.getDimensions()); TODO: constant correctness
      return meta_equal; // && dimension_names_equal;
    }

    inline bool operator!=(const TensorAxisConcept& other) const
    {
      return !(*this == other);
    }

    virtual int getId() const = 0;
    virtual std::string getName() const = 0;
    virtual size_t getNLabels() const = 0;
    virtual size_t getNDimensions() const = 0;
    virtual Eigen::TensorMap<Eigen::Tensor<std::string, 1>> getDimensions() = 0;
    virtual void setLabels() = 0;

    // All TensorT combos of `getLabelsDatapointer`
    virtual void getLabelsDataPointer(std::shared_ptr<int>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<float>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<double>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<char>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray8<char>>& data_copy) = 0;
#if COMPILE_WITH_CUDA
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu8<char>>& data_copy) = 0;
#endif

    // All DeviceT combos of `deleteFromAxis`
    virtual void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;

    // All TensorT DeviceT combos of `selectFromAxis`
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
#endif

    // All TensorT and DeviceT combos of `appendLabelsToAxis`
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
#endif

    // All DeviceT combos of tensorDataWrappers
    virtual bool syncHAndDData(DeviceT& device) = 0;
    virtual void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) = 0;
    virtual std::pair<bool, bool> getDataStatus() = 0;

    // All DeviceT combos of `sortLabels`
    virtual void sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;

    // All DeviceT combos of `loadLabelsBinary`
    virtual bool loadLabelsBinary(const std::string& filename, DeviceT& device) = 0;

    // All DeviceT combos of `storeLabelsBinary`
    virtual bool storeLabelsBinary(const std::string& filename, DeviceT& device) = 0;

    virtual std::vector<std::string> getLabelsAsStrings(DeviceT& device) = 0;
    virtual std::vector<std::string> getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) = 0;

    // All DeviceT combos of `appendLabelsToAxisFromCsv`
    virtual void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, DeviceT& device) = 0;

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) { }
  };

  /// The erasure wrapper around the Tensor Axis interface
  template<typename T, typename DeviceT>
  class TensorAxisWrapper : public TensorAxisConcept<DeviceT> {
    std::shared_ptr<T> tensor_axis_;
  public:
    TensorAxisWrapper(const std::shared_ptr<T>& tensor_axis) : tensor_axis_(tensor_axis) {};
    TensorAxisWrapper() = default;
    ~TensorAxisWrapper() = default;

    int getId() const { return tensor_axis_->getId(); };
    std::string getName() const { return tensor_axis_->getName(); };
    size_t getNLabels() const { return tensor_axis_->getNLabels(); };
    size_t getNDimensions() const { return tensor_axis_->getNDimensions(); };
    Eigen::TensorMap<Eigen::Tensor<std::string, 1>> getDimensions() { return tensor_axis_->getDimensions(); };
    void setLabels() { tensor_axis_->setLabels(); }

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
    void getLabelsDataPointer(std::shared_ptr<TensorArray8<char>>& data_copy) {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
#if COMPILE_WITH_CUDA
    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu8<char>>& data_copy) {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
#endif

    void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) {
      tensor_axis_->deleteFromAxis(indices, device);
    };

    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_select, DeviceT& device) {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels_select, DeviceT& device) {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels_select, DeviceT& device) {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels_select, DeviceT& device) {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels_select, DeviceT& device) {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
#if COMPILE_WITH_CUDA
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels_select, DeviceT& device) {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
#endif

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
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
#if COMPILE_WITH_CUDA
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
#endif

    bool syncHAndDData(DeviceT& device) { return  tensor_axis_->syncHAndDData(device); };  
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_axis_->setDataStatus(h_data_updated, d_data_updated); } 
    std::pair<bool, bool> getDataStatus() { return  tensor_axis_->getDataStatus(); };  

    void sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) {
      tensor_axis_->sortLabels(indices, device);
    };

    bool loadLabelsBinary(const std::string& filename, DeviceT& device) {
      return tensor_axis_->loadLabelsBinary(filename, device);
    }

    bool storeLabelsBinary(const std::string& filename, DeviceT& device) {
      return tensor_axis_->storeLabelsBinary(filename, device);
    }

    std::vector<std::string> getLabelsAsStrings(DeviceT& device) {
      return tensor_axis_->getLabelsAsStrings(device);
    }
    std::vector<std::string> getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) {
      return tensor_axis_->getLabelsAsStrings(offset, span);
    }

    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, DeviceT& device) {
      tensor_axis_->appendLabelsToAxisFromCsv(labels, device);
    }

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxisConcept<DeviceT>>(this), tensor_axis_);
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<int>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<float>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<double>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<char>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray8<char>>, Eigen::DefaultDevice>);

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<int>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<float>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<double>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<char>, Eigen::ThreadPoolDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisWrapper<TensorBase::TensorAxisCpu<TensorBase::TensorArray8<char>>, Eigen::ThreadPoolDevice>);
#endif //TENSORBASE_TENSORAXISCONCEPT_H