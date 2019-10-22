/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONCONCEPT_H
#define TENSORBASE_TENSORDIMENSIONCONCEPT_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorArrayGpu.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorDimension.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /// The erasure interface for TensorDimension
  template<typename DeviceT>
  class TensorDimensionConcept {
  public:
    TensorDimensionConcept() = default;
    virtual ~TensorDimensionConcept() = default;

    inline bool operator==(const TensorDimensionConcept& other) const
    {
      bool meta_equal = (this->getId() == other.getId() && this->getName() == other.getName() &&
        this->getNLabels() == other.getNLabels());
      return meta_equal;
    }

    inline bool operator!=(const TensorDimensionConcept& other) const
    {
      return !(*this == other);
    }

    virtual int getId() const = 0;
    virtual std::string getName() const = 0;
    virtual size_t getNLabels() const = 0;

//    // All TensorT combos of `getLabelsDatapointer`
//    virtual void getLabelsDataPointer(std::shared_ptr<int>& data_copy) = 0;
//    virtual void getLabelsDataPointer(std::shared_ptr<float>& data_copy) = 0;
//    virtual void getLabelsDataPointer(std::shared_ptr<double>& data_copy) = 0;
//    virtual void getLabelsDataPointer(std::shared_ptr<char>& data_copy) = 0;
//    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray8<char>>& data_copy) = 0;
//#if COMPILE_WITH_CUDA
//    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu8<char>>& data_copy) = 0;
//#endif

    // All DeviceT combos of tensorDataWrappers
    virtual bool syncHAndDData(DeviceT& device) = 0;
    virtual void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) = 0;
    virtual std::pair<bool, bool> getDataStatus() = 0;

    // All DeviceT combos of `loadLabelsBinary`
    virtual bool loadLabelsBinary(const std::string& filename, DeviceT& device) = 0;

    // All DeviceT combos of `storeLabelsBinary`
    virtual bool storeLabelsBinary(const std::string& filename, DeviceT& device) = 0;

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) { }
  };

  /// The erasure wrapper around the Tensor Axis interface
  template<typename T, typename DeviceT>
  class TensorDimensionWrapper : public TensorDimensionConcept<DeviceT> {
    std::shared_ptr<T> tensor_dimension_;
  public:
    TensorDimensionWrapper(const std::shared_ptr<T>& tensor_dimension) : tensor_dimension_(tensor_dimension) {};
    TensorDimensionWrapper() = default;
    ~TensorDimensionWrapper() = default;

    int getId() const override { return tensor_dimension_->getId(); };
    std::string getName() const override { return tensor_dimension_->getName(); };
    size_t getNLabels() const override { return tensor_dimension_->getNLabels(); };

//    void getLabelsDataPointer(std::shared_ptr<int>& data_copy) override {
//      tensor_dimension_->getLabelsDataPointer(data_copy);
//    };
//    void getLabelsDataPointer(std::shared_ptr<float>& data_copy) override {
//      tensor_dimension_->getLabelsDataPointer(data_copy);
//    };
//    void getLabelsDataPointer(std::shared_ptr<double>& data_copy) override {
//      tensor_dimension_->getLabelsDataPointer(data_copy);
//    };
//    void getLabelsDataPointer(std::shared_ptr<char>& data_copy) override {
//      tensor_dimension_->getLabelsDataPointer(data_copy);
//    };
//    void getLabelsDataPointer(std::shared_ptr<TensorArray8<char>>& data_copy) override {
//      tensor_dimension_->getLabelsDataPointer(data_copy);
//    };
//#if COMPILE_WITH_CUDA
//    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu8<char>>& data_copy) override {
//      tensor_dimension_->getLabelsDataPointer(data_copy);
//    };
//#endif

    bool syncHAndDData(DeviceT& device) override { return  tensor_dimension_->syncHAndDData(device); };  
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_dimension_->setDataStatus(h_data_updated, d_data_updated); } 
    std::pair<bool, bool> getDataStatus() { return  tensor_dimension_->getDataStatus(); };  

    bool loadLabelsBinary(const std::string& filename, DeviceT& device) override {
      return tensor_dimension_->loadLabelsBinary(filename, device);
    }

    bool storeLabelsBinary(const std::string& filename, DeviceT& device) override {
      return tensor_dimension_->storeLabelsBinary(filename, device);
    }

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimensionConcept<DeviceT>>(this), tensor_dimension_);
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double and DeviceTs: Default, ThreadPool, Gpu
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<int>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<float>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<double>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<char>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray8<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray32<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray128<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray512<char>>, Eigen::DefaultDevice>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionWrapper<TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray2048<char>>, Eigen::DefaultDevice>);
#endif //TENSORBASE_TENSORDIMENSIONCONCEPT_H