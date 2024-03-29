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
    virtual std::shared_ptr<TensorAxisConcept<DeviceT>> copyToHost(DeviceT& device) = 0;
    virtual std::shared_ptr<TensorAxisConcept<DeviceT>> copyToDevice(DeviceT& device) = 0;
    virtual void setLabels() = 0;

    // All TensorT combos of `getLabelsDatapointer`
    virtual void getLabelsDataPointer(std::shared_ptr<int[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<float[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<double[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<char[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray8<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray32<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray128<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray512<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArray2048<char>[]>& data_copy) = 0;
#if COMPILE_WITH_CUDA
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu8<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu32<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu128<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu512<char>[]>& data_copy) = 0;
    virtual void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu2048<char>[]>& data_copy) = 0;
#endif

    // All TensorT combos of `getLabelsHDataPointer`
    virtual void getLabelsHDataPointer(std::shared_ptr<int[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<float[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<double[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<char[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArray8<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArray32<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArray128<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArray512<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArray2048<char>[]>& data_copy) = 0;
#if COMPILE_WITH_CUDA
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu8<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu32<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu128<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu512<char>[]>& data_copy) = 0;
    virtual void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu2048<char>[]>& data_copy) = 0;
#endif

    // All DeviceT combos of `deleteFromAxis`
    virtual void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) = 0;

    // All TensorT DeviceT combos of `selectFromAxis`
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray32<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray128<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray512<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray2048<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu32<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu128<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu512<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu2048<char>, DeviceT, 2>>& labels_select, DeviceT& device) = 0;
#endif

    // All TensorT and DeviceT combos of `appendLabelsToAxis`
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray32<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray128<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray512<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray2048<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
#if COMPILE_WITH_CUDA
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu32<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu128<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu512<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu2048<char>, DeviceT, 2>>& labels, DeviceT& device) = 0;
#endif

    // All DeviceT combos of tensorDataWrappers
    virtual bool syncDData(DeviceT& device) = 0;
    virtual bool syncHData(DeviceT& device) = 0;
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

    int getId() const override { return tensor_axis_->getId(); };
    std::string getName() const override { return tensor_axis_->getName(); };
    size_t getNLabels() const override { return tensor_axis_->getNLabels(); };
    size_t getNDimensions() const override { return tensor_axis_->getNDimensions(); };
    Eigen::TensorMap<Eigen::Tensor<std::string, 1>> getDimensions() override { return tensor_axis_->getDimensions(); };
    std::shared_ptr<TensorAxisConcept<DeviceT>> copyToHost(DeviceT& device) override {
      auto tensor_axis_copy = tensor_axis_->copyToHost(device);
      return std::make_shared<TensorAxisWrapper<T, DeviceT>>(tensor_axis_copy);
    }
    std::shared_ptr<TensorAxisConcept<DeviceT>> copyToDevice(DeviceT& device) override {
      auto tensor_axis_copy = tensor_axis_->copyToDevice(device);
      return std::make_shared<TensorAxisWrapper<T, DeviceT>>(tensor_axis_copy);
    }
    void setLabels() override { tensor_axis_->setLabels(); }

    void getLabelsDataPointer(std::shared_ptr<int[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<float[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<double[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<char[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArray8<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArray32<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArray128<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArray512<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArray2048<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
#if COMPILE_WITH_CUDA
    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu8<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu32<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu128<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu512<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
    void getLabelsDataPointer(std::shared_ptr<TensorArrayGpu2048<char>[]>& data_copy) override {
      tensor_axis_->getLabelsDataPointer(data_copy);
    };
#endif

    void getLabelsHDataPointer(std::shared_ptr<int[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<float[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<double[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<char[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArray8<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArray32<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArray128<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArray512<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArray2048<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
#if COMPILE_WITH_CUDA
    void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu8<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu32<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu128<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu512<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
    void getLabelsHDataPointer(std::shared_ptr<TensorArrayGpu2048<char>[]>& data_copy) override {
      tensor_axis_->getLabelsHDataPointer(data_copy);
    };
#endif

    void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) override {
      tensor_axis_->deleteFromAxis(indices, device);
    };

    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<float, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<double, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<char, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray32<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray128<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray512<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArray2048<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
#if COMPILE_WITH_CUDA
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu32<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu128<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu512<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
    void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorArrayGpu2048<char>, DeviceT, 2>>& labels_select, DeviceT& device) override {
      tensor_axis_->selectFromAxisConcept(indices, labels_select, device);
    };
#endif

    void appendLabelsToAxis(const std::shared_ptr<TensorData<int, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<float, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<double, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<char, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray8<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray32<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray128<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray512<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArray2048<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
#if COMPILE_WITH_CUDA
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu8<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu32<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu128<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu512<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorArrayGpu2048<char>, DeviceT, 2>>& labels, DeviceT& device) override {
      tensor_axis_->appendLabelsToAxisConcept(labels, device);
    };
#endif

    bool syncDData(DeviceT& device) override { return  tensor_axis_->syncDData(device); };
    bool syncHData(DeviceT& device) override { return  tensor_axis_->syncHData(device); };
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) override { tensor_axis_->setDataStatus(h_data_updated, d_data_updated); } 
    std::pair<bool, bool> getDataStatus() override { return  tensor_axis_->getDataStatus(); };  

    void sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device) override {
      tensor_axis_->sortLabels(indices, device);
    };

    bool loadLabelsBinary(const std::string& filename, DeviceT& device) override {
      return tensor_axis_->loadLabelsBinary(filename, device);
    }

    bool storeLabelsBinary(const std::string& filename, DeviceT& device) override {
      return tensor_axis_->storeLabelsBinary(filename, device);
    }

    std::vector<std::string> getLabelsAsStrings(DeviceT& device) override {
      return tensor_axis_->getLabelsAsStrings(device);
    }
    std::vector<std::string> getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) {
      return tensor_axis_->getLabelsAsStrings(offset, span);
    }

    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, DeviceT& device) override {
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
#endif //TENSORBASE_TENSORAXISCONCEPT_H