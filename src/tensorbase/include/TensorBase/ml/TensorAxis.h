/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXIS_H
#define TENSORBASE_TENSORAXIS_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>
#include <string>

namespace TensorBase
{
  /**
    @brief Tensor axis class
  */
  template<typename TensorT, typename DeviceT>
  class TensorAxis
  {
  public:
    TensorAxis() = default;  ///< Default constructor
    TensorAxis(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
    virtual ~TensorAxis() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< naem setter
    std::string getName() const { return name_; }; ///< name getter

    size_t getNLabels() const { return n_labels_; }; ///< n_labels getter
    size_t getNDimensions() const { return n_dimensions_; }; ///< n_labels getter

    virtual void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) = 0; ///< dimensions and labels setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getLabels() { return tensor_dimension_labels_->getData(); };  ///< labels getter
    std::shared_ptr<TensorT> getLabelsDataPointer() {return tensor_dimension_labels_->getDataPointer(); };
    Eigen::Tensor<std::string, 1>& getDimensions() { return tensor_dimension_names_; };  ///< dimensions getter

    template<typename T>
    void getLabelsDataPointer(std::shared_ptr<T>& data_copy); ///< TensorAxisConcept data getter

  protected:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter
    void setNDimensions(const size_t& n_dimenions) { n_dimensions_ = n_dimenions; }; ///< n_tensor_dimensions setter

    int id_ = -1;
    std::string name_ = "";
    size_t n_dimensions_ = 0;  ///< the number of "primary keys" or dimensions that compose the axis
    size_t n_labels_ = 0;  ///< The size or length of the axis
    Eigen::Tensor<std::string, 1> tensor_dimension_names_;  ///< array of TensorDimension names
    std::shared_ptr<TensorData<TensorT, DeviceT, 2>> tensor_dimension_labels_; ///< dim=0: tensor_dimension_name; dim=1 tensor_dimension_labels

    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
  };
  template<typename TensorT, typename DeviceT>
  TensorAxis<TensorT, DeviceT>::TensorAxis(const std::string& name,
    const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    setName(name);
    setDimensionsAndLabels(dimensions, labels);
  }

  template<typename TensorT, typename DeviceT>
  template<typename T>
  void TensorAxis<TensorT, DeviceT>::getLabelsDataPointer(std::shared_ptr<T>& data_copy) {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T>(tensor_dimension_labels_->getDataPointer()); // required for compilation: no conversion should be done
  }

  template<typename TensorT>
  class TensorAxisDefaultDevice : public TensorAxis<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorAxisDefaultDevice() = default;  ///< Default constructor
    TensorAxisDefaultDevice(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
    ~TensorAxisDefaultDevice() = default; ///< Default destructor
    void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
  };
  template<typename TensorT>
  TensorAxisDefaultDevice<TensorT>::TensorAxisDefaultDevice(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    setName(name);
    setDimensionsAndLabels(dimensions, labels);
  }
  template<typename TensorT>
  void TensorAxisDefaultDevice<TensorT>::setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    assert(labels.dimension(0) == dimensions.dimension(0));
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_.reset(new TensorDataDefaultDevice<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->tensor_dimension_names_ = dimensions;
    this->setNDimensions(labels.dimension(0));
    this->setNLabels(labels.dimension(1));
  };

  template<typename TensorT>
  class TensorAxisCpu : public TensorAxis<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorAxisCpu() = default;  ///< Default constructor
    TensorAxisCpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
    ~TensorAxisCpu() = default; ///< Default destructor
    void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
  };
  template<typename TensorT>
  TensorAxisCpu<TensorT>::TensorAxisCpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    setName(name);
    setDimensionsAndLabels(dimensions, labels);
  }
  template<typename TensorT>
  void TensorAxisCpu<TensorT>::setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) {
    assert(labels.dimension(0) == dimensions.dimension(0));
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_.reset(new TensorDataCpu<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->tensor_dimension_names_ = dimensions;
    this->setNDimensions(labels.dimension(0));
    this->setNLabels(labels.dimension(1));
  };

#if COMPILE_WITH_CUDA
  template<typename TensorT>
  class TensorAxisGpu : public TensorAxis<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorAxisGpu() = default;  ///< Default constructor
    TensorAxisGpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
    ~TensorAxisGpu() = default; ///< Default destructor
    void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels);
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
#endif
};
#endif //TENSORBASE_TENSORAXIS_H