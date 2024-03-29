/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSIONGPU_H
#define TENSORBASE_TENSORDIMENSIONGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorDataGpu.h>

#include <TensorBase/ml/TensorDimension.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT>
  class TensorDimensionGpuPrimitiveT : public TensorDimension<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorDimensionGpuPrimitiveT() = default;  ///< Default constructor
    TensorDimensionGpuPrimitiveT(const std::string& name) : TensorDimension(name) {};
    TensorDimensionGpuPrimitiveT(const std::string& name, const size_t& n_labels) : TensorDimension(name, n_labels) { this->setLabels(); };
    TensorDimensionGpuPrimitiveT(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) : TensorDimension(name) { this->setLabels(labels); };
    ~TensorDimensionGpuPrimitiveT() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(TensorDataGpuPrimitiveT<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    void setLabels() override {
      Eigen::array<Eigen::Index, 1> dimensions;
      dimensions.at(0) = this->n_labels_;
      this->labels_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(TensorDataGpuPrimitiveT<TensorT, 1>(dimensions));
      this->labels_->setData();
    };
    bool loadLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override {
      this->setDataStatus(true, false);
      Eigen::Tensor<TensorT, 1> data((int)this->n_labels_);
      DataFile::loadDataBinary<TensorT, 1>(filename + ".td", data);
      this->getLabels() = data;
      this->syncHAndDData(device); // H to D
      return true;
    };
    bool storeLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override {
      this->syncHAndDData(device); // D to H
      gpuErrchk(cudaStreamSynchronize(device.stream()));
      DataFile::storeDataBinary<TensorT, 1>(filename + ".td", this->getLabels());
      this->setDataStatus(false, true);
      return true;
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<TensorT, Eigen::GpuDevice>>(this));
    }
  };

  template<template<class> class ArrayT, class TensorT>
  class TensorDimensionGpuClassT : public TensorDimension<ArrayT<TensorT>, Eigen::GpuDevice>
  {
  public:
    TensorDimensionGpuClassT() = default;  ///< Default constructor
    TensorDimensionGpuClassT(const std::string& name) : TensorDimension(name) {};
    TensorDimensionGpuClassT(const std::string& name, const size_t& n_labels) : TensorDimension(name, n_labels) { this->setLabels(); };
    TensorDimensionGpuClassT(const std::string& name, const Eigen::Tensor<ArrayT<TensorT>, 1>& labels) : TensorDimension(name) { this->setLabels(labels); };
    ~TensorDimensionGpuClassT() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<ArrayT<TensorT>, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 1>>(TensorDataGpuClassT<ArrayT, TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    void setLabels() override {
      Eigen::array<Eigen::Index, 1> dimensions;
      dimensions.at(0) = this->n_labels_;
      this->labels_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 1>>(TensorDataGpuClassT<ArrayT, TensorT, 1>(dimensions));
      this->labels_->setData();
    };
    bool loadLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override {
      this->setDataStatus(true, false);
      Eigen::Tensor<ArrayT<TensorT>, 1> data((int)this->n_labels_);
      DataFile::loadDataBinary<ArrayT<TensorT>, 1>(filename + ".td", data);
      this->getLabels() = data;
      this->syncHAndDData(device); // H to D
      return true;
    };
    bool storeLabelsBinary(const std::string& filename, Eigen::GpuDevice& device) override {
      this->syncHAndDData(device); // D to H
      gpuErrchk(cudaStreamSynchronize(device.stream()));
      DataFile::storeDataBinary<ArrayT<TensorT>, 1>(filename + ".td", this->getLabels());
      this->setDataStatus(false, true);
      return true;
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<ArrayT<TensorT>, Eigen::GpuDevice>>(this));
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double, charArray8, charArray32, charArray128, charArray512, charArray2048
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu8, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu32, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu128, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu512, char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu2048, char>);
#endif
#endif //TENSORBASE_TENSORDIMENSIONGPU_H