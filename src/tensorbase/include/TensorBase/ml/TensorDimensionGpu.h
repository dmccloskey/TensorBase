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
    TensorDimensionGpuPrimitiveT(const std::string& name, const std::string& dir) : TensorDimension(name, dir) {};
    TensorDimensionGpuPrimitiveT(const std::string& name, const std::string& dir, const Eigen::Tensor<TensorT, 1>& labels) : TensorDimension(name, dir) { setLabels(labels); };
    ~TensorDimensionGpuPrimitiveT() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataGpuPrimitiveT<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    bool loadLabelsBinary(const std::string& dir, Eigen::GpuDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<TensorT, 1>(filename, this->getLabels());
      this->syncHAndDData(device); // H to D
    };
    bool storeLabelsBinary(const std::string& dir, Eigen::GpuDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<TensorT, 1>(filename, this->getLabels());
      this->setDataStatus(false, true);
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
    TensorDimensionGpuClassT(const std::string& name, const std::string& dir) : TensorDimension(name, dir) {};
    TensorDimensionGpuClassT(const std::string& name, const std::string& dir, const Eigen::Tensor<ArrayT<TensorT>, 1>& labels) : TensorDimension(name, dir) { setLabels(labels); };
    ~TensorDimensionGpuClassT() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<ArrayT<TensorT>, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataGpuClassT<ArrayT, TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    bool loadLabelsBinary(const std::string& dir, Eigen::GpuDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<< ArrayT<TensorT>, 1>(filename, this->getLabels());
      this->syncHAndDData(device); // H to D
    };
    bool storeLabelsBinary(const std::string& dir, Eigen::GpuDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<< ArrayT<TensorT>, 1>(filename, this->getLabels());
      this->setDataStatus(false, true);
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<ArrayT<TensorT>, Eigen::GpuDevice>>(this));
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double, charArray8
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuPrimitiveT<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionGpuClassT<TensorBase::TensorArrayGpu8, char>);
#endif
#endif //TENSORBASE_TENSORDIMENSIONGPU_H