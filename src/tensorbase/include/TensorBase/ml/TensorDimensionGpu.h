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

namespace TensorBase
{
  template<typename TensorT>
  class TensorDimensionGpu : public TensorDimension<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorDimensionGpu() = default;  ///< Default constructor
    TensorDimensionGpu(const std::string& name) { setName(name); };
    TensorDimensionGpu(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) { setName(name); setLabels(labels); };
    ~TensorDimensionGpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataGpu<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
  };
};
#endif
#endif //TENSORBASE_TENSORDIMENSIONGPU_H