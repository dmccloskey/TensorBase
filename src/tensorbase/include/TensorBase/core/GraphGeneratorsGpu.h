#ifndef TENSORBASE_GRAPHGENERATORSGPU_H
#define TENSORBASE_GRAPHGENERATORSGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/core/GraphGenerators.h>
#include <TensorBase/ml/TensorDataGpu.h>

namespace TensorBase
{
  template<typename LabelsT, typename TensorT>
  class KroneckerGraphGeneratorGpu: public KroneckerGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using KroneckerGraphGenerator<LabelsT, TensorT, Eigen::GpuDevice>::KroneckerGraphGenerator;
    /// allocate memory for the kronecker graph
    void initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, const int& M, Eigen::GpuDevice& device) const override;
  };
  template<typename LabelsT, typename TensorT>
  inline void KroneckerGraphGeneratorGpu<LabelsT, TensorT>::initKroneckerGraph(std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& weights, const int& M, Eigen::GpuDevice& device) const
  {
    TensorDataGpuPrimitiveT<LabelsT, 2> indices_tmp(Eigen::array<Eigen::Index, 2>({ M, 2 }));
    indices_tmp.setData();
    indices_tmp.syncHAndDData(device);
    indices = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(indices_tmp);
    TensorDataGpuPrimitiveT<TensorT, 2> weights_tmp(Eigen::array<Eigen::Index, 2>({ M, 1 }));
    weights_tmp.setData();
    weights_tmp.syncHAndDData(device);
    weights = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(weights_tmp);
  }
}
#endif
#endif //TENSORBASE_GRAPHGENERATORSGPU_H