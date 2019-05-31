/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTIONGPU_H
#define TENSORBASE_TENSORCOLLECTIONGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConceptGpu.h>
#include <TensorBase/ml/TensorCollection.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    @brief Class for managing heterogenous Tensors
  */
  template<typename DeviceT>
  class TensorCollectionGpu: public TensorCollection<Eigen::GpuDevice>
  {
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorCollection<Eigen::GpuDevice>>(this));
    }
  };
};
#endif
#endif //TENSORBASE_TENSORCOLLECTIONGPU_H