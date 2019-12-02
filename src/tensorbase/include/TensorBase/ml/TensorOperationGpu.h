/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATIONGPU_H
#define TENSORBASE_TENSOROPERATIONGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorOperation.h>
#include <TensorBase/ml/TensorDataGpu.h>

namespace TensorBase
{
  /**
    @brief Gpu with primitive types specialization of `TensorDeleteFromAxis`
  */
  template<typename LabelsT, typename TensorT, int TDim>
  class TensorDeleteFromAxisGpuPrimitiveT : public TensorDeleteFromAxis<LabelsT, TensorT, Eigen::GpuDevice, TDim> {
  public:
    using TensorDeleteFromAxis<LabelsT, TensorT, Eigen::GpuDevice, TDim>::TensorDeleteFromAxis;
    void allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device) override;
  };

  template<typename LabelsT, typename TensorT, int TDim>
  inline void TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, TDim>::allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice & device)
  {
    // Determine the dimensions of the values that will be deleted
    Eigen::array<Eigen::Index, TDim> dimensions_new;
    for (auto& axis_map: tensor_collection->tables_.at(this->table_name_)->getAxes()) {
      dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(axis_map.second->getName())) = axis_map.second->getNLabels();
    }
    dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(this->axis_name_)) = this->indices_->getTensorSize();

    // Allocate memory for the values
    TensorDataGpuPrimitiveT<TensorT, TDim> values_tmp(dimensions_new);
    values_tmp.setData();
    this->values_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, TDim>>(values_tmp);
  }

  /**
    @brief Gpu with class types specialization of `TensorDeleteFromAxis`
  */
  template<typename LabelsT, template<class> class ArrayT, class TensorT, int TDim>
  class TensorDeleteFromAxisGpuClassT : public TensorDeleteFromAxis<LabelsT, ArrayT<TensorT>, Eigen::GpuDevice, TDim> {
  public:
    using TensorDeleteFromAxis<LabelsT, ArrayT<TensorT>, Eigen::GpuDevice, TDim>::TensorDeleteFromAxis;
    void allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device) override;
  };

  template<typename LabelsT, template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDeleteFromAxisGpuClassT<LabelsT, ArrayT, TensorT, TDim>::allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::GpuDevice>>& tensor_collection, Eigen::GpuDevice& device)
  {
    // Determine the dimensions of the values that will be deleted
    Eigen::array<Eigen::Index, TDim> dimensions_new;
    for (auto& axis_map : tensor_collection->tables_.at(this->table_name_)->getAxes()) {
      dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(axis_map.second->getName())) = axis_map.second->getNLabels();
    }
    dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(this->axis_name_)) -= this->indices_->getTensorSize();

    // Allocate memory for the values
    TensorDataGpuClassT<ArrayT, TensorT, TDim> values_tmp(dimensions_new);
    values_tmp.setData();
    this->values_ = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, TDim>>(values_tmp);
  }
};
#endif
#endif //TENSORBASE_TENSOROPERATIONGPU_H