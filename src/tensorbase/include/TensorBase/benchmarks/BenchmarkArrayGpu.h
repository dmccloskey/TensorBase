/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAYGPU_H
#define TENSORBASE_BENCHMARKARRAYGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkArray.h>
#include <TensorBase/ml/TensorDataGpu.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
  /*
  @brief Class for generating arrays of different types and lengths
  */
  template<template<class> class ArrayT, class TensorT>
  class ArrayManagerGpuClassT: public ArrayManager<ArrayT, TensorT, Eigen::GpuDevice> {
  public:
    using ArrayManager::ArrayManager;
    void makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& values_ptr) override;
  };
  template<template<class> class ArrayT, class TensorT>
  void ArrayManagerGpuClassT<ArrayT, TensorT>::makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& values_ptr) {
    TensorDataGpuClassT<ArrayT, TensorT, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, 1>>(values_data);
  }
   
  /*
  @brief A class for running various benchmarks on different flavors of TensorArrays
  */
  template<typename TensorT>
  class BenchmarkArrayGpuClassT: public BenchmarkArray<TensorT, Eigen::GpuDevice> {
  public:
    using BenchmarkArray::BenchmarkArray;
  protected:
    void sortBenchmark_(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const override;
    void partitionBenchmark_(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const override;
    void makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& selection_ptr) const override;
  };
  template<typename TensorT>
  void BenchmarkArrayGpuClassT<TensorT>::sortBenchmark_(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const {
    if (array_size == 8) {
      ArrayManagerGpuClassT<TensorArrayGpu8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu8<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 32) {
      ArrayManagerGpuClassT<TensorArrayGpu32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu32<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 128) {
      ArrayManagerGpuClassT<TensorArrayGpu128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu128<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 512) {
      ArrayManagerGpuClassT<TensorArrayGpu512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu512<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 2048) {
      ArrayManagerGpuClassT<TensorArrayGpu2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu2048<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
    }
  }
  template<typename TensorT>
  inline void BenchmarkArrayGpuClassT<TensorT>::partitionBenchmark_(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const
  {
    if (array_size == 8) {
      ArrayManagerGpuClassT<TensorArrayGpu8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu8<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 32) {
      ArrayManagerGpuClassT<TensorArrayGpu32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu32<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 128) {
      ArrayManagerGpuClassT<TensorArrayGpu128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu128<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 512) {
      ArrayManagerGpuClassT<TensorArrayGpu512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu512<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 2048) {
      ArrayManagerGpuClassT<TensorArrayGpu2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu2048<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
    }
  }
  template<typename TensorT>
  inline void BenchmarkArrayGpuClassT<TensorT>::makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& selection_ptr) const
  {
    TensorDataGpuPrimitiveT<int, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    selection_ptr = std::make_shared<TensorDataGpuClassT<int, 1>>(values_data);
  }
};
#endif
#endif //TENSORBASE_BENCHMARKARRAYGPU_H