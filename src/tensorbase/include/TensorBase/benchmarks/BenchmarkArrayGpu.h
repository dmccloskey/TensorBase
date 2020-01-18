/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAYGPU_H
#define TENSORBASE_BENCHMARKARRAYGPU_H

#include <ctime> // time format
#include <chrono> // current time

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
  inline void ArrayManagerGpuClassT<ArrayT, TensorT>::makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& values_ptr) {
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
    std::string sortBenchmark(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const override;
    std::string partitionBenchmark(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const override;
  protected:
    void makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& selection_ptr) const override;
  };
  template<typename TensorT>
  inline std::string BenchmarkArrayGpuClassT<TensorT>::sortBenchmark(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const {
    if (array_size == 8) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu8<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->sort("ASC", device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 32) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu32<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->sort("ASC", device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 128) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu128<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->sort("ASC", device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 512) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu512<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->sort("ASC", device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 2048) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu2048<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->sort("ASC", device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
      return std::string();
    }
  }
  template<typename TensorT>
  inline std::string BenchmarkArrayGpuClassT<TensorT>::partitionBenchmark(const int& data_size, const int& array_size, Eigen::GpuDevice& device) const
  {
    if (array_size == 8) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu8<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->partition(selection_ptr, device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 32) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu32<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->partition(selection_ptr, device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 128) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu128<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->partition(selection_ptr, device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 512) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu512<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->partition(selection_ptr, device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else if (array_size == 2048) {
      // Make the data
      ArrayManagerGpuClassT<TensorArrayGpu2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArrayGpu2048<TensorT>, Eigen::GpuDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);

      // Start the timer
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

      // Run the operation
      values_ptr->partition(selection_ptr, device);

      // Stop the timer
      auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      std::string milli_time = std::to_string(stop - start);
      return milli_time;
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
      return std::string();
    }
  }
  template<typename TensorT>
  inline void BenchmarkArrayGpuClassT<TensorT>::makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& selection_ptr) const
  {
    TensorDataGpuPrimitiveT<int, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    selection_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(values_data);
  }
};
#endif
#endif //TENSORBASE_BENCHMARKARRAYGPU_H