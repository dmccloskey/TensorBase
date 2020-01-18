/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAYCPU_H
#define TENSORBASE_BENCHMARKARRAYCPU_H

#include <ctime> // time format
#include <chrono> // current time

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkArray.h>
#include <TensorBase/ml/TensorDataCpu.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
  /*
  @brief Class for generating arrays of different types and lengths
  */
  template<template<class> class ArrayT, class TensorT>
  class ArrayManagerCpu: public ArrayManager<ArrayT, TensorT, Eigen::ThreadPoolDevice> {
  public:
    using ArrayManager::ArrayManager;
    void makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::ThreadPoolDevice, 1>>& values_ptr) override;
  };
  template<template<class> class ArrayT, class TensorT>
  inline void ArrayManagerCpu<ArrayT, TensorT>::makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::ThreadPoolDevice, 1>>& values_ptr) {
    TensorDataCpu<ArrayT<TensorT>, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataCpu<ArrayT<TensorT>, 1>>(values_data);
  }
   
  /*
  @brief A class for running various benchmarks on different flavors of TensorArrays
  */
  template<typename TensorT>
  class BenchmarkArrayCpu: public BenchmarkArray<TensorT, Eigen::ThreadPoolDevice> {
  public:
    using BenchmarkArray::BenchmarkArray;
    std::string sortBenchmark(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const override;
    std::string partitionBenchmark(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const override;
  protected:
    void makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& selection_ptr) const override;
  };
  template<typename TensorT>
  inline std::string BenchmarkArrayCpu<TensorT>::sortBenchmark(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const {
    if (array_size == 8) {
      // Make the data
      ArrayManagerCpu<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
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
      ArrayManagerCpu<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
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
      ArrayManagerCpu<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
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
      ArrayManagerCpu<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
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
      ArrayManagerCpu<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
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
  inline std::string BenchmarkArrayCpu<TensorT>::partitionBenchmark(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const
  {
    if (array_size == 8) {
      // Make the data
      ArrayManagerCpu<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
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
      ArrayManagerCpu<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
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
      ArrayManagerCpu<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
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
      ArrayManagerCpu<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
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
      ArrayManagerCpu<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
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
  inline void BenchmarkArrayCpu<TensorT>::makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& selection_ptr) const
  {
    TensorDataCpu<int, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    selection_ptr = std::make_shared<TensorDataCpu<int, 1>>(values_data);
  }
};
#endif //TENSORBASE_BENCHMARKARRAYCPU_H