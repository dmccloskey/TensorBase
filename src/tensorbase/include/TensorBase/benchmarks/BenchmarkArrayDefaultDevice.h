/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAYDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKARRAYDEFAULTDEVICE_H

#include <ctime> // time format
#include <chrono> // current time

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkArray.h>
#include <TensorBase/ml/TensorDataDefaultDevice.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
  /*
  @brief Class for generating arrays of different types and lengths
  */
  template<template<class> class ArrayT, class TensorT>
  class ArrayManagerDefaultDevice: public ArrayManager<ArrayT, TensorT, Eigen::DefaultDevice> {
  public:
    using ArrayManager::ArrayManager;
    void makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::DefaultDevice, 1>>& values_ptr) override;
  };
  template<template<class> class ArrayT, class TensorT>
  inline void ArrayManagerDefaultDevice<ArrayT, TensorT>::makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::DefaultDevice, 1>>& values_ptr) {
    TensorDataDefaultDevice<ArrayT<TensorT>, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    values_ptr = std::make_shared<TensorDataDefaultDevice<ArrayT<TensorT>, 1>>(values_data);
  }
   
  /*
  @brief A class for running various benchmarks on different flavors of TensorArrays
  */
  template<typename TensorT>
  class BenchmarkArrayDefaultDevice: public BenchmarkArray<TensorT, Eigen::DefaultDevice> {
  public:
    using BenchmarkArray::BenchmarkArray;
    std::string sortBenchmark(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const override;
    std::string partitionBenchmark(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const override;
  protected:
    void makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& selection_ptr) const override;
  };
  template<typename TensorT>
  inline std::string BenchmarkArrayDefaultDevice<TensorT>::sortBenchmark(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const {
    if (array_size == 8) {
      // Make the data
      ArrayManagerDefaultDevice<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
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
      ArrayManagerDefaultDevice<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
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
      ArrayManagerDefaultDevice<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
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
      ArrayManagerDefaultDevice<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
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
      ArrayManagerDefaultDevice<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
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
  inline std::string BenchmarkArrayDefaultDevice<TensorT>::partitionBenchmark(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const
  {
    if (array_size == 8) {
      // Make the data
      ArrayManagerDefaultDevice<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> selection_ptr;
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
      ArrayManagerDefaultDevice<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> selection_ptr;
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
      ArrayManagerDefaultDevice<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> selection_ptr;
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
      ArrayManagerDefaultDevice<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> selection_ptr;
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
      ArrayManagerDefaultDevice<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> selection_ptr;
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
  inline void BenchmarkArrayDefaultDevice<TensorT>::makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& selection_ptr) const
  {
    TensorDataDefaultDevice<int, 1> values_data(Eigen::array<Eigen::Index, 1>({ values.dimension(0) }));
    values_data.setData(values);
    selection_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(values_data);
  }
};
#endif //TENSORBASE_BENCHMARKARRAYDEFAULTDEVICE_H