/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAYCPU_H
#define TENSORBASE_BENCHMARKARRAYCPU_H

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
  void ArrayManagerCpu<ArrayT, TensorT>::makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::ThreadPoolDevice, 1>>& values_ptr) {
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
  protected:
    void sortBenchmark_(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const override;
    void partitionBenchmark_(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const override;
    void makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& selection_ptr) const override;
  };
  template<typename TensorT>
  void BenchmarkArrayCpu<TensorT>::sortBenchmark_(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const {
    if (array_size == 8) {
      ArrayManagerCpu<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 32) {
      ArrayManagerCpu<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 128) {
      ArrayManagerCpu<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 512) {
      ArrayManagerCpu<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 2048) {
      ArrayManagerCpu<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      values_ptr->sort("ASC", device);
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
    }
  }
  template<typename TensorT>
  inline void BenchmarkArrayCpu<TensorT>::partitionBenchmark_(const int& data_size, const int& array_size, Eigen::ThreadPoolDevice& device) const
  {
    if (array_size == 8) {
      ArrayManagerCpu<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 32) {
      ArrayManagerCpu<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 128) {
      ArrayManagerCpu<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 512) {
      ArrayManagerCpu<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else if (array_size == 2048) {
      ArrayManagerCpu<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::ThreadPoolDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr, device);
      std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> selection_ptr;
      this->makeRandomTensorSelectionData(data_size, selection_ptr, device);
      values_ptr->partition(selection_ptr, device);
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
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