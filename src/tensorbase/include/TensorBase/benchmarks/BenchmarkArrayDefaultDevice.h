/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAYDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKARRAYDEFAULTDEVICE_H

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
  void ArrayManagerDefaultDevice<ArrayT, TensorT>::makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::DefaultDevice, 1>>& values_ptr) {
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
  protected:
    void sortBenchmark_(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const override;
    void partitionBenchmark_(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const override;
  };
  template<typename TensorT>
  void BenchmarkArrayDefaultDevice<TensorT>::sortBenchmark_(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const {
    if (array_size == 8) {
      ArrayManagerDefaultDevice<TensorArray8, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray8<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 32) {
      ArrayManagerDefaultDevice<TensorArray32, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray32<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 128) {
      ArrayManagerDefaultDevice<TensorArray128, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray128<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 512) {
      ArrayManagerDefaultDevice<TensorArray512, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray512<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr);
      values_ptr->sort("ASC", device);
    }
    else if (array_size == 2048) {
      ArrayManagerDefaultDevice<TensorArray2048, TensorT> array_manager(data_size, array_size);
      std::shared_ptr<TensorData<TensorArray2048<TensorT>, Eigen::DefaultDevice, 1>> values_ptr;
      array_manager.getArrayData(values_ptr);
      values_ptr->sort("ASC", device);
    }
    else {
      std::cout << "Array size " << array_size << " is not supported at this time." << std::endl;
    }
  }
  template<typename TensorT>
  inline void BenchmarkArrayDefaultDevice<TensorT>::partitionBenchmark_(const int& data_size, const int& array_size, Eigen::DefaultDevice& device) const
  {
  }
};
#endif //TENSORBASE_BENCHMARKARRAYDEFAULTDEVICE_H