/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKARRAY_H
#define TENSORBASE_BENCHMARKARRAY_H

#include <random> // random number generator
#include <typeinfo> // typeid

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/*
	@brief Class for generating arrays of different types and lengths
	*/
	template<template<class> class ArrayT, class TensorT, typename DeviceT>
	class ArrayManager {
	public:
		ArrayManager(const int& data_size, const int& array_size) : data_size_(data_size), array_size_(array_size){};
		~ArrayManager() = default;
		void getArrayData(std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr, DeviceT& device);
		virtual void makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr) = 0;

		/*
		@brief Generate a random value
		*/
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int> = 0>
		T getRandomValue();
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, int> = 0>
    T getRandomValue();
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value, int> = 0>
    T getRandomValue();
	protected:
		int data_size_;
    int array_size_;
    //Eigen::Tensor<ArrayT<TensorT>, 1> values_; ///< cache for pre-made values
	};
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  void ArrayManager<ArrayT, TensorT, DeviceT>::getArrayData(std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr, DeviceT& device) {
    //if (values_.size() <= 0) { // Make the values
    //  Eigen::Tensor<ArrayT<TensorT>, 1> values(data_size_);
    //  for (int i = 0; i < data_size_; ++i) {
    //    Eigen::Tensor<TensorT, 1> array_tmp(array_size_);
    //    for (int j = 0; j < array_size_; ++j) array_tmp(j) = getRandomValue();
    //    values(i) = ArrayT<TensorT>(array_tmp);
    //  }
    //  values_ = values;
    //}
    //makeValuesPtr(values_, values_ptr);
    Eigen::Tensor<ArrayT<TensorT>, 1> values(data_size_);
    for (int i = 0; i < data_size_; ++i) {
      Eigen::Tensor<TensorT, 1> array_tmp(array_size_);
      for (int j = 0; j < array_size_; ++j) array_tmp(j) = getRandomValue();
      values(i) = ArrayT<TensorT>(array_tmp);
    }
    makeValuesPtr(values, values_ptr);
    values_ptr->syncHAndDData(device);
  }
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, char>::value, int>>
	T ArrayManager<ArrayT, TensorT, DeviceT>::getRandomValue() {
    std::vector<char> elements = { ' ','!','#','$','%','&','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',
      ':',';','<','=','>','?','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
      '[',']','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','}'};
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_int_distribution<int> choose(0, elements.size() - 1);
    return elements.at(choose(engine));
	}
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, int>>
  T ArrayManager<ArrayT, TensorT, DeviceT>::getRandomValue() {
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 0.0f, 10.0f };
    return T(d(gen));
  }
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, int>::value, int>>
  T ArrayManager<ArrayT, TensorT, DeviceT>::getRandomValue() {
    std::vector<int> elements = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_int_distribution<int> choose(0, elements.size() - 1);
    return elements.at(choose(engine));
  }
   
	/*
	@brief A class for running various benchmarks on different flavors of TensorArrays
	*/
  template<typename TensorT, typename DeviceT>
	class BenchmarkArray {
	public:
		BenchmarkArray() = default;
		~BenchmarkArray() = default;
		/*
		@brief sort randomly generated arrays into ascending order

		@param[in] data_size
		@param[in] array_size
		@param[in] device

		@returns A string with the total time of the benchmark in milliseconds
		*/
		virtual std::string sortBenchmark(const int& data_size, const int& array_size, DeviceT& device) const = 0;
    /*
    @brief partition randomly generated arrays

    @param[in] data_size
		@param[in] array_size
    @param[in] device

    @returns A string with the total time of the benchmark in milliseconds
    */
    virtual std::string partitionBenchmark(const int& data_size, const int& array_size, DeviceT& device) const = 0;
    /*
    @brief partition randomly generated arrays

    @param[in] data_size
    @param[in] array_size
    @param[in] device

    @returns A string with the total time of the benchmark in milliseconds
    */
    void makeRandomTensorSelectionData(const int& data_size, std::shared_ptr<TensorData<int, DeviceT, 1>>& selection_ptr, DeviceT& device) const;
	protected:
    virtual void makeTensorSelectionData_(const Eigen::Tensor<int, 1>& values, std::shared_ptr<TensorData<int, DeviceT, 1>>& selection_ptr) const = 0;
	};
  template<typename TensorT, typename DeviceT>
  inline void BenchmarkArray<TensorT, DeviceT>::makeRandomTensorSelectionData(const int& data_size, std::shared_ptr<TensorData<int, DeviceT, 1>>& selection_ptr, DeviceT& device) const
  {
    Eigen::Tensor<int, 1> selection_values(data_size);
    auto zeroOrOne = [](const int& v) {
      std::vector<int> elements = { 0, 1 };
      std::random_device seed;
      std::mt19937 engine(seed());
      std::uniform_int_distribution<int> choose(0, elements.size() - 1);
      return elements.at(choose(engine));
    };
    selection_values = selection_values.unaryExpr(zeroOrOne);
    makeTensorSelectionData_(selection_values, selection_ptr);
    selection_ptr->syncHAndDData(device);
  }

  template<typename TensorT, typename DeviceT>
	static void runBenchmarkArray(const int& data_size, const int& array_size, const BenchmarkArray<TensorT, DeviceT>& benchmark_array, DeviceT& device) {
		std::cout << "Starting TensorArray benchmarks for array_size=" << array_size << ", TensorT=" << typeid(TensorT).name() << ", data_size=" << data_size << std::endl;

		// Run the table through the benchmarks
		std::cout << "Sort took " << benchmark_array.sortBenchmark(data_size, array_size, device) << " milliseconds." << std::endl;
		std::cout << "Partition took " << benchmark_array.partitionBenchmark(data_size, array_size, device) << " milliseconds." << std::endl;
	}

	///Parse the command line arguments
  static void parseCmdArgsArray(const int& argc, char** argv, int& data_size, int& array_size, int& n_engines) {
    if (argc >= 3) {
      if (argv[2] == std::string("XS")) {
        data_size = 1e3;
      }
      else if (argv[2] == std::string("S")) {
        data_size = 1e5;
      }
      else if (argv[2] == std::string("M")) {
        data_size = 1e6;
      }
      else if (argv[2] == std::string("L")) {
        data_size = 1e7;
      }
      else if (argv[2] == std::string("XL")) {
        data_size = 1e9;
      }
    }
    if (argc >= 4) {
      if (argv[3] == std::string("XS")) {
        array_size = 8;
      }
      else if (argv[3] == std::string("S")) {
        array_size = 32;
      }
      else if (argv[3] == std::string("M")) {
        array_size = 128;
      }
      else if (argv[3] == std::string("L")) {
        array_size = 512;
      }
      else if (argv[3] == std::string("XL")) {
        array_size = 2048;
      }
    }
    if (argc >= 5) {
      try {
        n_engines = std::stoi(argv[4]);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
    }
  }
};
#endif //TENSORBASE_BENCHMARKARRAY_H