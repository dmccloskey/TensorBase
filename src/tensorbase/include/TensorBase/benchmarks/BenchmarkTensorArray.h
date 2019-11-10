/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKTENSORARRAY_H
#define TENSORBASE_BENCHMARKTENSORARRAY_H

#include <ctime> // time format
#include <chrono> // current time
#include <math.h> // std::pow
#include <random> // random number generator

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
		void getArrayData(std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr);
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
	};
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  TensorT ArrayManager<ArrayT, TensorT, DeviceT>::getArrayData(std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr) {
    Eigen::Tensor<ArrayT<TensorT>, 1> values(data_size_);
    for (int i = 0; i < data_size_; ++i) {
      Eigen::TensorT<TensorT, 1> array_tmp(array_size_);
      for (int j = 0; j < array_size_; ++j) array_tmp(j) = getRandomValue();
      values(i) = ArrayT<TensorT>(array_tmp);
    }
    makeValuesPtr(values, values_ptr);
  }
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int>>
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
  template<typename T = TensorT, std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value, int>>
  T ArrayManager<ArrayT, TensorT, DeviceT>::getRandomValue() {
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 0.0f, 10.0f };
    return T(d(gen));
  }
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
  template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value, int>>
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
		std::string sortBenchmark(const int& data_size, const int& array_size, DeviceT& device) const;
    /*
    @brief match the contents of randomly generated arrays

    @param[in] data_size
		@param[in] array_size
    @param[in] device

    @returns A string with the total time of the benchmark in milliseconds
    */
		std::string matchBenchmark(const int& data_size, const int& array_size, DeviceT& device) const;
    /*
    @brief partition randomly generated arrays

    @param[in] data_size
		@param[in] array_size
    @param[in] device

    @returns A string with the total time of the benchmark in milliseconds
    */
    std::string paritionBenchmark(const int& data_size, const int& array_size, DeviceT& device) const;
	protected:
    virtual void sortBenchmark_(const int& data_size, const int& array_size, DeviceT& device) const = 0;
    virtual void matchBenchmark_(const int& data_size, const int& array_size, DeviceT& device) const = 0;
    virtual void partitionBenchmark_(const int& data_size, const int& array_size, DeviceT& device) const = 0;
	};
  template<typename TensorT, typename DeviceT>
	std::string BenchmarkArray<TensorT, DeviceT>::sortBenchmark(const int& data_size, const int& array_size, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    sortBenchmark_(data_size, array_size, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
  template<typename TensorT, typename DeviceT>
	std::string BenchmarkArray<TensorT, DeviceT>::matchBenchmark(const int& data_size, const int& array_size, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    matchBenchmark_(data_size, array_size, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
  template<typename TensorT, typename DeviceT>
	std::string BenchmarkArray<TensorT, DeviceT>::paritionBenchmark(const int& data_size, const int& array_size, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    paritionBenchmark_(data_size, array_size, device);

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}

  template<typename TensorT, typename DeviceT>
	static void runBenchmarkPixels(const std::string& data_dir, const int& data_size, const int& array_size,
		const BenchmarkArray<TensorT, DeviceT>& benchmark_array, DeviceT& device) {
		std::cout << "Starting TensorArray benchmarks for array_size=" << array_size << ", TensorT=" << TensorT << ", data_size=" << data_size << std::endl;

		// Run the table through the benchmarks
		std::cout << "Sort took " << benchmark_array.sortBenchmark(data_size, array_size, device) << " milliseconds." << std::endl;
		std::cout << "Match took " << benchmark_array.matchBenchmark(data_size, array_size, device) << " milliseconds." << std::endl;
		std::cout << "Partition took " << benchmark_array.paritionBenchmark(data_size, array_size, device) << " milliseconds." << std::endl;
	}

	///Parse the command line arguments
  static void parseCmdArgs(const int& argc, char** argv, int& data_size, int& array_size, int& n_engines) {
    if (argc >= 2) {
      if (argv[1] == std::string("XS")) {
        data_size = 10;
      }
      else if (argv[1] == std::string("S")) {
        data_size = 1e3;
      }
      else if (argv[1] == std::string("M")) {
        data_size = 1e6;
      }
      else if (argv[1] == std::string("L")) {
        data_size = 1e9;
      }
      else if (argv[1] == std::string("XL")) {
        data_size = 1e12;
      }
    }
    if (argc >= 3) {
      if (argv[2] == std::string("XS")) {
        array_size = 8;
      }
      else if (argv[2] == std::string("S")) {
        array_size = 32;
      }
      else if (argv[2] == std::string("M")) {
        array_size = 128;
      }
      else if (argv[2] == std::string("L")) {
        array_size = 512;
      }
      else if (argv[2] == std::string("XL")) {
        array_size = 2048;
      }
    }
    if (argc >= 4) {
      try {
        n_engines = std::stoi(argv[3]);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
    }
  }
};
#endif //TENSORBASE_BENCHMARKTENSORARRAY_H