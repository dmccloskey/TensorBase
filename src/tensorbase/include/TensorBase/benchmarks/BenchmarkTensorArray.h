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
		ArrayManager(const int& data_size, const bool& use_random_values = false) : data_size_(data_size), use_random_values_(use_random_values){};
		~ArrayManager() = default;
		void getArrayData(std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr) = 0;
		virtual void makeValuesPtr(const Eigen::Tensor<ArrayT<TensorT>, 1>& values, std::shared_ptr<TensorData<ArrayT<TensorT>, DeviceT, 1>>& values_ptr) = 0;

		/*
		@brief Generate a random value
		*/
		TensorT getRandomValue();
	protected:
		int data_size_;
		bool use_random_values_;
	};
  template<template<class> class ArrayT, class TensorT, typename DeviceT>
	TensorT ArrayManager<ArrayT, TensorT, DeviceT>::getRandomValue() {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> d{ 0.0f, 10.0f };
		return TensorT(d(gen));
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