/**TODO:  Add copyright*/

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkArrayCpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

/* Benchmark for the TensorArray fixed-size array class

Example usage:
	BenchmarkArray [data_size] [array_size] [n_engines]
	BenchmarkArray S S 8

@param[in] data_size Options include XS, S, M, L, and XL (i.e., 1e3, 1e5, 1e6, 1e7, and 1e9 respectively) with default of S
@param[in] array_size Options include XS, S, M, L, and XL (i.e., 8, 32, 128, 512, and 2048, respectively) with default of S
@param[in] n_engines The number of engines (i.e., threads for CPU and streams for GPU) to use (only available on CPU and GPU versions)
*/
int main(int argc, char** argv)
{
	// Parse the user commands
	int data_size = 1e5;
  int array_size = 32;
  int n_engines = 2;
  parseCmdArgsArray(argc, argv, data_size, array_size, n_engines);

  // Setup the benchmarking suite
  BenchmarkArrayCpu<char> benchmark_array;

	// Setup the device
  Eigen::ThreadPool pool(n_engines);
  Eigen::ThreadPoolDevice device(&pool, n_engines);

	// run the application
  runBenchmarkArray(data_size, array_size, benchmark_array, device);

	return 0;
}