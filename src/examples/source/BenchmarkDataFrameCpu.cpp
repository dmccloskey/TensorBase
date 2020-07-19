/**TODO:  Add copyright*/

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkDataFrameCpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

/* Benchmark for a DataFrame use case where multiple "columns" (which maybe multi-dimensional) are aligned by a common index.
  The columns include "time", "label", "image_2D", and "is_valid" that represent an array to track sec, min, hr, day, mo, and yr attributes of when
  the image was uploaded, a fixed char to describe the image label, a 2D black and white image, and a bool to indicate if
  the image is valid.

Example usage:
	pixels_benchmark [data_dir] [data_size] [in_memory] [is_columnar] [shard_size_perc]
	pixels_benchmark C:/Users/dmccloskey/Documents/GitHub/mnist/ S true true

@param[in] data_size Options include small, medium, large, and XL (i.e., 1296, 104976, 1048576, 10556001, 1003875856, and 1e12 pixels, respectively) with default of small
	where x, y, z, and t span 1 to 6, 18, 32, 57, 178, and 1000 respectively
@param[in] in_memory Simulate all data loaded into memory (true) or JIT load into memory from disk (false) with default of true
@param[in] is_columnar If the dataframe should be column orientation or not with default of true
@param[in] shard_size_perc Different shard span configurations.  Options include 5, 20, and 100 with a default of 100
*/
int main(int argc, char** argv)
{
	// Parse the user commands
	std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
	int data_size = 1296;
	bool in_memory = true;
  bool is_columnar = true;
	double shard_span_perc = 1;
  int n_engines = 2;
  parseCmdArgsDataFrame(argc, argv, data_dir, data_size, in_memory, is_columnar, shard_span_perc, n_engines);

  // Setup the device
  Eigen::ThreadPool pool(n_engines);
  Eigen::ThreadPoolDevice device(&pool, n_engines);

	// Setup the Benchmarking suite, TensorCollectionGenerator, and run the application
  BenchmarkDataFrame1TimePointCpu benchmark_1_tp;
  DataFrameTensorCollectionGeneratorCpu tensor_collection_generator;
  runBenchmarkDataFrame(data_dir, data_size, in_memory, is_columnar, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);

	return 0;
}