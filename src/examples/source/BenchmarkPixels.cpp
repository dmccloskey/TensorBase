/**TODO:  Add copyright*/

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkPixelsDefaultDevice.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

/* Benchmark for toy 4D pixels data where x, y, and z describe the coordinates of the pixel in 3D space (type=int),
	t describes the time of the pixel (type=int), and the value of the pixel (from 0 to 255) describes the intensity of the pixel

Example usage:
	pixels_benchmark [data_dir] [n_dims] [data_size] [in_memory] [shard_size_perc] 
	pixels_benchmark C:/Users/dmccloskey/Documents/GitHub/mnist/ 1 S true 100

@param[in] n_dims The number of dimensions (i.e., 1-4) with default of 4
	1 dimension: x, y, z, and t on a single axis with a "values" dimensions on the other axis
	2 dimensions: x, y, z on a single axis, and t on another axis
	3 dimensions: y, z on a single axis, x on an axis, and t on an axis
	4 dimensions: x, y, z, and t on seperate axes
@param[in] data_size Options include small, medium, large, and XL (i.e., 1296, 104976, 1048576, 10556001, 1003875856, and 1e12 pixels, respectively) with default of small
	where x, y, z, and t span 1 to 6, 18, 32, 57, 178, and 1000 respectively
@param[in] in_memory Simulate all data loaded into memory (true) or JIT load into memory from disk (false) with default of true
@param[in] shard_size_perc Different shard span configurations.  Options include 5, 20, and 100 with a default of 100
*/
int main(int argc, char** argv)
{
	// Parse the user commands
	std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
	int n_dims = 0;
	int data_size = 1296;
	bool in_memory = true;
	double shard_span_perc = 1;
  int n_engines = 1;
  parseCmdArgs(argc, argv, data_dir, n_dims, data_size, in_memory, shard_span_perc, n_engines);

	// Setup the Benchmarking suite
	//Benchmark1TimePointDefaultDevice<int, float> benchmark_1_tp;
	Benchmark1TimePointDefaultDevice<int, int> benchmark_1_tp; // 0D only

	// Setup the TensorCollectionGenerator
	//TensorCollectionGeneratorDefaultDevice<int, float> tensor_collection_generator;
	TensorCollectionGeneratorDefaultDevice<int, int> tensor_collection_generator; // 0D only

	// Setup the device
	Eigen::DefaultDevice device;

	// run the application
	runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);

	return 0;
}