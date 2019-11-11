/**TODO:  Add copyright*/

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkPixelsCpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

/* Benchmark for toy 4D pixels data where x, y, and z describe the coordinates of the pixel in 3D space (type=int),
	t describes the time of the pixel (type=int), and the value of the pixel (from 0 to 255) describes the intensity of the pixel
*/
int main(int argc, char** argv)
{
	// Parse the user commands
	std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
	int n_dims = 2;
	int data_size = 1296;
	bool in_memory = true;
	double shard_span_perc = 1;
  int n_engines = 2;
  parseCmdArgs(argc, argv, data_dir, n_dims, data_size, in_memory, shard_span_perc, n_engines);

	// Setup the Benchmarking suite
	Benchmark1TimePointCpu<int, float> benchmark_1_tp;

	// Setup the TensorCollectionGenerator
	TensorCollectionGeneratorCpu<int, float> tensor_collection_generator;

	// Setup the device
  Eigen::ThreadPool pool(n_engines);
  Eigen::ThreadPoolDevice device(&pool, n_engines);

	// run the application
	runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);

	return 0;
}