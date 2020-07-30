/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkGraphGpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

/* Benchmark for a Graph analytics use case with tables for Sparse indices represented as node in/out pairs, weights, node properties, and link properties.
	A Kronecker graph generator is used to simulate the node and links according the scale and edge_factor parameters, 
	weights are randomly assigned in the range of [0, 1], and node and link properties are randomly assigned and represented by a string value.

Example usage:
	BenchmarkGraph [data_dir] [scale] [edge_factor] [in_memory] [shard_size_perc] [n_engines]
	BenchmarkGraph C:/Users/dmccloskey/Documents/GitHub/mnist/ S S true 100 2

@param[in] scale Options include XS, small, medium, large, and XL (i.e., 8, 14, 16, 20, 24, and 26, respectively) with default of small
@param[in] edge_factor Options include only 16 for now
@param[in] in_memory Simulate all data loaded into memory (true) or JIT load into memory from disk (false) with default of true
@param[in] shard_size_perc Different shard span configurations.  Options include 5, 20, and 100 with a default of 100
@param[in] n_engines The number of CPUs or GPUs to use
*/
int main(int argc, char** argv)
{
	// Parse the user commands
	std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
	int scale = 8;
	int edge_factor = 16;
	bool in_memory = true;
	double shard_span_perc = 1;
  int n_engines = 1;
  parseCmdArgsGraph(argc, argv, data_dir, scale, edge_factor, in_memory, shard_span_perc, n_engines);

  // Setup the device
	cudaStream_t stream;
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	// Setup the Benchmarking suite, TensorCollectionGenerator, and run the application
	BenchmarkGraph1LinkGpu benchmark_1_link;
  GraphTensorCollectionGeneratorGpu tensor_collection_generator;
  runBenchmarkGraph(data_dir, scale, edge_factor, in_memory, shard_span_perc, benchmark_1_link, tensor_collection_generator, device);

	return 0;
}
#endif