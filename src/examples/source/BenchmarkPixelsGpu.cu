/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkPixelsGpu.h>

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

/* Benchmark for toy 4D pixels data where x, y, and z describe the coordinates of the pixel in 3D space (type=int),
  t describes the time of the pixel (type=int), and the value of the pixel (from 0 to 255) describes the intensity of the pixel

Example usage:
  pixels_benchmark [data_dir] [n_dims] [data_size] [in_memory] [shard_size_perc] [labels_type] [tensor_type]
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
@param[in] labels_type The type of the labels.  Options include int, float, double with a default of int.
@param[in] tensor_type The type of the tensor data.  Options include int, float, double with a default of int.
*/
int main(int argc, char** argv)
{
  // Parse the user commands
  std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
  int n_dims = 1;
  int data_size = 1296;
  bool in_memory = true;
  double shard_span_perc = 1;
  int n_engines = 1;
  std::string labels_type = "int";
  std::string tensor_type = "int";
  parseCmdArgs(argc, argv, data_dir, n_dims, data_size, in_memory, shard_span_perc, n_engines, labels_type, tensor_type);

  // Setup the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Setup the Benchmarking suite, TensorCollectionGenerator, and run the application
  // NOTE: currently 0D requires labesl_type == tensor_type == int
  if (labels_type == "int" && tensor_type == "int") {
    Benchmark1TimePointGpu<int, int> benchmark_1_tp;
    TensorCollectionGeneratorGpu<int, int> tensor_collection_generator;
    runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);
  }
  else if (labels_type == "int" && tensor_type == "float") {
    Benchmark1TimePointGpu<int, float> benchmark_1_tp;
    TensorCollectionGeneratorGpu<int, float> tensor_collection_generator;
    runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);
  }
  else if (labels_type == "int" && tensor_type == "double") {
    Benchmark1TimePointGpu<int, double> benchmark_1_tp;
    TensorCollectionGeneratorGpu<int, double> tensor_collection_generator;
    runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);
  }
  else {
    Benchmark1TimePointGpu<int, int> benchmark_1_tp;
    TensorCollectionGeneratorGpu<int, int> tensor_collection_generator;
    runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);
  }

	return 0;
}
#endif