set(benchmark_executables_list
  BenchmarkArray
  BenchmarkArrayCpu
  BenchmarkPixels
  BenchmarkPixelsCpu
)

set(cuda_executables_list
  CUDA_example
  BenchmarkArrayGpu
  BenchmarkPixelsGpu
)

### collect example executables
set(EXAMPLE_executables
  ${benchmark_executables_list}
  ${cuda_executables_list}
)
