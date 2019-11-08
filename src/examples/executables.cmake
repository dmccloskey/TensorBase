set(benchmark_executables_list
  BenchmarkPixels
  BenchmarkPixelsCpu
)

set(cuda_executables_list
  CUDA_example
  BenchmarkPixelsGpu
)

### collect example executables
set(EXAMPLE_executables
  ${benchmark_executables_list}
  ${cuda_executables_list}
)
