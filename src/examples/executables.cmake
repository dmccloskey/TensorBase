set(benchmark_executables_list
  BenchmarkPixels
)

set(cuda_executables_list
  CUDA_example
)

### collect example executables
set(EXAMPLE_executables
  ${benchmark_executables_list}
  ${cuda_executables_list}
)
