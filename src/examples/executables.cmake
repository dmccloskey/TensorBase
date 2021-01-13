set(benchmark_executables_list
  BenchmarkArray
  BenchmarkArrayCpu
  BenchmarkDataFrame
  BenchmarkDataFrameCpu
  BenchmarkGraph
  BenchmarkGraphCpu
  BenchmarkPixels
  BenchmarkPixelsCpu
)

set(cuda_executables_list
  BenchmarkArrayGpu
  BenchmarkDataFrameGpu
  BenchmarkGraphGpu
  BenchmarkPixelsGpu
)

### collect example executables
set(EXAMPLE_executables
  ${benchmark_executables_list}
  ${cuda_executables_list}
)
