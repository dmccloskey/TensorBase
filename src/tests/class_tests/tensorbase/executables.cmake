set(benchmarks_executables_list
  BenchmarkPixelsCpu_test
  BenchmarkPixelsDefaultDevice_test
  BenchmarkPixelsGpu_test
)

set(core_executables_list
  Helloworld_test
  Statistics_test
  StringParsing_test
)

set(io_executables_list
  CSVWriter_test
  DataFile_test
  TensorTableFile_test
)

set(ml_executables_list
  TensorArray_test
  TensorArrayGpu_test
  TensorAxis_test
  TensorAxisCpu_test
  TensorAxisGpu_test
  TensorClauses_test
  TensorCollection_test
  TensorCollectionGpu_test
  TensorData_test
  TensorDataCpu_test
  TensorDataGpu_test
  TensorDimension_test
  TensorDimensionCpu_test
  TensorDimensionGpu_test
  TensorOperation_test
  TensorShard_test
  TensorSelect_test
  TensorTable_test
  TensorTableCpu_test
  TensorTableGpuClassT_test
  TensorTableGpuPrimitiveT_test
  TensorType_test
  TransactionManager_test
)

### collect test executables
set(TEST_executables
	${benchmarks_executables_list}
    ${core_executables_list}
    ${io_executables_list}
    ${ml_executables_list}
)
