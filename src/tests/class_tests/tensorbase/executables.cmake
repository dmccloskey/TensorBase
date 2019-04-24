set(core_executables_list
  Helloworld_test
  Statistics_test
  StringParsing_test
  TupleAlgorithms_test
)

set(io_executables_list
  CSVWriter_test
  DataFile_test
)

set(ml_executables_list
  TensorAxis_test
  TensorCollection_test
  TensorDimension_test
  TensorData_test
  TensorOperation_test
  TensorTable_test
  TensorType_test
)

### collect test executables
set(TEST_executables
    ${core_executables_list}
    ${io_executables_list}
    ${ml_executables_list}
)
