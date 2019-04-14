set(core_executables_list
  Helloworld_test
  Statistics_test
  StringParsing_test
)

set(io_executables_list
  CSVWriter_test
  DataFile_test
)

set(ml_executables_list
  TensorData_test
)

### collect test executables
set(TEST_executables
    ${core_executables_list}
    ${io_executables_list}
    ${ml_executables_list}
)
