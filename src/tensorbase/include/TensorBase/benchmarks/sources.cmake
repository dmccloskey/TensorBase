### the directory name
set(directory include/TensorBase/benchmarks)

### list all header files of the directory here
set(sources_list_h
	BenchmarkArray.h
	BenchmarkArrayCpu.h
	BenchmarkArrayDefaultDevice.h
	BenchmarkArrayGpu.h
	BenchmarkDataFrame.h
	BenchmarkDataFrameCpu.h
	BenchmarkDataFrameDefaultDevice.h
	BenchmarkDataFrameGpu.h
	BenchmarkPixels.h
	BenchmarkPixelsCpu.h
	BenchmarkPixelsDefaultDevice.h
	BenchmarkPixelsGpu.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\TensorBase\\benchmarks" FILES ${sources_h})

set(TensorBase_sources_h ${TensorBase_sources_h} ${sources_h})

