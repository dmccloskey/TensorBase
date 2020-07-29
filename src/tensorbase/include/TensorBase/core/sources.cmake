### the directory name
set(directory include/TensorBase/core)

### list all header files of the directory here
set(sources_list_h
	Helloworld.h
	GraphAlgorithms.h
	GraphAlgorithmsCpu.h
	GraphAlgorithmsDefaultDevice.h
	GraphAlgorithmsGpu.h
	GraphGenerators.h
	GraphGeneratorsCpu.h
	GraphGeneratorsDefaultDevice.h
	GraphGeneratorsGpu.h
	Statistics.h
	StringParsing.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\TensorBase\\core" FILES ${sources_h})

set(TensorBase_sources_h ${TensorBase_sources_h} ${sources_h})

