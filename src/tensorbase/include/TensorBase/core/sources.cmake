### the directory name
set(directory include/TensorBase/core)

### list all header files of the directory here
set(sources_list_h
	Helloworld.h
	Statistics.h
	StringParsing.h
	TupleAlgorithms.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\TensorBase\\core" FILES ${sources_h})

set(TensorBase_sources_h ${TensorBase_sources_h} ${sources_h})

