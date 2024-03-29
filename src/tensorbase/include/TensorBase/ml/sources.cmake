### the directory name
set(directory include/TensorBase/ml)

### list all header files of the directory here
set(sources_list_h
	TensorArray.h
	TensorArrayGpu.h
	TensorAxis.h
	TensorAxisConcept.h
	TensorAxisConceptCpu.h
	TensorAxisConceptDefaultDevice.h
	TensorAxisConceptGpu.h
	TensorAxisCpu.h
	TensorAxisDefaultDevice.h
	TensorAxisGpu.h
	TensorClauses.h
	TensorCollection.h
	TensorCollectionCpu.h
	TensorCollectionDefaultDevice.h
	TensorCollectionGpu.h
	TensorDimension.h
	TensorDimensionConcept.h
	TensorDimensionConceptCpu.h
	TensorDimensionConceptDefaultDevice.h
	TensorDimensionConceptGpu.h
	TensorDimensionCpu.h
	TensorDimensionDefaultDevice.h
	TensorDimensionGpu.h
	TensorData.h
	TensorDataCpu.h
	TensorDataDefaultDevice.h
	TensorDataGpu.h
	TensorFunctor.h
	TensorOperation.h
	TensorOperationCpu.h
	TensorOperationDefaultDevice.h
	TensorOperationGpu.h
	TensorShard.h
	TensorTable.h
	TensorTableConcept.h
	TensorTableConceptCpu.h
	TensorTableConceptDefaultDevice.h
	TensorTableConceptGpu.h
	TensorTableCpu.h
	TensorTableDefaultDevice.h
	TensorTableGpuClassT.h
	TensorTableGpuPrimitiveT.h
	TensorType.h
	TransactionManager.h
)

### add path to the filenames
set(sources_h)
foreach(i ${sources_list_h})
	list(APPEND sources_h ${directory}/${i})
endforeach(i)

### source group definition
source_group("Header Files\\TensorBase\\ml" FILES ${sources_h})

set(TensorBase_sources_h ${TensorBase_sources_h} ${sources_h})

