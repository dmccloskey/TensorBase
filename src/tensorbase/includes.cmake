set(TensorBase_sources  CACHE INTERNAL "This variable should hold all TensorBase sources at the end of the config step" )

## ATTENTION: The order of includes should be similar to the inclusion hierarchy
include(source/core/sources.cmake)
include(source/io/sources.cmake)

set(TensorBase_sources_h  CACHE INTERNAL "This variable should hold all TensorBase sources at the end of the config step" )

## ATTENTION: The order of includes should be similar to the inclusion hierarchy
include(include/TensorBase/benchmarks/sources.cmake)
include(include/TensorBase/core/sources.cmake)
include(include/TensorBase/io/sources.cmake)
include(include/TensorBase/ml/sources.cmake)

## add configured config.h&Co to source group
source_group("Header Files\\TensorBase" FILES ${TensorBase_configured_headers})
## merge all headers to sources (for source group view in VS)
list(APPEND TensorBase_sources ${TensorBase_sources_h} ${TensorBase_configured_headers})

# TODO track why the duplicate warnings are thrown for all (!) MOC sources
# Macro problem?
list(REMOVE_DUPLICATES TensorBase_sources)
