cmake_minimum_required(VERSION 3.18)
project(culip LANGUAGES CXX CUDA)

find_package(CUDA 12.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Git REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
  set(CMAKE_${lang}_STANDARD          14)
  set(CMAKE_${lang}_STANDARD_REQUIRED ON)
  set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/CULiP/*.hpp")

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda-exponent-distribution-statistics/Makefile)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
endif()

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda-cutoff-small-abs-values/Makefile)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
endif()

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/json/CMakeLists.txt)
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
endif()


foreach(library cublas)
  set(lib_name culip_${library})
  set(exp_lib_home ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda-exponent-distribution-statistics)
  set(cutoff_lib_home ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda-cutoff-small-abs-values)
  include_directories(${lib_name} PRIVATE ${exp_lib_home}/include ${exp_lib_home}/src/cutf/include)
  include_directories(${lib_name} PRIVATE ${cutoff_lib_home}/include ${cutoff_lib_home}/src/cutf/include)
  add_library(${lib_name} SHARED
    ${SRCDIR}/${library}.cu
    ${SRCDIR}/utils.cu
    ${SRCDIR}/utils.hpp
    ${SRCDIR}/params.hpp
    ${exp_lib_home}/src/main.cu
    ${cutoff_lib_home}/src/main.cu
    ${HEADERS}
    )

  target_include_directories(${lib_name} PUBLIC ${INCDIR})
  target_link_libraries(${lib_name} PRIVATE
    cuda
    cublas
    cublasLt_static
    culibos
    )

  set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER ${INCDIR}/CULiP/${library}.hpp)
endforeach()

##########################################################################
# Installing
##########################################################################
install(TARGETS culip_cublas
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/CULiP
  )

##########################################################################
# Tests
##########################################################################

# Directory
set(TESTSRCDIR tests)

foreach(test cublas)
  set(test_out ${test}.test)
  add_executable(${test_out} ${TESTSRCDIR}/${test}_test.cu ${HEADERS})
  target_include_directories(${test_out} PRIVATE ${INCDIR})
  target_link_libraries(${test_out} PRIVATE
    culip_${test}
    CUDA::${test}
    )
endforeach()

##########################################################################
# Analyzer
##########################################################################

add_executable(CULiP_analyzer ${SRCDIR}/analyzer.cpp ${SRCDIR}/params.hpp)
target_include_directories(CULiP_analyzer PRIVATE ${INCDIR})
set_target_properties(CULiP_analyzer PROPERTIES RUNTIME CULiP_analyzer)

##########################################################################
# ExpStats Analyzer
##########################################################################

add_executable(CULiP_exp_stats_analyzer ${SRCDIR}/exp_stats_analyzer.cpp)
set(jsonlib ${CMAKE_CURRENT_SOURCE_DIR}/src/json)
include_directories(CULiP_exp_stats_analyzer PRIVATE ${jsonlib}/include)
target_include_directories(CULiP_exp_stats_analyzer PRIVATE ${INCDIR})

##########################################################################
# Installing
##########################################################################
install(TARGETS culip_cublas CULiP_analyzer
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/CULiP
  RUNTIME DESTINATION bin
  )
