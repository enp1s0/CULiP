cmake_minimum_required(VERSION 3.18)
project(culip LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          14)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/CULiP/*.hpp")

foreach(library cublas)
	SET(lib_name culip_${library})
	add_library(${lib_name} SHARED
		${SRCDIR}/${library}.cu
		${SRCDIR}/utils.cu
		${SRCDIR}/utils.hpp
		${SRCDIR}/params.hpp
		${HEADERS}
		)

	target_include_directories(${lib_name} PUBLIC ${INCDIR})
	target_link_libraries(${lib_name} PRIVATE
		cuda
		cublas_static
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
	target_link_libraries(${test_out} PRIVATE culip_${test})
	target_link_libraries(${test_out} PRIVATE
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
# Installing
##########################################################################
install(TARGETS culip_cublas CULiP_analyzer
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/CULiP
	RUNTIME DESTINATION bin
	)
