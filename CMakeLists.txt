cmake_minimum_required(VERSION 3.9)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(342pr2 VERSION 0.1.0 LANGUAGES CXX CUDA)

add_executable(sequential "${CMAKE_CURRENT_SOURCE_DIR}/src/seq_main.cpp")
add_executable(parallel "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi_main.cpp")
add_executable(cuda "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_main.cu")

find_package(MPI REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("BUILDING DEBUG!")
    target_compile_options(sequential
            PUBLIC -W -Wall -Wextra -g -Og -DDebug -pedantic -pedantic-errors)
    target_compile_options(parallel
            PUBLIC -W -Wall -Wextra -g -Og -DDebug -pedantic -pedantic-errors)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    message("BUILDING RELEASE!")
    target_compile_options(sequential
            PUBLIC -DNDEBUG -O2)
    target_compile_options(parallel
            PUBLIC -DNDEBUG -O2)
endif()

# Silence CUDA Warnings
target_compile_options(cuda
        PUBLIC -w)

target_link_libraries(sequential PUBLIC MPI::MPI_CXX)
target_link_libraries(parallel PUBLIC MPI::MPI_CXX)
target_link_libraries(cuda PUBLIC MPI::MPI_CXX)

target_compile_definitions(sequential PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")
target_compile_definitions(parallel PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")
target_compile_definitions(cuda PUBLIC RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/res/")

target_compile_definitions(sequential PUBLIC SEQUENTIAL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sequential_output/")
target_compile_definitions(parallel PUBLIC SEQUENTIAL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/sequential_output/")
target_compile_definitions(parallel PUBLIC PARALLEL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/parallel_output/")
target_compile_definitions(cuda PUBLIC CUDA_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/cuda_output/")
target_compile_definitions(cuda PUBLIC PARALLEL_OUTPUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/parallel_output/")

target_include_directories(sequential PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(parallel PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
