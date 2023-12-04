cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(softActorCritic)

set(CMAKE_CXX_STANDARD 17)

# Include directories for JSONcpp
include_directories("./lib/JsonCpp")

# Set the path to libtorch
set(TORCH_CUDA_ENABLED OFF)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib/libtorch")
set(TORCH_CUDA_ENABLED OFF)
find_package(Torch)

file(GLOB_RECURSE
        softActorCritic_files
        ./src/*.cpp
        ./src/*.h
        ./lib/*.cpp
        )

add_executable(softActorCritic ${softActorCritic_files})
target_link_libraries(softActorCritic ${TORCH_LIBRARIES})
target_compile_definitions(softActorCritic PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

set_property(TARGET softActorCritic PROPERTY CXX_STANDARD 17)
