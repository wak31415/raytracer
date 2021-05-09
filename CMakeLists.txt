cmake_minimum_required(VERSION 3.13)
project(raytracer LANGUAGES CXX CUDA)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(scene SHARED
    src/raytrace.cu
    src/scene.cu
)

#include_directories(include)
target_link_libraries(scene PRIVATE cudart nlohmann_json::nlohmann_json)
target_include_directories(scene PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_compile_features(scene PUBLIC cxx_std_17)
set_target_properties(scene PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_FILES 
    src/main.cu
)

add_executable(render ${SOURCE_FILES})
target_link_libraries(render scene cudart)
set_target_properties(render PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET scene PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET render PROPERTY CUDA_ARCHITECTURES 61)