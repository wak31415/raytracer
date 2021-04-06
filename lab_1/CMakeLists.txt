cmake_minimum_required(VERSION 3.13)
project(lab_1 LANGUAGES CXX CUDA)

set(SOURCE_FILES 
    src/main.cu
    src/write_image.cpp
)

include_directories(include)

add_executable(lab_1 ${SOURCE_FILES})

set_property(TARGET lab_1 PROPERTY CUDA_ARCHITECTURES 52 61)

target_link_libraries(lab_1 cudart)