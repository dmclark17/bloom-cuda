cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(bloom-cuda LANGUAGES CXX CUDA)

#add_subdirectory(src)
add_subdirectory(dependencies)

add_executable(main main.cu)

target_link_libraries(main murmurlib-cuda)
target_include_directories(main PUBLIC src)
target_include_directories(main PUBLIC ${depend_src})

set_target_properties(main PROPERTIES CXX_STANDARD 11)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
