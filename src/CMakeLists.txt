add_library(bloomlib-cuda STATIC
  bloom.cu
  bloom.h
)

target_link_libraries(bloomlib-cuda murmurlib-cuda)
target_include_directories(bloomlib-cuda PUBLIC ../${depend_src})

target_compile_features(bloomlib-cuda PUBLIC cxx_std_11)
set_target_properties(bloomlib-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
