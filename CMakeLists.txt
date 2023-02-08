cmake_minimum_required(VERSION 3.24)
project(ga_cnn CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(ga_cnn main.cu)

set_target_properties(ga_cnn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
