cmake_minimum_required(VERSION 3.24)
project(ga_cnn CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(ga_cnn main.cu args_handler/args_handler_help.cu args_handler/args_handler_help.cuh args_handler/args_handler.cu args_handler/args_handler.cuh version.cuh args_handler/args_handler_version.cu args_handler/args_handler_version.cuh args_handler/args_extractor.cu args_handler/args_extractor.cuh data/data_loader.cu data/data_loader.cuh data/file_loader.cu data/file_loader.cuh data/data_loader_cpu.cu data/data_loader_cpu.cuh data/data_loader_cpu.cuh data/data_utils.cu data/data_utils.cuh data/data_loader_gpu.cu data/data_loader_gpu.cuh)

set_target_properties(ga_cnn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
