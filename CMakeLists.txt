cmake_minimum_required(VERSION 3.24)
project(ga_cnn CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "--ptxas-options=-v --maxrregcount 16")

add_executable(ga_cnn main.cu args_handler/args_handler_help.cu args_handler/args_handler_help.cuh args_handler/args_handler.cu args_handler/args_handler.cuh version.cuh args_handler/args_handler_version.cu args_handler/args_handler_version.cuh args_handler/args_extractor.cu args_handler/args_extractor.cuh data/data_loader.cu data/data_loader.cuh data/file_loader.cu data/file_loader.cuh data/data_loader_cpu.cu data/data_loader_cpu.cuh data/data_loader_cpu.cuh data/data_utils.cu data/data_utils.cuh data/data_loader_gpu.cu data/data_loader_gpu.cuh info/device_info.cu info/device_info.cuh network/init_networks.cu network/init_networks.cuh network/init_networks_utils.cu network/init_networks_utils.cuh network/fitness_calculator.cu network/fitness_calculator.cuh network/fitness_calculator_gpu.cu network/fitness_calculator_gpu.cuh network/fitness_calculator_cpu.cu network/fitness_calculator_cpu.cuh output/output_utils.cu output/output_utils.cuh genetic/evolve_population.cu genetic/evolve_population.cuh genetic/evolve_population_cpu.cu genetic/evolve_population_cpu.cuh genetic/evolve_population_gpu.cu genetic/evolve_population_gpu.cuh defines.cu defines.cuh)

set_target_properties(ga_cnn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)