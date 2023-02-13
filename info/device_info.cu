//
// Created by emman on 13/02/2023.
//

#include "device_info.cuh"
#include <stdio.h>

/**
 * Displays information about the CUDA device
 */
void displayGPUHeader() {
    const int kb = 1024;
    const int mb = kb * kb;
    printf("NBody.GPU\n=========\n\n");

    printf("CUDA version:   v%i\n", CUDART_VERSION);

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Devices: \n\n");

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props{};
        cudaGetDeviceProperties(&props, i);
        printf("Device %i: %s: %i.%i\n", i, props.name, props.major, props.minor);
        printf("\tGlobal memory:\t\t%zu mb\n", props.totalGlobalMem / mb);
        printf("\tShared memory:\t\t%zu kb\n", props.sharedMemPerBlock / kb);
        printf("\tConstant memory:\t%zu kb\n", props.totalConstMem / kb);
        printf("\tBlock registers:\t%i\n\n", props.regsPerBlock);

        printf("\tWarp size:\t\t%i\n", props.warpSize);
        printf("\tThreads per block:\t%i\n", props.maxThreadsPerBlock);
        printf("\tMax block dimensions:\t[ %i,\t\t%i,\t%i ]\n", props.maxThreadsDim[0], props.maxThreadsDim[1],
               props.maxThreadsDim[2]);
        printf("\tMax grid dimensions:\t[ %i,\t%i,\t%i ]\n", props.maxGridSize[0], props.maxGridSize[1],
               props.maxGridSize[2]);
        printf("\n");
    }
}