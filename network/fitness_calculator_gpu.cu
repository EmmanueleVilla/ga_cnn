//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_gpu.cuh"
#include "init_networks.cuh"
#include "../defines.cuh"

#include <stdio.h>


__global__ void calculateConvolutionGPU(
        const float *images,
        const float *networks
) {
    int imageIndex = blockIdx.x;
    int networkIndex = blockIdx.y;
    int filterIndex = blockIdx.z;
    int pixelIndex = threadIdx.x;
    printf("imageIndex: %d, networkIndex: %d, filterIndex: %d", imageIndex, networkIndex, filterIndex);
}

void calculateFitnessGPU(
        const int *labels,
        const float *images,
        const float *networks,
        int networkCount,
        int dataCount,
        float *fitness) {

    // copy data to gpu
    float *d_images;
    float *d_networks;
    int *d_labels;
    // TODO check if pinned memory is faster
    cudaMalloc((void **) &d_images, dataCount * 28 * 28 * sizeof(float));
    cudaMalloc((void **) &d_networks, networkCount * NUM_WEIGHTS * sizeof(float));
    cudaMalloc((void **) &d_labels, dataCount * sizeof(int));

    cudaMemcpy(d_images, images, dataCount * 28 * 28 * sizeof(float), H2D);
    cudaMemcpy(d_networks, networks, networkCount * NUM_WEIGHTS * sizeof(float), H2D);
    cudaMemcpy(d_labels, labels, dataCount * sizeof(int), H2D);

    calculateConvolutionGPU<<<(dataCount, networkCount, FILTER_SIZE), 26 * 26>>>(d_images, d_networks);
}