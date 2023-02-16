//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_gpu.cuh"
#include "../defines.cuh"

#include <stdio.h>


__global__ void calculateConvolutionGPU(
        const float *images,
        const float *networks
) {
    int imageIndex = blockIdx.x;
    int networkIndex = blockIdx.y;
    int i = threadIdx.x + 1;
    int j = threadIdx.y + 1;
    int filter = threadIdx.z;

    __shared__ float image[IMAGE_INPUT_SIZE];
    __shared__ float network[NUM_WEIGHTS];
    __shared__ float conv[CONV_IMAGE_SIZE];

    // copy the image to the block shared memory
    image[threadIdx.x * 28 + threadIdx.y] = images[imageIndex * 28 * 28 + threadIdx.x * 28 + threadIdx.y];

    // copy the network to the block shared memory. I need to copy only the filters (so the first 45 weights)
    int weightIndex = blockIdx.x;
    if (weightIndex < 45) {
        network[weightIndex] = networks[networkIndex * NUM_WEIGHTS + weightIndex];
    }

    __syncthreads();

    float sum = 0;

    // Split this for in multiple threads with threadId.z? Or loop unrolling?
    int start = filter * 9;
    int i_1 = (i - 1) * 28;
    int j_1 = j - 1;
    int i_2 = i * 28;
    int i_3 = (i + 1) * 28;
    int j_3 = j + 1;
    sum += image[i_1 + j_1] * network[start];
    sum += image[i_1 + j] * network[start + 1];
    sum += image[i_1 + j_3] * network[start + 2];

    sum += image[i_2 + j_1] * network[start + 3];
    sum += image[i_2 + j] * network[start + 4];
    sum += image[i_2 + j_3] * network[start + 5];

    sum += image[i_3 + j_1] * network[start + 6];
    sum += image[i_3 + j] * network[start + 7];
    sum += image[i_3 + j_3] * network[start + 8];

    int count = i * 26 + j * 26;

    conv[count] = sum;

    __syncthreads();
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

    // grid = data and network indexes
    dim3 grid(dataCount, networkCount);

    // 1 block = 1 network with 1 input image, 26x26xFILTER_SIZE threads
    dim3 block(26, 26, FILTER_SIZE);

    calculateConvolutionGPU<<<grid, block>>>(d_images, d_networks);

    CHECK(cudaDeviceSynchronize());
}