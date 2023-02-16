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
    int filter = blockIdx.z;
    int i = threadIdx.x + 1;
    int j = threadIdx.y + 1;

    __shared__ float image[IMAGE_INPUT_SIZE];
    __shared__ float network[NUM_WEIGHTS];
    __shared__ float conv[CONV_IMAGE_SIZE];
    //__shared__ float pooled[POOLED_IMAGE_SIZE_TOT];
    __shared__ float output[OUTPUT_PER_FILTER];

    // copy the image to the block shared memory
    image[threadIdx.x * 28 + threadIdx.y] = images[imageIndex * 28 * 28 + threadIdx.x * 28 + threadIdx.y];

    // copy the network to the block shared memory. I need to copy only the filters (so the first 45 weights)
    int weightIndex = blockIdx.x;
    if (weightIndex < 45) {
        network[weightIndex] = networks[networkIndex * NUM_WEIGHTS + weightIndex];
    }

    __syncthreads();

    float sum = 0;

    //TODO: merge convolution and pooling to get rid of the __shared__ float conv[CONV_IMAGE_SIZE]; variable

    // --- CONVOLUTION ---

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

    // can I avoid this access?
    conv[i * 26 + j * 26] = sum;

    __syncthreads();

    // --- MAX POOLING ---
    if (i < 26 && i % 2 == 1 && j < 26 && j % 2 == 1) {
        int ii_1 = (i - 1) * 26;
        int ii_2 = i * 26;
        float v1 = conv[ii_1 + j - 1];
        float v2 = conv[ii_1 + j];
        float v3 = conv[ii_1 + j + 1];
        float v4 = conv[ii_2 + j - 1];
        float v5 = conv[ii_2 + j];
        float v6 = conv[ii_2 + j + 1];

        float m1 = max(v1, v2);
        float m2 = max(v3, v4);
        float m3 = max(v5, v6);

        float m4 = max(m1, m2);
        float m6 = max(m3, m4);

        // I can avoid this access and directly use the value to calculate the partial output
        //pooled[i * 13 + j * 13] = m6;

        for (int out = 0; out < 10; out++) {
            output[filter * 10 + out] = m6 * network[45 + out * 13 * 13 * 5 + i * 13 + j * 13];
        }
    }
    __syncthreads();

    // Now I have all the partial outputs for this network and this image, for every filter.
    // It's time to sum them together and calculate the fitness

    // Only one thread per block is responsible to sum
    if (i == 1 && j == 1) {
        float results[10];
        int maxIndex = 0;
        results[0] = output[0] + output[10 + 0] + output[20 + 0] + output[30 + 0] + output[40 + 0];
        results[1] = output[1] + output[10 + 1] + output[20 + 1] + output[30 + 1] + output[40 + 1];
        results[2] = output[2] + output[10 + 2] + output[20 + 2] + output[30 + 2] + output[40 + 2];
        results[3] = output[3] + output[10 + 3] + output[20 + 3] + output[30 + 3] + output[40 + 3];
        results[4] = output[4] + output[10 + 4] + output[20 + 4] + output[30 + 4] + output[40 + 4];
        results[5] = output[5] + output[10 + 5] + output[20 + 5] + output[30 + 5] + output[40 + 5];
        results[6] = output[6] + output[10 + 6] + output[20 + 6] + output[30 + 6] + output[40 + 6];
        results[7] = output[7] + output[10 + 7] + output[20 + 7] + output[30 + 7] + output[40 + 7];
        results[8] = output[8] + output[10 + 8] + output[20 + 8] + output[30 + 8] + output[40 + 8];
        results[9] = output[9] + output[10 + 9] + output[20 + 9] + output[30 + 9] + output[40 + 9];
        if (results[0] > results[maxIndex]) {
            maxIndex = 0;
        }
        if (results[1] > results[maxIndex]) {
            maxIndex = 1;
        }
        if (results[2] > results[maxIndex]) {
            maxIndex = 2;
        }
        if (results[3] > results[maxIndex]) {
            maxIndex = 3;
        }
        if (results[4] > results[maxIndex]) {
            maxIndex = 4;
        }
        if (results[5] > results[maxIndex]) {
            maxIndex = 5;
        }
        if (results[6] > results[maxIndex]) {
            maxIndex = 6;
        }
        if (results[7] > results[maxIndex]) {
            maxIndex = 7;
        }
        if (results[8] > results[maxIndex]) {
            maxIndex = 8;
        }
        if (results[9] > results[maxIndex]) {
            maxIndex = 9;
        }

        // finally, maxIndex is the prediction of this network for this image!
    }
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

    // grid = data, network and filter indexes
    dim3 grid(dataCount, networkCount, NUM_FILTERS);

    // 1 block = 1 network with 1 input image, 26x26 threads
    // To be able to sync the numFilters and avoid saving the conv in shared memory,
    // I could launch 13x13x5 threads, so < 1024
    dim3 block(26, 26);

    calculateConvolutionGPU<<<grid, block>>>(d_images, d_networks);

    CHECK(cudaDeviceSynchronize());
}