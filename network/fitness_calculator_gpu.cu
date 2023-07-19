//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_gpu.cuh"
#include "../defines.cuh"

#include <stdio.h>


__global__ void calculateConvolutionGPU(
        const float *images,
        const float *networks,
        const int *labels,
        float *fitness
) {
    __shared__ float image[IMAGE_INPUT_SIZE];
    __shared__ float network[45];
    __shared__ float maxPooled[POOLED_SIZE];

    unsigned int xx;
    unsigned int yy;
    unsigned int reused;

    // I have 14x14=196 threads
    // but the image is 28x28=784 pixels
    // so each one will copy 4 pixels
#pragma unroll
    for (xx = 0; xx < 845; xx += 169) {
        reused = threadIdx.x * 13 + threadIdx.y + xx;
        image[threadIdx.x * 13 + threadIdx.y + xx] = images[blockIdx.x * 28 * 28 + reused];
    }

    // copy the first 45 weights (the filters)
    xx = threadIdx.x * 13 + threadIdx.y;
    if (xx < 45) {
        network[xx] = networks[blockIdx.y * NUM_WEIGHTS + xx];
    }

    __syncthreads();


    // To avoid saving partial values in memory, I merge the convolution and pooling steps.
    // The thread i, j will take care of calculating the convolution of the 4 pixels:
    // (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    // and then it will take only the maximum value

    float pooled = 0;
    float sum = 0;

    // x = i, y = j
    xx = (threadIdx.x + 1) * 2;
    yy = (threadIdx.y + 1) * 2;


    unsigned int i_1 = (xx - 1) * 28;

#pragma unroll
    for (reused = 0; reused < 45; reused += 9) {
        pooled = 0;
        sum = 0;
        sum += image[i_1 + yy - 1] * network[reused];
        sum += image[i_1 + yy] * network[reused + 1];
        sum += image[i_1 + yy + 1] * network[reused + 2];

        sum += image[i_1 + 28 + yy - 1] * network[reused + 3];
        sum += image[i_1 + 28 + yy] * network[reused + 4];
        sum += image[i_1 + 28 + yy + 1] * network[reused + 5];

        sum += image[i_1 + 56 + yy - 1] * network[reused + 6];
        sum += image[i_1 + 56 + yy] * network[reused + 7];
        sum += image[i_1 + 56 + yy + 1] * network[reused + 8];

        pooled = max(sum, pooled);

        // x = i + 1, y = j

        sum = 0;
        sum += image[i_1 + 28 + yy - 1] * network[reused];
        sum += image[i_1 + 28 + yy] * network[reused + 1];
        sum += image[i_1 + 28 + yy + 1] * network[reused + 2];

        sum += image[i_1 + 56 + yy - 1] * network[reused + 3];
        sum += image[i_1 + 56 + yy] * network[reused + 4];
        sum += image[i_1 + 56 + yy + 1] * network[reused + 5];

        sum += image[i_1 + 84 + yy - 1] * network[reused + 6];
        sum += image[i_1 + 84 + yy] * network[reused + 7];
        sum += image[i_1 + 84 + yy + 1] * network[reused + 8];

        pooled = max(sum, pooled);

        // x = i, y = j + 1

        sum = 0;
        sum += image[i_1 + yy] * network[reused];
        sum += image[i_1 + yy + 1] * network[reused + 1];
        sum += image[i_1 + yy + 2] * network[reused + 2];

        sum += image[i_1 + 28 + yy] * network[reused + 3];
        sum += image[i_1 + 28 + yy + 1] * network[reused + 4];
        sum += image[i_1 + 28 + yy + 2] * network[reused + 5];

        sum += image[i_1 + 56 + yy] * network[reused + 6];
        sum += image[i_1 + 56 + yy + 1] * network[reused + 7];
        sum += image[i_1 + 56 + yy + 2] * network[reused + 8];

        pooled = max(sum, pooled);

        // x = i + 1, y = j + 1

        sum = 0;
        sum += image[i_1 + 28 + yy] * network[reused];
        sum += image[i_1 + 28 + yy + 1] * network[reused + 1];
        sum += image[i_1 + 28 + yy + 2] * network[reused + 2];

        sum += image[i_1 + 56 + yy] * network[reused + 3];
        sum += image[i_1 + 56 + yy + 1] * network[reused + 4];
        sum += image[i_1 + 56 + yy + 2] * network[reused + 5];

        sum += image[i_1 + 84 + yy] * network[reused + 6];
        sum += image[i_1 + 84 + yy + 1] * network[reused + 7];
        sum += image[i_1 + 84 + yy + 2] * network[reused + 8];

        pooled = max(sum, pooled);

        maxPooled[reused / 9 * 13 * 13 + threadIdx.x * 13 + threadIdx.y] = pooled;
    }

    __syncthreads();

    __shared__ float sums[10];
    __shared__ float max;
    __shared__ int index;

    max = -999;
    index = 0;

    // threadIdx.x is the index of the output neuron (0-9)
    // threadIx.y is the index of the filtered image (0-4)
    // this is a tradeoff between using all threads for the output calculation
    // and not having to many conflicts in the atomicAdd operation.
    if (threadIdx.x < 10 && threadIdx.y < 5) {
        yy = blockIdx.y * NUM_WEIGHTS + 45 + threadIdx.x * 13 * 13 * 5;
#pragma unroll
        for (xx = threadIdx.x * 10 + threadIdx.y; xx < 13 * 13 * 5; xx += 105) {
            atomicAdd(&sums[threadIdx.x], maxPooled[xx] * networks[yy + xx]);
        }
    }

    // At last, only one thread per block checks the output label
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (sums[0] > max) {
            max = sums[0];
            index = 0;
        }
        if (sums[1] > max) {
            max = sums[1];
            index = 1;
        }
        if (sums[2] > max) {
            max = sums[2];
            index = 2;
        }
        if (sums[3] > max) {
            max = sums[3];
            index = 3;
        }
        if (sums[4] > max) {
            max = sums[4];
            index = 4;
        }
        if (sums[5] > max) {
            max = sums[5];
            index = 5;
        }
        if (sums[6] > max) {
            max = sums[6];
            index = 6;
        }
        if (sums[7] > max) {
            max = sums[7];
            index = 7;
        }
        if (sums[8] > max) {
            max = sums[8];
            index = 8;
        }
        if (sums[9] > max) {
            index = 9;
        }

        if (index == labels[blockIdx.x]) {
            //fitness[blockIdx.y] += 1.0f / 600.0f;
            atomicAdd(&fitness[blockIdx.y], 1.0f / 600.0f);
        }
    }
}

void calculateFitnessGPU(
        const int *d_labels,
        const float *d_images,
        const float *networks,
        int networkCount,
        int dataCount,
        float *fitness) {

// copy data to gpu
    float *d_networks;
    float *d_fitness;

    cudaMalloc((void **) &d_networks, networkCount * NUM_WEIGHTS * sizeof(float));
    cudaMalloc((void **) &d_fitness, networkCount * sizeof(float));
    cudaMemset(d_fitness, 0, networkCount * sizeof(float));
    cudaMemcpy(d_networks, networks, networkCount
                                     * NUM_WEIGHTS * sizeof(float), H2D);

// grid = data and network index
    dim3 grid(dataCount, networkCount);

// 1 block = 1 network with 1 input image, 28x28x5 threads (too many)
// To be able to sync the numFilters and avoid saving the conv in shared memory,
// I could launch 13x13x5 threads, so < 1024
// But I use 14x14 to parallelize the copy of the 28x28 image in shared memory
// And avoid the x5 to divide by 5 the register usage
    dim3 block(14, 14);

    calculateConvolutionGPU<<<grid, block>>>(
            d_images,
            d_networks,
            d_labels,
            d_fitness
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(fitness, d_fitness, networkCount * sizeof(float), D2H));

    cudaFree(d_networks);
    cudaFree(d_fitness);
}