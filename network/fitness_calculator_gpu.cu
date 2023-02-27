//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_gpu.cuh"
#include "../defines.cuh"

#include <stdio.h>

__device__ int v = 0;

__global__ void calculateConvolutionGPU(
        const float *images,
        const float *networks,
        const int *labels,
        float *fitness
        //bool *d_big_fitness,
        //int networkCount
) {
    unsigned int imageIndex = blockIdx.x;
    unsigned int networkIndex = blockIdx.y;
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;
    unsigned int filter = threadIdx.z;

    __shared__ float image[IMAGE_INPUT_SIZE];
    __shared__ float network[NUM_WEIGHTS];
    __shared__ float maxPooled[POOLED_SIZE];

    unsigned int xx = threadIdx.x * 2;
    unsigned int yy = threadIdx.y * 2;
    unsigned int reused = xx * 28 + yy;
    image[reused] = images[imageIndex * 28 * 28 + reused];
    reused = (xx + 1) * 28 + yy;
    image[reused] = images[imageIndex * 28 * 28 + reused];
    reused = xx * 28 + yy + 1;
    image[reused] = images[imageIndex * 28 * 28 + reused];
    reused = (xx + 1) * 28 + yy + 1;
    image[reused] = images[imageIndex * 28 * 28 + reused];

    // there are 8495 weights to be copied
    // I have 845 threads per block
    // so each thread will copy 11 weights
    reused = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    //printf("tid: %d\n", tid);
    for (xx = 0; xx < 20; xx++) {
        xx = reused + blockDim.x * blockDim.y * blockDim.z * xx;
        if (xx < NUM_WEIGHTS) {
            network[xx] = networks[networkIndex * NUM_WEIGHTS + xx];
        }
    }

    __syncthreads();

    /*
    //TODO: DEBUG MODE - COPIED IMAGE
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == debugFilterIndex
        && blockIdx.x == debugImageIndex && blockIdx.y == 0 && blockIdx.z == 0
            ) {
        for (int image_i = 0; image_i < 28; image_i++) {
            for (int image_j = 0; image_j < 28; image_j++) {
                int index = image_i * 28 + image_j;
                if (image[index] > 0.5f) {
                    printf("X");
                } else if (image[index] > 0.25f) {
                    printf("x");
                } else if (image[index] > -0.25f) {
                    printf(",");
                } else if (image[index] > -0.5f) {
                    printf(".");
                } else {
                    printf(" ");
                }
            }
            printf("\n");
        }
    }

    //TODO: DEBUG MODE
    // Verify all weights are copied
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
        && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0
            ) {
        for (int iii = 0; iii < NUM_WEIGHTS; iii++) {
            if (network[iii] == 0) {
                printf("\nError at index %d\n", iii);
            }
        }
    }
*/
    // To avoid saving partial values in memory, I merge the convolution, pooling and output steps.
    // The thread i, j will take care of calculating the convolution of the 4 pixels:
    // (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    // and then it will take only the maximum value
    // I skip the two threads used to copy the image
    if (threadIdx.x < 13 && threadIdx.y < 13) {

        float pooled = 0;
        float sum = 0;

        // x = i, y = j
        xx = (i + 1) * 2;
        yy = (j + 1) * 2;

        reused = filter * 9;

        unsigned int i_1 = (xx - 1) * 28;
        unsigned int i_2 = xx * 28;
        unsigned int i_3 = (xx + 1) * 28;

        unsigned int j_1 = yy - 1;
        unsigned int j_2 = yy;
        unsigned int j_3 = yy + 1;

/*
        //TODO: DEBUG MODE - FILTER
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == debugFilterIndex
            && blockIdx.x == debugImageIndex && blockIdx.y == 0 && blockIdx.z == 0
                ) {
            for (int filterIndex = 0; filterIndex < 5; filterIndex++) {
                printf("\nFilter %d:\n", filterIndex);
                for (int image_i = 0; image_i < 3; image_i++) {
                    for (int image_j = 0; image_j < 3; image_j++) {
                        int index = filterIndex * 9 + image_i * 3 + image_j;
                        if (network[index] > 0.05f) {
                            printf("X");
                        } else if (network[index] > 0.025f) {
                            printf("x");
                        } else if (network[index] > -0.025f) {
                            printf(",");
                        } else {
                            printf(".");
                        }
                    }
                    printf("\n");
                }
            }
            printf("\n");
        }
*/
        sum = 0;
        sum += image[i_1 + j_1] * network[reused];
        sum += image[i_1 + j_2] * network[reused + 1];
        sum += image[i_1 + j_3] * network[reused + 2];

        sum += image[i_2 + j_1] * network[reused + 3];
        sum += image[i_2 + j_2] * network[reused + 4];
        sum += image[i_2 + j_3] * network[reused + 5];

        sum += image[i_3 + j_1] * network[reused + 6];
        sum += image[i_3 + j_2] * network[reused + 7];
        sum += image[i_3 + j_3] * network[reused + 8];

        if (sum > pooled) {
            pooled = sum;
        }

        // x = i + 1, y = j

        xx = (i + 1) * 2 + 1;
        yy = (j + 1) * 2;

        i_1 = (xx - 1) * 28;
        i_2 = xx * 28;
        i_3 = (xx + 1) * 28;

        j_1 = yy - 1;
        j_2 = yy;
        j_3 = yy + 1;

        sum = 0;
        sum += image[i_1 + j_1] * network[reused];
        sum += image[i_1 + j_2] * network[reused + 1];
        sum += image[i_1 + j_3] * network[reused + 2];

        sum += image[i_2 + j_1] * network[reused + 3];
        sum += image[i_2 + j_2] * network[reused + 4];
        sum += image[i_2 + j_3] * network[reused + 5];

        sum += image[i_3 + j_1] * network[reused + 6];
        sum += image[i_3 + j_2] * network[reused + 7];
        sum += image[i_3 + j_3] * network[reused + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        // x = i, y = j + 1

        xx = (i + 1) * 2;
        yy = (j + 1) * 2 + 1;

        i_1 = (xx - 1) * 28;
        i_2 = xx * 28;
        i_3 = (xx + 1) * 28;

        j_1 = yy - 1;
        j_2 = yy;
        j_3 = yy + 1;
        sum = 0;
        sum += image[i_1 + j_1] * network[reused];
        sum += image[i_1 + j_2] * network[reused + 1];
        sum += image[i_1 + j_3] * network[reused + 2];

        sum += image[i_2 + j_1] * network[reused + 3];
        sum += image[i_2 + j_2] * network[reused + 4];
        sum += image[i_2 + j_3] * network[reused + 5];

        sum += image[i_3 + j_1] * network[reused + 6];
        sum += image[i_3 + j_2] * network[reused + 7];
        sum += image[i_3 + j_3] * network[reused + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        // x = i + 1, y = j + 1

        xx = (i + 1) * 2 + 1;
        yy = (j + 1) * 2 + 1;

        reused = filter * 9;
        i_1 = (xx - 1) * 28;
        i_2 = xx * 28;
        i_3 = (xx + 1) * 28;

        j_1 = yy - 1;
        j_2 = yy;
        j_3 = yy + 1;
        sum = 0;
        sum += image[i_1 + j_1] * network[reused];
        sum += image[i_1 + j_2] * network[reused + 1];
        sum += image[i_1 + j_3] * network[reused + 2];

        sum += image[i_2 + j_1] * network[reused + 3];
        sum += image[i_2 + j_2] * network[reused + 4];
        sum += image[i_2 + j_3] * network[reused + 5];

        sum += image[i_3 + j_1] * network[reused + 6];
        sum += image[i_3 + j_2] * network[reused + 7];
        sum += image[i_3 + j_3] * network[reused + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        maxPooled[filter * 13 * 13 + i * 13 + j] = pooled;
    }

    __syncthreads();

    /*
    //TODO: DEBUG max pooled image
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == debugFilterIndex
        && blockIdx.x == debugImageIndex && blockIdx.y == 0 && blockIdx.z == 0
            ) {
        for (int filterIndex = 0; filterIndex < 5; filterIndex++) {
            printf("Max pooled image #%d:\n", filterIndex);
            float *img = (float *) &maxPooled[filterIndex * 13 * 13];
            for (int max_i = 0; max_i < 13; max_i++) {
                for (int max_j = 0; max_j < 13; max_j++) {
                    int index = max_i * 13 + max_j;
                    if (img[index] > 0.5f) {
                        printf("X");
                    } else if (img[index] > 0.25f) {
                        printf("x");
                    } else if (img[index] > -0.25f) {
                        printf(",");
                    } else if (img[index] > -0.5f) {
                        printf(".");
                    } else {
                        printf(" ");
                    }
                }
                printf("\n");
            }
        }
    }
*/
    // Only one thread per block is responsible to calculate the dense layer
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

        __shared__ float sums[10];
        __shared__ float max;
        __shared__ int index;

        max = -999;
        index = 0;

#pragma unroll
        for (int poolIndex = 0; poolIndex < 13 * 13 * 5; poolIndex++) {
            float input = maxPooled[poolIndex];
            sums[0] += input * network[NET0 + poolIndex];
            sums[1] += input * network[NET1 + poolIndex];
            sums[2] += input * network[NET2 + poolIndex];
            sums[3] += input * network[NET3 + poolIndex];
            sums[4] += input * network[NET4 + poolIndex];
            sums[5] += input * network[NET5 + poolIndex];
            sums[6] += input * network[NET6 + poolIndex];
            sums[7] += input * network[NET7 + poolIndex];
            sums[8] += input * network[NET8 + poolIndex];
            sums[9] += input * network[NET9 + poolIndex];
        }

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

        if (index == labels[imageIndex]) {
            fitness[networkIndex] += 1.0f / 60000.0f;
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

// grid = data, network and filter indexes
    dim3 grid(dataCount, networkCount);

// 1 block = 1 network with 1 input image, 26x26 threads
// To be able to sync the numFilters and avoid saving the conv in shared memory,
// I could launch 13x13x5 threads, so < 1024
// But I use 14x14 to parallelize the copy of the 28x28 image in shared memory
    dim3 block(14, 14, NUM_FILTERS);

    calculateConvolutionGPU<<<grid, block>>>(
            d_images,
            d_networks,
            d_labels,
            d_fitness
    );

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(fitness, d_fitness, networkCount * sizeof(float), D2H));

    cudaFree(d_networks);
    cudaFree(d_fitness);
}