//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_gpu.cuh"
#include "../defines.cuh"
#include "../output/output_utils.cuh"

#include <stdio.h>


__global__ void calculateConvolutionGPU(
        const float *images,
        const float *networks,
        const int *labels,
        float *fitness
) {
    int imageIndex = blockIdx.x;
    int networkIndex = blockIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int filter = threadIdx.z;

    __shared__ float image[IMAGE_INPUT_SIZE];
    __shared__ float network[NUM_WEIGHTS];
    __shared__ float maxPooled[POOLED_SIZE];
    //__shared__ float output[OUTPUT_PER_FILTER];

    //TODO: fix that the last two rows are missing because I have 13 threads * 2 = 26 and not 28
    if (threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        int xx = threadIdx.x * 2;
        int yy = threadIdx.y * 2;
        int pixel = xx * 28 + yy;
        image[pixel] = images[imageIndex * 28 * 28 + pixel];
        int pixel_1 = (xx + 1) * 28 + yy;
        image[pixel_1] = images[imageIndex * 28 * 28 + pixel_1];
        int pixel_2 = xx * 28 + yy + 1;
        image[pixel_2] = images[imageIndex * 28 * 28 + pixel_2];
        int pixel_3 = (xx + 1) * 28 + yy + 1;
        image[pixel_3] = images[imageIndex * 28 * 28 + pixel_3];
    }

    __syncthreads();

    // there are 8495 weights to be copied
    // I have 845 threads per block
    // so each thread will copy 11 weights
    // TODO: fix this fng tid calculation
    int tid = (threadIdx.x * blockDim.x + threadIdx.y) * threadIdx.z;
    //printf("tid: %d\n", tid);
    for (int w_i = 0; w_i < 11; w_i++) {
        if (tid + tid * w_i < NUM_WEIGHTS) {
            network[tid + tid * w_i] = networks[networkIndex * NUM_WEIGHTS + tid + tid * w_i];
        }
    }

    __syncthreads();


    //TODO: remove this, debug only
    // Verify all weights are copied
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
        && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0
            ) {
        for (int iii = 0; iii < NUM_WEIGHTS; iii++) {
            if (network[iii] == 0) {
                printf("Error at index %d\n", iii);
            }
        }
    }

    return;
    // To avoid saving partial values in memory, I merge the convolution, pooling and output steps.
    // The thread i, j will take care of calculating the convolution of the 4 pixels:
    // (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    // and then it will take only the maximum value
    // I skip the two threads used to copy the image
    if (i < 14 && j < 14) {
        float pooled = 0;
        float sum = 0;

        // x = i, y = j
        int x = i + 1;
        int y = j + 1;

        int start = filter * 9;
        int i_1 = (x - 1) * 28;
        int i_2 = x * 28;
        int i_3 = (x + 1) * 28;

        int j_1 = y - 1;
        int j_2 = y;
        int j_3 = y + 1;
        sum = 0;
        sum += image[i_1 + j_1] * network[start];
        sum += image[i_1 + j_2] * network[start + 1];
        sum += image[i_1 + j_3] * network[start + 2];

        sum += image[i_2 + j_1] * network[start + 3];
        sum += image[i_2 + j_2] * network[start + 4];
        sum += image[i_2 + j_3] * network[start + 5];

        sum += image[i_3 + j_1] * network[start + 6];
        sum += image[i_3 + j_2] * network[start + 7];
        sum += image[i_3 + j_3] * network[start + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        // x = i + 1, y = j

        x = i + 1;
        y = j;

        start = filter * 9;
        i_1 = (x - 1) * 28;
        i_2 = x * 28;
        i_3 = (x + 1) * 28;

        j_1 = y - 1;
        j_2 = y;
        j_3 = y + 1;
        sum = 0;
        sum += image[i_1 + j_1] * network[start];
        sum += image[i_1 + j_2] * network[start + 1];
        sum += image[i_1 + j_3] * network[start + 2];

        sum += image[i_2 + j_1] * network[start + 3];
        sum += image[i_2 + j_2] * network[start + 4];
        sum += image[i_2 + j_3] * network[start + 5];

        sum += image[i_3 + j_1] * network[start + 6];
        sum += image[i_3 + j_2] * network[start + 7];
        sum += image[i_3 + j_3] * network[start + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        // x = i, y = j + 1

        x = i;
        y = j + 1;

        start = filter * 9;
        i_1 = (x - 1) * 28;
        i_2 = x * 28;
        i_3 = (x + 1) * 28;

        j_1 = y - 1;
        j_2 = y;
        j_3 = y + 1;
        sum = 0;
        sum += image[i_1 + j_1] * network[start];
        sum += image[i_1 + j_2] * network[start + 1];
        sum += image[i_1 + j_3] * network[start + 2];

        sum += image[i_2 + j_1] * network[start + 3];
        sum += image[i_2 + j_2] * network[start + 4];
        sum += image[i_2 + j_3] * network[start + 5];

        sum += image[i_3 + j_1] * network[start + 6];
        sum += image[i_3 + j_2] * network[start + 7];
        sum += image[i_3 + j_3] * network[start + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        // x = i + 1, y = j + 1

        x = i + 1;
        y = j + 1;

        start = filter * 9;
        i_1 = (x - 1) * 28;
        i_2 = x * 28;
        i_3 = (x + 1) * 28;

        j_1 = y - 1;
        j_2 = y;
        j_3 = y + 1;
        sum = 0;
        sum += image[i_1 + j_1] * network[start];
        sum += image[i_1 + j_2] * network[start + 1];
        sum += image[i_1 + j_3] * network[start + 2];

        sum += image[i_2 + j_1] * network[start + 3];
        sum += image[i_2 + j_2] * network[start + 4];
        sum += image[i_2 + j_3] * network[start + 5];

        sum += image[i_3 + j_1] * network[start + 6];
        sum += image[i_3 + j_2] * network[start + 7];
        sum += image[i_3 + j_3] * network[start + 8];
        if (sum > pooled) {
            pooled = sum;
        }

        maxPooled[filter * 13 * 13 + i * 13 + j] = pooled;
    }

    __syncthreads();

    // Only one thread per block is responsible to calculate the dense layer
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
        && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 //TODO: remove this row, image debug only
            ) {

        // Print the loaded image: correct
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int index = i * 28 + j;
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

        // Print the max pooled image: wrong
        float *img = (float *) &maxPooled[filter * 13 * 13];
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 13; j++) {
                int index = i * 13 + j;
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
        float max = -999;
        for (int outputIndex = 0; i < 10; i++) {
            float sumValue = 0;
            for (int poolIndex = 0; poolIndex < 13 * 13 * 5; poolIndex++) {
                sumValue += maxPooled[poolIndex] * network[45 + outputIndex * 13 * 13 * 5 + poolIndex];
            }
            if (sumValue > max || max == -999) {
                max = (int) outputIndex;
            }
        }

        //printf("Label is %d\n", labels[imageIndex]);
        if (max == labels[imageIndex]) {
            atomicAdd(&fitness[networkIndex], 1.0f / 60000.0f);
        }
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
    float *d_fitness;

    // TODO check if pinned memory is faster
    cudaMalloc((void **) &d_images, dataCount * 28 * 28 * sizeof(float));
    cudaMalloc((void **) &d_networks, networkCount * NUM_WEIGHTS * sizeof(float));
    cudaMalloc((void **) &d_labels, dataCount * sizeof(int));
    cudaMalloc((void **) &d_fitness, networkCount * sizeof(float));

    cudaMemcpy(d_images, images, dataCount * 28 * 28 * sizeof(float), H2D);
    cudaMemcpy(d_networks, networks, networkCount * NUM_WEIGHTS * sizeof(float), H2D);
    cudaMemcpy(d_labels, labels, dataCount * sizeof(int), H2D);

    // grid = data, network and filter indexes
    dim3 grid(dataCount, networkCount);

    // 1 block = 1 network with 1 input image, 26x26 threads
    // To be able to sync the numFilters and avoid saving the conv in shared memory,
    // I could launch 13x13x5 threads, so < 1024
    // But I use 14x14 to parallelize the copy of the 28x28 image in shared memory
    dim3 block(14, 14, NUM_FILTERS);

    calculateConvolutionGPU<<<grid, block>>>(d_images, d_networks, d_labels, d_fitness);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(fitness, d_fitness, networkCount * sizeof(float), D2H));
}