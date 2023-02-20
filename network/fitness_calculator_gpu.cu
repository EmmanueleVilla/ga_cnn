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
    unsigned int pixel = xx * 28 + yy;
    image[pixel] = images[imageIndex * 28 * 28 + pixel];
    unsigned int pixel_1 = (xx + 1) * 28 + yy;
    image[pixel_1] = images[imageIndex * 28 * 28 + pixel_1];
    unsigned int pixel_2 = xx * 28 + yy + 1;
    image[pixel_2] = images[imageIndex * 28 * 28 + pixel_2];
    unsigned int pixel_3 = (xx + 1) * 28 + yy + 1;
    image[pixel_3] = images[imageIndex * 28 * 28 + pixel_3];

    // there are 8495 weights to be copied
    // I have 845 threads per block
    // so each thread will copy 11 weights
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    //printf("tid: %d\n", tid);
    for (int w_i = 0; w_i < 20; w_i++) {
        unsigned int w_index = tid + blockDim.x * blockDim.y * blockDim.z * w_i;
        if (w_index < NUM_WEIGHTS) {
            network[w_index] = networks[networkIndex * NUM_WEIGHTS + w_index];
        }
    }

    __syncthreads();

    int debugImageIndex = 25765;
    int debugFilterIndex = 2;
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

    // To avoid saving partial values in memory, I merge the convolution, pooling and output steps.
    // The thread i, j will take care of calculating the convolution of the 4 pixels:
    // (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    // and then it will take only the maximum value
    // I skip the two threads used to copy the image
    if (threadIdx.x < 13 && threadIdx.y < 13) {

        float pooled = 0;
        float sum = 0;

        // x = i, y = j
        unsigned int x = (i + 1) * 2;
        unsigned int y = (j + 1) * 2;
        unsigned int start = filter * 9;

        unsigned int i_1 = (x - 1) * 28;
        unsigned int i_2 = x * 28;
        unsigned int i_3 = (x + 1) * 28;

        unsigned int j_1 = y - 1;
        unsigned int j_2 = y;
        unsigned int j_3 = y + 1;


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

        x = (i + 1) * 2 + 1;
        y = (j + 1) * 2;

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

        x = (i + 1) * 2;
        y = (j + 1) * 2 + 1;

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

        x = (i + 1) * 2 + 1;
        y = (j + 1) * 2 + 1;

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

    //TODO: DEBUG max pooled image: wrong
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

    // Only one thread per block is responsible to calculate the dense layer
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

        int max = -999;
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
    //TODO: copy images and labels only the first time
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