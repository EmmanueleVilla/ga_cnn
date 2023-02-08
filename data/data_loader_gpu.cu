//
// Created by emman on 08/02/2023.
//

#include "data_loader_gpu.cuh"
#include <stdio.h>
#include <ctime>


#define CHECK(call) {\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

/**
 * This kernel counts the number of delimeters in the given buffer.
 * The buffer arrives as a string of characters, containing the data from the mnist_train.csv file.
 * @param buf
 * @param result
 */
__global__ void countDelimiters(const char *buf, int *result) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid * 2500;
    unsigned int end = start + 2500;
    int count = 0;
    for (unsigned int i = start; i < end; i++) {
        if (buf[i] == ',' || buf[i] == '\n') {
            count++;
        }
    }
    result[tid] = count;
}

__global__ void fillFieldsIndexes(const char *buf, const int *prefixSum, int *result) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid * 2500;
    unsigned int end = start + 2500;
    int count = 0;
    for (int i = (int) start; i < end; i++) {
        if (buf[i] == ',' || buf[i] == '\n') {
            count++;
            result[prefixSum[tid] + count] = i;
        }
    }
}

/**
 * This loads the data from the mnist_train.csv file using the GPU.
 * It parses the file and loads the labels and images into the given arrays.
 * Info: https://www.clemenslutz.com/pdfs/msc_thesis_alexander_kumaigorodski.pdf
 * @param size: the number of entries to be read
 * @param labels: the array to store the labels
 * @param images: the array to store the images
 */
void loadDataWithGPU(int size, int *labels, float *images, FILE *stream) {

    clock_t start, stop;

    start = clock();

    // Calculate the file size and ceil it to the nearest multiple of 64000
    // Because I want 64 threads in a block, each parsing 10k characters
    fseek(stream, 0, SEEK_END);
    int file_size;
    file_size = (int) ceil(ftell(stream) / 320000.0) * 320000;
    rewind(stream);
    int blockSize = file_size / 320000;
    int numThreads = 128;

    // Read the file into a buffer
    char *buffer = (char *) malloc(sizeof(char) * file_size);
    fread(buffer, 1, file_size, stream);
    fclose(stream);

    // Copy buffer to the device
    char *d_buffer;
    cudaMalloc((void **) &d_buffer, file_size * sizeof(char));
    cudaMemcpy(d_buffer, buffer, file_size * sizeof(char), cudaMemcpyHostToDevice);

    // Create result buffers
    int *delimiters;
    int *d_delimiters;
    delimiters = (int *) malloc(blockSize * numThreads * sizeof(int));
    cudaMalloc((void **) &d_delimiters, blockSize * numThreads * sizeof(int));

    // First step: count the delimiter characters
    countDelimiters <<<blockSize, numThreads>>>(d_buffer, d_delimiters);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(delimiters, d_delimiters, blockSize * numThreads * sizeof(int), cudaMemcpyDeviceToHost);

    // Second step: calculate the prefix sum of the delimiters (on the CPU because I think it won't change much)
    int *prefix_sum;
    prefix_sum = (int *) malloc(blockSize * numThreads * sizeof(int));
    prefix_sum[0] = 0;
    for (int i = 1; i < blockSize * numThreads; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + delimiters[i - 1];
    }
    printf("total delimiters: %d\n", prefix_sum[blockSize * numThreads - 1]);

    // Third step: fill the fieldsIndex array
    int *fieldsIndex;
    fieldsIndex = (int *) malloc((prefix_sum[blockSize * numThreads - 1] + 1) * sizeof(int));

    //Prepare the device memory
    int *d_prefix_sum;
    int *d_fieldsIndex;
    cudaMalloc((void **) &d_prefix_sum, blockSize * numThreads * sizeof(int));
    cudaMalloc((void **) &d_fieldsIndex, (prefix_sum[blockSize * numThreads - 1] + 1) * sizeof(int));
    cudaMemcpy(d_prefix_sum, prefix_sum, blockSize * numThreads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fieldsIndex, fieldsIndex, (prefix_sum[blockSize * numThreads - 1] + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    // Run the kernel
    fillFieldsIndexes <<<blockSize, numThreads>>>(d_buffer, d_prefix_sum, d_fieldsIndex);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(fieldsIndex, d_fieldsIndex, (prefix_sum[blockSize * numThreads - 1] + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    stop = clock();
    printf("start: %6.3ld\n", start);
    printf("stop: %6.3ld\n", stop);
}