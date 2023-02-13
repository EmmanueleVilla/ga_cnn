//
// Created by emmanuele on 08/02/2023.
//

#include "data_loader_gpu.cuh"
#include "../info/device_info.cuh"
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

int atoi(char string[]);

/**
 * This kernel counts the number of delimiters in the given buffer.
 * The buffer arrives as a string of characters, containing the data from the mnist_train.csv file.
 * @param buf the buffer containing the data from the mnist_train.csv file.
 * @param result the array containing the number of delimiters for each block.
 */
__global__ void countDelimiters(const char *buf, int *result) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid * 313;
    unsigned int end = start + 313;
    int count = 0;
    for (unsigned int i = start; i < end; i++) {
        if (buf[i] == ',' || buf[i] == '\n') {
            count++;
        }
    }
    result[tid] = count;
}

/**
 * This kernel fills the fieldsIndex array using the information from the prefixSum array.
 * @param buf the buffer containing the data from the mnist_train.csv file.
 * @param prefixSum the array containing the prefix sum of the delimiters.
 * @param result the array to store the indexes of the delimiters.
 */
__global__ void fillFieldsIndexes(const char *buf, const int *prefixSum, int *result) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid * 313;
    unsigned int end = start + 313;
    int count = 0;
    for (unsigned int i = start; i < end; i++) {
        if (buf[i] == ',' || buf[i] == '\n') {
            count++;
            result[prefixSum[tid] + count] = i;
        }
    }
}

/**
 * Extracts the data from the buffer and stores it in the labels and images arrays.
 * @param buf the buffer containing the data from the mnist_train.csv file.
 * @param fieldsIndex the array containing the indexes of the delimiters
 * @param labels the array to store the labels
 * @param images the array to store the images
 */
__global__ void extractData(const char *buf, const int *fieldsIndex, int *labels, float *images, int size) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    int res = 0;
    int start = max(0, fieldsIndex[tid] + 1);
    int end = fieldsIndex[tid + 1];
    for (int i = start; i < end; i++) {
        res = res * 10 + (buf[i] - '0');
    }
    if (threadIdx.x == 0) {
        labels[blockIdx.x] = res;
    } else {
        images[threadIdx.x + 784 * blockIdx.x] = ((float) res) / 255.0f;
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

    displayGPUHeader();

    clock_t start, stop;

    start = clock();

    fseek(stream, 0, SEEK_END);
    int file_size;
    file_size = (int) ceil(ftell(stream) / 320512.0) * 320512;
    rewind(stream);
    int blockSize = file_size / 320512;
    int numThreads = 1024;
    int totalThreads = blockSize * numThreads;

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
    delimiters = (int *) malloc(totalThreads * sizeof(int));
    cudaMalloc((void **) &d_delimiters, totalThreads * sizeof(int));

    // First step: count the delimiter characters
    countDelimiters <<<blockSize, numThreads>>>(d_buffer, d_delimiters);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(delimiters, d_delimiters, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

    // Second step: calculate the prefix sum of the delimiters (on the CPU because I think it won't change much)
    int *prefix_sum;
    prefix_sum = (int *) malloc(totalThreads * sizeof(int));
    prefix_sum[0] = 0;
    for (int i = 1; i < totalThreads; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + delimiters[i - 1];
    }
    printf("total delimiters: %d\n", prefix_sum[totalThreads - 1]);

    // Third step: fill the fieldsIndex array
    int numFields = prefix_sum[totalThreads - 1] + 1;

    //Prepare the device memory
    int *d_prefix_sum;
    int *d_fieldsIndex;
    cudaMalloc((void **) &d_prefix_sum, totalThreads * sizeof(int));
    cudaMalloc((void **) &d_fieldsIndex, numFields * sizeof(int));
    cudaMemcpy(d_prefix_sum, prefix_sum, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

    // Run the kernel
    fillFieldsIndexes <<<blockSize, numThreads>>>(d_buffer, d_prefix_sum, d_fieldsIndex);
    CHECK(cudaDeviceSynchronize());

    // Fourth step: extract the data
    // In the csv we have 60k entries, each entry has 785 fields (784 pixels + 1 label)
    // So to do everything in a single step, we'll have 60k blocks, each with 785 threads
    int numBlocks = 60000;
    numThreads = 785;

    int *d_labels;
    float *d_images;
    cudaMalloc((void **) &d_labels, size * sizeof(int));
    cudaMalloc((void **) &d_images, size * 784 * sizeof(float));

    // Run the kernel
    extractData <<<numBlocks, numThreads>>>(d_buffer, d_fieldsIndex, d_labels, d_images, numFields);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(labels, d_labels, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(images, d_images, size * 784 * sizeof(float), cudaMemcpyDeviceToHost);

    stop = clock();
    printf("\nstart: %6.3ld\n", start);
    printf("stop: %6.3ld\n", stop);
}
