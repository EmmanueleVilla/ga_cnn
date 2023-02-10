//
// Created by emmanuele on 08/02/2023.
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

int atoi(char string[]);

/**
 * This kernel counts the number of delimiters in the given buffer.
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
    //TODO Loop unrolling?
    for (unsigned int i = start; i < end; i++) {
        if (buf[i] == ',' || buf[i] == '\n') {
            count++;
        }
    }
    result[tid] = count;
}

/**
 * This kernel fills the fieldsIndex array using the information from the prefixSum array.
 * @param buf
 * @param prefixSum
 * @param result
 */
__global__ void fillFieldsIndexes(const char *buf, const int *prefixSum, int *result) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid * 2500;
    unsigned int end = start + 2500;
    int count = 0;
    //TODO Loop unrolling?
    for (int i = (int) start; i < end; i++) {
        if (buf[i] == ',' || buf[i] == '\n') {
            count++;
            result[prefixSum[tid] + count] = i;
        }
    }
}

__global__ void extractData(const char *buf, const int *fieldsIndex, int *fields, int fieldsSize) {
    unsigned int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("tid: %d\n", tid);
    if (tid > fieldsSize - 1) {
        return;
    }
    int res = 0;
    int start = max(0, fieldsIndex[tid] + 1);
    int end = fieldsIndex[tid + 1];
    //printf("start: %d, end: %d\n", start, end);
    for (int i = start; i < end; i++) {
        //printf("%c\n", buf[i]);
        res = res * 10 + (buf[i] - '0');
    }
    fields[tid] = res;
    //printf("fields[%d]: %d\n", tid, fields[tid]);
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
    int *fieldsIndex;
    int numFields = prefix_sum[totalThreads - 1] + 1;

    fieldsIndex = (int *) malloc(numFields * sizeof(int));

    //Prepare the device memory
    int *d_prefix_sum;
    int *d_fieldsIndex;
    cudaMalloc((void **) &d_prefix_sum, totalThreads * sizeof(int));
    cudaMalloc((void **) &d_fieldsIndex, numFields * sizeof(int));
    cudaMemcpy(d_prefix_sum, prefix_sum, totalThreads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fieldsIndex, fieldsIndex, numFields * sizeof(int), cudaMemcpyHostToDevice);

    // Run the kernel
    fillFieldsIndexes <<<blockSize, numThreads>>>(d_buffer, d_prefix_sum, d_fieldsIndex);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(fieldsIndex, d_fieldsIndex, (prefix_sum[totalThreads - 1] + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Now we are ready to split the fields
    // The number of fields is the number of delimiters plus one,
    // so 60k entries * (28 * 28 + 1) = around 47 million fields
    // I split the fields into N blocks of 1024 threads each,
    // so I need 47 million / 1024 = 46k blocks
    int numBlocks = ceil(numFields / 1024);
    numThreads = 1024;

    int *fields;
    int *d_fields;
    fields = (int *) malloc(numFields * sizeof(int));
    cudaMalloc((void **) &d_fields, numFields * sizeof(int));

    // Run the kernel
    extractData <<<numBlocks, numThreads>>>(d_buffer, d_fieldsIndex, d_fields, numFields);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(fields, d_fields, numFields * sizeof(int), cudaMemcpyDeviceToHost);

    stop = clock();
    printf("\nstart: %6.3ld\n", start);
    printf("stop: %6.3ld\n", stop);
}
