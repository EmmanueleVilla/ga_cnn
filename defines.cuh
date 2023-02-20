//
// Created by emman on 16/02/2023.
//

#ifndef GA_CNN_DEFINES_CUH
#define GA_CNN_DEFINES_CUH

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

#define CHECK(call) {\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

#define CONV_SIZE (26*26*NUM_FILTERS)
#define POOLED_SIZE (13*13*NUM_FILTERS)
#define TOURNAMENT_SIZE 10
#define IMAGE_INPUT_SIZE (28*28)
#define CONV_IMAGE_SIZE (26*26)
#define OUTPUT_PER_FILTER (10 * NUM_FILTERS)

#define FILTER_SIZE 3
#define NUM_FILTERS 5
#define POOLED_IMAGE_SIZE 13
#define NUM_WEIGHTS (NUM_FILTERS * FILTER_SIZE * FILTER_SIZE + POOLED_IMAGE_SIZE * POOLED_IMAGE_SIZE * NUM_FILTERS * 10)
#define EULER_NUMBER 2.7182f

#endif //GA_CNN_DEFINES_CUH
