//
// Created by emman on 15/02/2023.
//

#include "evolve_population_gpu.cuh"
#include "../defines.cuh"
#include "../network/init_networks_utils.cuh"
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

__device__ int my_min_gpu(int a, int b) {
    return a < b ? a : b;
}

__device__ int my_max_gpu(int a, int b) {
    return a > b ? a : b;
}

__device__ float randGaussian_gpu(curandState *state) {
    float a = curand_uniform(state) * (float) RAND_MAX;
    float b = curand_uniform(state) * (float) RAND_MAX;
    if (a == 0) {
        a = 0.0000001;
    }
    if (b == 0) {
        b = 0.0000001;
    }

    float R0 = sqrt(-2.0 * log(a)) * cos(2 * M_PI * b);

    return R0 / 5;
}

__global__ void
evolveNetwork(float *oldPopulation,
              float *fitness,
              float *randomNumbers,
              float *newPopulation,
              int popSize,
              curandState *state,
              time_t seed
) {

    // index of the new network
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed + tid, tid, 0, &state[tid]);

    auto *numbers = &randomNumbers[tid * RANDOM_NUMBERS_FOR_EVOLUTION];

    int parent1Index = numbers[0] * popSize;
    for (int j = 0; j < TOURNAMENT_SIZE; j++) {
        int candidate = numbers[j + 1] * popSize;
        if (fitness[candidate] > fitness[parent1Index]) {
            parent1Index = candidate;
        }
    }
    int parent2Index = numbers[TOURNAMENT_SIZE + 2] * popSize;
    for (int j = 0; j < TOURNAMENT_SIZE; j++) {
        int candidate = numbers[TOURNAMENT_SIZE + 2 + j] * popSize;
        if (fitness[candidate] > fitness[parent2Index]) {
            parent2Index = candidate;
        }
    }

    auto *parent1 = &oldPopulation[parent1Index * NUM_WEIGHTS];
    auto *parent2 = &oldPopulation[parent2Index * NUM_WEIGHTS];

    auto *child = &newPopulation[tid];

    int first = numbers[TOURNAMENT_SIZE * 2 + 3] * NUM_WEIGHTS;
    int second = numbers[TOURNAMENT_SIZE * 2 + 4] * NUM_WEIGHTS;
    int third = numbers[TOURNAMENT_SIZE * 2 + 5] * NUM_WEIGHTS;

    int start = my_min_gpu(first, my_min_gpu(second, third));
    int end = my_max_gpu(first, my_max_gpu(second, third));
    for (int i = 0; i < start; i++) {
        child[i] = parent1[i];
    }
    for (int i = start; i < end; i++) {
        child[i] = parent2[i];
    }
    for (int i = end; i < NUM_WEIGHTS; i++) {
        child[i] = parent1[i];
    }


    for (int i = 0; i < NUM_WEIGHTS; i++) {
        if (numbers[TOURNAMENT_SIZE * 2 + 5 + i] < 0.001) {
            printf("Mutation\n");
            child[i] = randGaussian_gpu(&state[tid]);
        }
    }

    printf("Child %d: %f\n", tid, child[0]);
}

void evolveGPU(float *networks, float *fitness, int popSize) {

    // Create the new population
    float *newPop = new float[popSize * NUM_WEIGHTS];

    // Each new individual will be created in a separate thread
    // Each one needs
    // 1) TOURNAMENT_SIZE * 2 random numbers for the tournament selection
    // 2) 3 random numbers for the crossover
    // 3) NUM_WEIGHTS random numbers for the mutation
    float *devData;
    int n = popSize * (RANDOM_NUMBERS_FOR_EVOLUTION);
    cudaMalloc((void **) &devData, n * sizeof(float));

    /*
    curandGenerator_t gen;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(gen, time(nullptr));

    curandGenerateUniform(gen, devData, n);
*/
    // I also create a curandState for each thread to generate values on the fly
    curandState *states;

    cudaMalloc((void **) &states, popSize * sizeof(curandState));

    evolveNetwork<<<popSize, 1>>>(
            networks,
            fitness,
            devData,
            newPop,
            popSize,
            states,
            time(0)
    );

    CHECK(cudaDeviceSynchronize());

    //curandDestroyGenerator(gen);
    cudaFree(devData);
}