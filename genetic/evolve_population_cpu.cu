//
// Created by emman on 15/02/2023.
//

#include "evolve_population_cpu.cuh"
#include "../network/init_networks.cuh"
#include "../network/init_networks_utils.cuh"
#include "../defines.cuh"
#include <stdio.h>
#include <random>

// A function to return a seeded random number generator.
inline std::mt19937 &generator() {
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// A function to generate integers in the range [min, max]
template<typename T, std::enable_if_t<std::is_integral_v<T>> * = nullptr>
T my_rand(T min, T max) {
    std::uniform_int_distribution<T> dist(min, max);
    return dist(generator());
}

int tournamentSelection(const float *fitness, int popSize) {
    int parent = my_rand(0, popSize);
    for (int j = 0; j < TOURNAMENT_SIZE; j++) {
        int candidate = my_rand(0, popSize);
        if (fitness[candidate] > fitness[parent]) {
            parent = candidate;
        }
    }
    return parent;
}

int my_min(int a, int b) {
    return a < b ? a : b;
}

int my_max(int a, int b) {
    return a > b ? a : b;
}


float *crossover(const float *parent1, const float *parent2) {
    auto *child = new float[NUM_WEIGHTS];

    int first = my_rand(0, NUM_WEIGHTS);
    int second = my_rand(0, NUM_WEIGHTS);
    int third = my_rand(0, NUM_WEIGHTS);

    int start = my_min(first, my_min(second, third));
    int end = my_max(first, my_max(second, third));
    for (int i = 0; i < start; i++) {
        child[i] = parent1[i];
    }
    for (int i = start; i < end; i++) {
        child[i] = parent2[i];
    }
    for (int i = end; i < NUM_WEIGHTS; i++) {
        child[i] = parent1[i];
    }

    // random number between 0 and 100
    int random = my_rand(0, 250);
    for (int i = 0; i < random; i++) {
        int index = my_rand(0, NUM_WEIGHTS);
        child[index] = randGaussian();
    }
    return child;
}

void evolveCPU(float *networks, float *fitness, int popSize) {
    //printf("Evolve CPU\n");

    float *newPop = new float[popSize * NUM_WEIGHTS];
    for (int i = 0; i < popSize; i++) {
        int first = tournamentSelection(fitness, popSize);
        //printf("first: %d", first);
        int second = tournamentSelection(fitness, popSize);
        //printf("second: %d", second);
        float *child = crossover(&networks[first * NUM_WEIGHTS], &networks[second * NUM_WEIGHTS]);
        //printf("child: %f", child[0]);
        memcpy(&newPop[i * NUM_WEIGHTS], child, sizeof(float) * NUM_WEIGHTS);
    }
    memcpy(networks, newPop, sizeof(float) * popSize * NUM_WEIGHTS);
    delete[] newPop;
}