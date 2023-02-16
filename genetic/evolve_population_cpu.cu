//
// Created by emman on 15/02/2023.
//

#include "evolve_population_cpu.cuh"
#include "../network/init_networks.cuh"
#include "../network/init_networks_utils.cuh"
#include "../defines.cuh"
#include <stdio.h>


int tournamentSelection(const float *fitness, int popSize) {
    int parent = rand() % popSize;
    for (int j = 0; j < TOURNAMENT_SIZE; j++) {
        int candidate = rand() % popSize;
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


float *crossover(const float *parent1, const float *parent2, bool *mask) {
    auto *child = new float[NUM_WEIGHTS];

    int first = rand() % NUM_WEIGHTS;
    int second = rand() % NUM_WEIGHTS;
    int third = rand() % NUM_WEIGHTS;

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

    // random number between 0 and 50
    int random = rand() % 50;
    for (int i = 0; i < random; i++) {
        int index = rand() % NUM_WEIGHTS;
        child[index] = randGaussian();
    }
    return child;
}

bool *createMask() {
    bool *mask = new bool[NUM_WEIGHTS];
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        mask[i] = rand() % 2 == 0;
    }
    return mask;
}

void evolveCPU(float *networks, float *fitness, int popSize) {
    //printf("Evolve CPU\n");

    bool *mask = createMask();
    float *newPop = new float[popSize * NUM_WEIGHTS];
    for (int i = 0; i < popSize; i++) {
        int first = tournamentSelection(fitness, popSize);
        //printf("first: %d", first);
        int second = tournamentSelection(fitness, popSize);
        //printf("second: %d", second);
        float *child = crossover(&networks[first * NUM_WEIGHTS], &networks[second * NUM_WEIGHTS], mask);
        //printf("child: %f", child[0]);
        memcpy(&newPop[i * NUM_WEIGHTS], child, sizeof(float) * NUM_WEIGHTS);
    }
    memcpy(networks, newPop, sizeof(float) * popSize * NUM_WEIGHTS);
    delete[] newPop;
}