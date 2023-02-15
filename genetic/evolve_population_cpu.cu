//
// Created by emman on 15/02/2023.
//

#include "evolve_population_cpu.cuh"
#include "../network/init_networks.cuh"
#include "../network/init_networks_utils.cuh"
#include <stdio.h>

#define TOURNAMENT_SIZE 10

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

float *crossover(const float *parent1, const float *parent2, bool *mask) {
    float *child = new float[NUM_WEIGHTS];
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        if (mask[i]) {
            child[i] = parent1[i];
        } else {
            child[i] = parent2[i];
        }
    }
    // try to mutate 50 times
    for (int i = 0; i < 50; i++) {
        if (rand() % 100 < 1) {
            int index = rand() % NUM_WEIGHTS;
            child[index] = randGaussian();
        }
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