//
// Created by emman on 15/02/2023.
//

#include "evolve_population_cpu.cuh"
#include "../network/init_networks.cuh"
#include "../network/init_networks_utils.cuh"
#include "../defines.cuh"
#include <stdio.h>
#include <random>

std::vector<int> parentIndex = std::vector<int>();
std::vector<int> crossoverIndex = std::vector<int>();
std::vector<int> mutationCount = std::vector<int>();
std::vector<int> mutationIndex = std::vector<int>();
std::vector<float> newWeights = std::vector<float>();

int tournamentSelection(const float *fitness, int popSize) {
    int parent = rand() % popSize;
    for (int j = 0; j < TOURNAMENT_SIZE; j++) {
        int candidate = rand() % popSize;
        if (fitness[candidate] > fitness[parent]) {
            parent = candidate;
        }
    }
    parentIndex.push_back(parent);
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

    int first = rand() % NUM_WEIGHTS;
    int second = rand() % NUM_WEIGHTS;
    int third = rand() % NUM_WEIGHTS;

    crossoverIndex.push_back(first);
    crossoverIndex.push_back(second);
    crossoverIndex.push_back(third);

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
    int random = rand() % 250;
    mutationCount.push_back(random);
    for (int i = 0; i < random; i++) {
        int index = rand() % NUM_WEIGHTS;
        mutationIndex.push_back(index);
        child[index] = randGaussian();
        newWeights.push_back(child[index]);
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

    FILE *fptr;
    fptr = fopen("parentIndex.txt", "w");
    for (int i = 0; i < parentIndex.size(); i++) {
        fprintf(fptr, "%d\n", parentIndex[i]);
    }
    fclose(fptr);

    fptr = fopen("crossoverIndex.txt", "w");
    for (int i = 0; i < crossoverIndex.size(); i++) {
        fprintf(fptr, "%d\n", crossoverIndex[i]);
    }
    fclose(fptr);

    fptr = fopen("mutationCount.txt", "w");
    for (int i = 0; i < mutationCount.size(); i++) {
        fprintf(fptr, "%d\n", mutationCount[i]);
    }
    fclose(fptr);

    fptr = fopen("mutationIndex.txt", "w");
    for (int i = 0; i < mutationIndex.size(); i++) {
        fprintf(fptr, "%f\n", mutationIndex[i]);
    }
    fclose(fptr);

    fptr = fopen("newWeights.txt", "w");
    for (int i = 0; i < newWeights.size(); i++) {
        fprintf(fptr, "%.2f\n", newWeights[i]);
    }
    fclose(fptr);


    delete[] newPop;
}