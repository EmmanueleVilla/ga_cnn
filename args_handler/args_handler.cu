//
// Created by emman on 08/02/2023.
//

#include "args_handler.cuh"
#include "args_handler_help.cuh"
#include "args_handler_version.cuh"
#include "args_extractor.cuh"
#include "../data/data_loader.cuh"
#include "../network/init_networks.cuh"
#include "../network/fitness_calculator.cuh"
#include "../genetic/evolve_population.cuh"
#include <stdio.h>
#include <ctime>

bool handle(int argc, char **argv) {
    if (argc == 2 && (strcmp("--help", argv[1]) == 0 || strcmp("-h", argv[1]) == 0)) {
        printHelp();
        return true;
    }

    if (argc == 2 && (strcmp("--version", argv[1]) == 0 || strcmp("-v", argv[1]) == 0)) {
        printVersion();
        return true;
    }

    enum MODE argMode = NONE;
    int size = -1;
    int populationSize = -1;
    extractArgs(argc, argv, argMode, size, populationSize);

    if (argMode == NONE) {
        printf("Mode not set. Use --mode or -m to set the mode.\n");
        return false;
    }

    if (size < 1 || size > 60000) {
        printf("Data size not set or invalid. Must be between 1 and 60000. Use --dataSize or -s to set the size.\n");
        return false;
    }

    if (populationSize < 1) {
        printf("Population size not set or invalid. Must be a positive number. Use --popSize or -p to set the size.\n");
        return false;
    }

    int *labels;
    float *images;
    float *networks;
    networks = (float *) malloc(sizeof(float) * populationSize * NUM_WEIGHTS);

    labels = (int *) malloc(size * sizeof(int));
    images = (float *) malloc(size * 28 * 28 * sizeof(float));

    // Load data also initialize the network weights
    // Because the GPU load parallelize it with the CPU
    loadData(size, labels, images, argMode, networks, populationSize);

    clock_t start, end;

    start = clock();

    auto *fitness = (float *) malloc(sizeof(float) * populationSize);

    float maxFitness = 0;

    int generation = 0;
    printf("Start fitness: \t%6.3ld\n", start);
    while (maxFitness < 95) {
        start = clock();
        calculateFitness(labels, images, networks, populationSize, size, fitness, argMode);
        evolve(networks, fitness, populationSize, argMode);
        for (int i = 0; i < populationSize; i++) {
            if (fitness[i] > maxFitness) {
                maxFitness = fitness[i];
            }
        }
        printf("%d) Max fitness: %f, generation time: %6.3ld\n", generation++, maxFitness, clock() - start);
    }

    return true;
}