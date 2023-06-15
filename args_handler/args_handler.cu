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
#include "../defines.cuh"
#include "../network/fitness_calculator_gpu.cuh"
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
    int gensCount = -1;
    bool debug = false;
    extractArgs(argc, argv, argMode, size, populationSize, gensCount, debug);

    if (argMode == NONE) {
        printf("Mode not set. Use --mode or -m to set the mode.\n");
        return false;
    }

    if (size < 1 || size > 60000) {
        printf("Data size not set or invalid. Must be between 1 and 60000. Use --data-size or -s to set the size.\n");
        return false;
    }

    if (populationSize < 1) {
        printf("Population size not set or invalid. Must be a positive number. Use --pop-size or -p to set the size.\n");
        return false;
    }
    if (gensCount < 0) {
        printf("Generation count not set or invalid. Must be a positive number. Use --gens-count or -g to set the size.\n");
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

    float *d_images = nullptr;
    int *d_labels = nullptr;

    if (argMode == GPU) {
        cudaMalloc((void **) &d_images, size * 28 * 28 * sizeof(float));
        cudaMalloc((void **) &d_labels, size * sizeof(int));
        cudaMemcpy(d_images, images, size * 28 * 28 * sizeof(float), H2D);
        cudaMemcpy(d_labels, labels, size * sizeof(int), H2D);
    }
    while (generation < gensCount) {
        start = clock();
        calculateFitness(labels, images, networks, populationSize, size, fitness, d_labels, d_images, argMode);
        evolve(networks, fitness, populationSize, argMode);
        for (int i = 0; i < populationSize; i++) {
            if (fitness[i] > maxFitness) {
                maxFitness = fitness[i];
            }
        }
        if (generation == 50) {
            //write all fitnesses to file
            FILE *fp;
            fp = fopen("fitnesses.txt", "w");
            for (int i = 0; i < populationSize; i++) {
                fprintf(fp, "%f\n", fitness[i]);
            }
            fclose(fp);
        }
        printf("%d) Max fitness: %f, generation time: %6.3ld\n", generation, maxFitness, clock() - start);
        generation++;
    }

    return true;
}