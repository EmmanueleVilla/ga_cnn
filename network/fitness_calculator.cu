//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator.cuh"
#include "fitness_calculator_gpu.cuh"
#include "fitness_calculator_cpu.cuh"

void calculateFitness(int *labels,
                      float *images,
                      float *networks,
                      int networkCount,
                      int dataCount,
                      int *fitness,
                      MODE mode) {
    fitness = (int *) malloc(sizeof(int) * networkCount);
    if (mode == CPU) {
        calculateFitnessCPU(labels, images, networks, networkCount, dataCount, fitness);
    } else {
        calculateFitnessGPU(labels, images, networks, networkCount, dataCount, fitness);
    }
}