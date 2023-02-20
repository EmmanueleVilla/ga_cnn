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
                      float *fitness,
                      int *d_labels,
                      float *d_images,
                      MODE mode) {

    if (mode == CPU) {
        calculateFitnessCPU(labels, images, networks, networkCount, dataCount, fitness);
    } else {
        calculateFitnessGPU(d_labels, d_images, networks, networkCount, dataCount, fitness);
    }
}