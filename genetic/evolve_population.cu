//
// Created by emman on 15/02/2023.
//

#include "evolve_population.cuh"
#include "evolve_population_cpu.cuh"
#include "evolve_population_gpu.cuh"

void evolve(float *networks, float *fitness, int popSize, MODE mode) {
    //if (mode == CPU) {
    evolveCPU(networks, fitness, popSize);
    //} else {
    //    evolveGPU(networks, fitness, popSize);
    //}
}