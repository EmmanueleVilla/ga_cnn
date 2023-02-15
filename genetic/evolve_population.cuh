//
// Created by emman on 15/02/2023.
//

#ifndef GA_CNN_EVOLVE_POPULATION_CUH
#define GA_CNN_EVOLVE_POPULATION_CUH

#include "../args_handler/args_extractor.cuh"

void evolve(float *networks, float *fitness, int popSize, MODE mode);

#endif //GA_CNN_EVOLVE_POPULATION_CUH
