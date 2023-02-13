//
// Created by emman on 13/02/2023.
//

#ifndef GA_CNN_FITNESS_CALCULATOR_CUH
#define GA_CNN_FITNESS_CALCULATOR_CUH


#include "../args_handler/args_extractor.cuh"

void
calculateFitness(int *labels, float *images, float *networks, int networkCount, int dataCount,
                 int *fitness, MODE mode);


#endif //GA_CNN_FITNESS_CALCULATOR_CUH
