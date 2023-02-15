//
// Created by emman on 13/02/2023.
//

#ifndef GA_CNN_FITNESS_CALCULATOR_GPU_CUH
#define GA_CNN_FITNESS_CALCULATOR_GPU_CUH


void calculateFitnessGPU(const int *labels, const float *images, const float *networks, int networkCount, int dataCount,
                         float *fitness);

#endif //GA_CNN_FITNESS_CALCULATOR_GPU_CUH
