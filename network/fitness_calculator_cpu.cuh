//
// Created by emman on 13/02/2023.
//

#ifndef GA_CNN_FITNESS_CALCULATOR_CPU_CUH
#define GA_CNN_FITNESS_CALCULATOR_CPU_CUH


void calculateFitnessCPU(
        const int *labels,
        const float *images,
        const float *networks,
        int networkCount,
        int dataCount,
        float *fitness
);


#endif //GA_CNN_FITNESS_CALCULATOR_CPU_CUH
