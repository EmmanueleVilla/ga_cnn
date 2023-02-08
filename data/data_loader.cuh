//
// Created by emman on 08/02/2023.
//

#ifndef GA_CNN_DATA_LOADER_CUH
#define GA_CNN_DATA_LOADER_CUH

#include "../args_handler/args_extractor.cuh"

void loadData(int size, int *labels, float *images, MODE mode);

void loadLine(char *line, int *label, float *image, int index);

#endif //GA_CNN_DATA_LOADER_CUH
