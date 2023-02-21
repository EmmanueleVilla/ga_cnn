//
// Created by emman on 08/02/2023.
//

#ifndef GA_CNN_DATA_LOADER_CPU_CUH
#define GA_CNN_DATA_LOADER_CPU_CUH

#include <stdio.h>

/**
 * This loads the data from the mnist_train__.csv file using the CPU.
 * It parses the file and loads the labels and images into the given arrays.
 * This process takes approximately 7-8 seconds.
 * @param size: the number of entries to be read
 * @param labels: the array to store the labels
 * @param images: the array to store the images
 */
void loadDataWithCPU(int size, int *labels, float *images, FILE *stream, float *networks, int networkCount);


#endif //GA_CNN_DATA_LOADER_CPU_CUH
