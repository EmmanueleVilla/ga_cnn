//
// Created by emman on 08/02/2023.
//

#include "data_loader_gpu.cuh"
#include <stdio.h>

/**
 * This loads the data from the mnist_train.csv file using the GPU.
 * It parses the file and loads the labels and images into the given arrays.
 * @param size: the number of entries to be read
 * @param labels: the array to store the labels
 * @param images: the array to store the images
 */
void loadDataWithGPU(int size, int *labels, float *images, FILE *stream) {

}