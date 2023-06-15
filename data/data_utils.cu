//
// Created by emman on 08/02/2023.
//

#include "data_utils.cuh"
#include "../output/output_utils.cuh"
#include <stdio.h>

/**
 * Prints the first [size] elements of the given array.
 * @param size
 * @param labels
 * @param images
 */
void printData(int size, const int *labels, const float *images) {
    for (int i = 0; i < size; i++) {
        printf("Label: %d\n", labels[i]);
        displayImage((float *) &images[i * 28 * 28], 28);
        printf("\n");
    }
}
