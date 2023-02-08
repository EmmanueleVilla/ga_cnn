//
// Created by emman on 08/02/2023.
//

#include "data_utils.cuh"
#include <stdio.h>

void printData(int size, const int *labels, const float *images) {
    for (int i = 0; i < size; i++) {
        printf("Label: %d\n", labels[i]);
        int column = 0;
        for (int j = i * 28 * 28; j < (i + 1) * 28 * 28 - 1; j++) {

            if (column == 28) {
                printf("\n");
                column = 0;
            }
            if (images[j] > 0.5f) {
                printf("X");
            } else {
                printf(" ");
            }
            column++;
        }
        printf("\n\n");
    }
}
