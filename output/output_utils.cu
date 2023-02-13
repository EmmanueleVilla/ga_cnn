//
// Created by emman on 13/02/2023.
//

#include "output_utils.cuh"
#include <stdio.h>

void displayImage(const float *images, int imageIndex) {
    for (int i = 1; i < 27; i++) {
        for (int j = 1; j < 27; j++) {
            int index = imageIndex * 28 * 28 + i * 28 + j;
            if (images[index] > 0.85f) {
                printf("X");
            } else if (images[index] > 0.75f) {
                printf("x");
            } else if (images[index] > 0.5f) {
                printf(".");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}

void displayFilter(const float *filters, int networkIndex, int filterIndex) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int index = networkIndex * 7850 + filterIndex * 9 + i * 3 + j;
            if (filters[index] > 0.5f) {
                printf("O");
            } else if (filters[index] > 0.25f) {
                printf("o");
            } else if (filters[index] > -0.25f) {
                printf("x");
            } else if (filters[index] > -0.5f) {
                printf(".");
            }
        }
        printf("\n");
    }
}