//
// Created by emman on 13/02/2023.
//

#include "output_utils.cuh"
#include <stdio.h>

void displayImage(const float *image, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int index = i * size + j;
            if (image[index] > 0.5f) {
                printf("X");
            } else if (image[index] > 0.25f) {
                printf("x");
            } else if (image[index] > -0.25f) {
                printf(",");
            } else if (image[index] > -0.5f) {
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