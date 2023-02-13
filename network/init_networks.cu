//
// Created by emman on 13/02/2023.
//

#include <ctime>
#include <stdio.h>
#include "init_networks_utils.cuh"

void initNetworks(float *networks, int count) {
    networks = (float *) malloc(sizeof(float) * count * 7850);
    clock_t start, stop;
    float max = 0;
    float min = 99;
    start = clock();
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 7850; j++) {
            float rand = randGaussian();
            networks[i * 7850 + j] = rand;
            if (rand > max) {
                max = rand;
            }
            if (rand < min) {
                min = rand;
            }
        }
    }

    printf("Max: %f\n", max);
    printf("Min: %f\n", min);

    stop = clock();

    printf("\n%6.3ld", start);
    printf("\n\n%6.3ld\n", stop);
}