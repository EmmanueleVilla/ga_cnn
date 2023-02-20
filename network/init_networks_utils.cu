//
// Created by emman on 13/02/2023.
//

#include "init_networks_utils.cuh"
#include <corecrt_math.h>

float randGaussian() {
    float a = (float) rand() / (float) RAND_MAX;
    float b = (float) rand() / (float) RAND_MAX;
    if (a == 0) {
        a = 0.0000001;
    }
    if (b == 0) {
        b = 0.0000001;
    }

    float R0 = sqrt(-2.0 * log(a)) * cos(2 * M_PI * b);

    return R0 / 5;
}