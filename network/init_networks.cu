//
// Created by emman on 13/02/2023.
//

#include <ctime>
#include "init_networks_utils.cuh"

void initNetworks(float *networks, int count) {
    srand(time(NULL));
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 7850; j++) {
            float rand = randGaussian();
            networks[i * 7850 + j] = rand;
        }
    }
}