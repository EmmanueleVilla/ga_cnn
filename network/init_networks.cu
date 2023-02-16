//
// Created by emman on 13/02/2023.
//

#include <ctime>
#include "init_networks_utils.cuh"
#include "init_networks.cuh"
#include "../defines.cuh"

void initNetworks(float *networks, int count) {
    srand(time(NULL));
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < NUM_WEIGHTS; j++) {
            float rand = randGaussian();
            networks[i * NUM_WEIGHTS + j] = rand;
        }
    }
}