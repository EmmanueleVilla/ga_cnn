//
// Created by emman on 13/02/2023.
//

#ifndef GA_CNN_INIT_NETWORKS_CUH
#define GA_CNN_INIT_NETWORKS_CUH


#include "../data/data_loader.cuh"

#define FILTER_SIZE 3
#define NUM_FILTERS 5
#define POOLED_IMAGE_SIZE 13
#define NUM_WEIGHTS (NUM_FILTERS * FILTER_SIZE * FILTER_SIZE + POOLED_IMAGE_SIZE * POOLED_IMAGE_SIZE * NUM_FILTERS * 10)

/**
 * Initialize the networks.
 * Each network has 5 3x3 filters = 45 weights
 * This leads to 5 images of 26x26, 13x13 after pooling = 845 pixels in the dense layer
 * The output layer has 10 neurons
 * So, the total number of weights is 45 + 845 * 10 = NUM_WEIGHTS
 * @param networks the array of networks
 * @param count the number of networks
 * @param mode the mode to use: GPU or CPU
 */
void initNetworks(float *networks, int count);


#endif //GA_CNN_INIT_NETWORKS_CUH
