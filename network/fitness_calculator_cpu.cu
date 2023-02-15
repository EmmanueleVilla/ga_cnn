//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_cpu.cuh"
#include "../output/output_utils.cuh"
#include "init_networks.cuh"
#include <stdio.h>

int calculateNetworkLabelCPU(
        const float *images,
        const float *networks,
        int networkIndex,
        int imageIndex
) {
    int numFilters = 5;
    // Apply convolution filters to images[imageIndex] reading the weights from networks[networkIndex]
    auto *conv = (float *) malloc(sizeof(float) * 26 * 26 * numFilters);
    int count = 0;
    auto *image = (float *) &images[imageIndex * 28 * 28];
    //displayImage(image, 28);
    auto *network = (float *) &networks[networkIndex * NUM_WEIGHTS];

    for (int filter = 0; filter < numFilters; filter++) {
        //printf("Filter %d\n", filter);
        //displayFilter(networks, networkIndex, filter);
        for (int i = 1; i < 27; i++) {
            for (int j = 1; j < 27; j++) {
                float sum = 0;
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        sum += image[(i + k - 1) * 28 + (j + l - 1)] * network[filter * 9 + k * 3 + l];
                    }
                }
                conv[count] = sum;
                count++;
            }
        }
    }

    for (int i = 0; i < numFilters; i++) {
        auto *convoluted = (float *) &conv[i * 26 * 26];
        //printf("Convolution result\n");
        //displayImage(convoluted, 26);
    }

    // Apply max pooling
    auto *pooled = (float *) malloc(sizeof(float) * 13 * 13 * numFilters);

    count = 0;
    for (int filter = 0; filter < numFilters; filter++) {
        //printf("Pooling\n");
        for (int i = 1; i < 26; i += 2) {
            for (int j = 1; j < 26; j += 2) {
                float max = 0;
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 2; l++) {
                        float value = conv[filter * 26 * 26 + (i + k - 1) * 26 + (j + l - 1)];
                        if (value > max) {
                            max = value;
                        }
                    }
                }
                pooled[count] = max;
                count++;
            }
        }
    }

    for (int i = 0; i < numFilters; i++) {
        auto *pool = (float *) &pooled[i * 13 * 13];
        //printf("Pooling result\n");
        //displayImage(pool, 13);
    }

    // Calculate dense layer
    auto *output = (float *) malloc(sizeof(float) * 10);
    int max = -999;
    for (int i = 0; i < 10; i++) {
        float sum = 0;
        for (int j = 0; j < 13 * 13; j++) {
            sum += pooled[i * 13 * 13 + j] * network[45 + i * 13 * 13 + j];
        }
        output[i] = sum;
        if (sum > max || max == -999) {
            max = i;
        }
    }

    free(conv);
    free(pooled);
    free(output);

    // Return the label with the highest value
    return max;
}

float calculateFitnessCPUSingleNetwork(
        const int *labels,
        const float *images,
        const float *networks,
        int dataCount,
        int netWorkIndex
) {
    //printf("Calculating fitness of network %d\n", netWorkIndex);
    int correct = 0;
    for (int i = 0; i < dataCount; i++) {
        int label = labels[i];
        int networkLabel = calculateNetworkLabelCPU(images, networks, netWorkIndex, i);
        if (label == networkLabel) {
            correct++;
        }
    }
    return ((float) correct / (float) dataCount * 100);
}


void calculateFitnessCPU(
        const int *labels,
        const float *images,
        const float *networks,
        int networkCount,
        int dataCount,
        float *fitness
) {
    //printf("Calculating fitness on CPU\n");
    for (int i = 0; i < networkCount; i++) {
        fitness[i] = calculateFitnessCPUSingleNetwork(labels, images, networks, dataCount, i);
        //printf("Fitness of network %d: %f\n", i, fitness[i]);
    }
}


