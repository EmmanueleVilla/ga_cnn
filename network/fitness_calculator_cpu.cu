//
// Created by emman on 13/02/2023.
//

#include "fitness_calculator_cpu.cuh"
#include "../output/output_utils.cuh"
#include <stdio.h>

int calculateNetworkLabelCPU(
        const float *images,
        const float *networks,
        int networkIndex,
        int imageIndex
) {
    int numFilters = 5;
    // Apply convolution filters to images[imageIndex] reading the weights from networks[networkIndex]
    float *conv = (float *) malloc(sizeof(float) * 26 * 26 * numFilters);
    printf("Calculating network %d label for image %d\n", networkIndex, imageIndex);
    int count = 0;
    float *image = (float *) &images[imageIndex * 28 * 28];
    displayImage(image, 28);
    float *network = (float *) &networks[networkIndex * 7850];

    for (int filter = 0; filter < numFilters; filter++) {
        printf("Filter %d\n", filter);
        displayFilter(networks, networkIndex, filter);
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
        float *convoluted = (float *) &conv[i * 26 * 26];
        printf("Convolution result\n");
        displayImage(convoluted, 26);
    }

    // Apply max pooling
    // Calculate dense layer
    // Return the label with the highest value
    return 5;
}

float calculateFitnessCPUSingleNetwork(
        const int *labels,
        const float *images,
        const float *networks,
        int dataCount,
        int netWorkIndex
) {
    printf("Calculating fitness of network %d\n", netWorkIndex);
    int correct = 0;
    for (int i = 0; i < dataCount; i++) {
        int label = labels[i];
        int networkLabel = calculateNetworkLabelCPU(images, networks, netWorkIndex, i);
        if (label == networkLabel) {
            correct++;
        }
        break;
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
    printf("Calculating fitness on CPU\n");
    for (int i = 0; i < networkCount; i++) {
        fitness[i] = calculateFitnessCPUSingleNetwork(labels, images, networks, dataCount, i);
        printf("Fitness of network %d: %f\n", i, fitness[i]);
        break;
    }
}


