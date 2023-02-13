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
    // Apply convolution filters to images[imageIndex] reading the weights from networks[networkIndex]
    float *conv = (float *) malloc(sizeof(float) * 26 * 26);
    printf("Calculating network %d label for image %d\n", networkIndex, imageIndex);
    int count = 0;
    displayImage(images, imageIndex);

    for (int filter = 0; filter < 5; filter++) {
        printf("Filter %d\n", filter);
        displayFilter(networks, networkIndex, filter);
        for (int i = 1; i < 27; i++) {
            for (int j = 1; j < 27; j++) {
                int filterIndex = filter * 9;
                conv[count] = 0;
            }
        }
    }

    // Apply max pooling
    // Calculate dense layer
    // Return the label with the highest value
    return 1;
}

int calculateFitnessCPUSingleNetwork(
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
    return (int) ((float) correct / dataCount * 100);
}


void calculateFitnessCPU(
        const int *labels,
        const float *images,
        const float *networks,
        int networkCount,
        int dataCount,
        int *fitness
) {
    printf("Calculating fitness on CPU\n");
    for (int i = 0; i < networkCount; i++) {
        fitness[i] = calculateFitnessCPUSingleNetwork(labels, images, networks, dataCount, i);
        printf("Fitness of network %d: %d\n", i, fitness[i]);
        break;
    }
}


