//
// Created by emman on 08/02/2023.
//

#include "data_loader_cpu.cuh"
#include "data_loader.cuh"
#include "../network/init_networks.cuh"
#include <stdio.h>
#include <ctime>

void loadDataWithCPU(int size, int *labels, float *images, FILE *stream, float *networks, int networkCount) {
    clock_t start, dataLoaded, networkInitialized;

    start = clock();

    char line[729 * 4];
    int count = 0;
    while (fgets(line, 28 * 28 * 4, stream)) {
        if (count == size) {
            break;
        }
        //printf("Read line #%d: %s\n", count, line);
        loadLine(line, labels, images, count);
        count++;
    }
    fclose(stream);
    dataLoaded = clock();

    initNetworks(networks, 100);

    networkInitialized = clock();

    printf("** CPU TIMES **\n");
    printf("Start: \t%6.3ld\n", start);
    printf("Data loaded:\t %6.3ld\n", dataLoaded);
    printf("Network init: \t%6.3ld\n", networkInitialized);
}