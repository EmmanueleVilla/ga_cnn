//
// Created by emman on 08/02/2023.
//

#include "data_loader_cpu.cuh"
#include "data_loader.cuh"
#include <stdio.h>
#include <ctime>

void loadDataWithCPU(int size, int *labels, float *images, FILE *stream) {
    clock_t start, stop;

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
    stop = clock();

    printf("\n%6.3ld", start);
    printf("\n\n%6.3ld\n", stop);
}