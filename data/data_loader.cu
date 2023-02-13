//
// Created by emman on 08/02/2023.
//

#include "data_loader.cuh"
#include "data_utils.cuh"
#include "data_loader_cpu.cuh"
#include "data_loader_gpu.cuh"
#include "file_loader.cuh"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void loadLine(char *line, int *label, float *image, int index) {
    char delim[] = ",";
    char *next_token = nullptr;
    char *ptr = strtok_s(line, delim, &next_token);
    int count = 0;
    while (ptr != nullptr) {
        char *unused;
        int cell = (int) strtol(ptr, &unused, 10);
        if (count == 0) {
            label[index] = cell;
        } else {
            float pixel = (float) cell / 255.0f;
            image[index * 28 * 28 + count - 1] = pixel;
        }
        count++;
        ptr = strtok_s(nullptr, delim, &next_token);
    }
}

void loadData(int size, int *labels, float *images, MODE mode) {
    FILE *stream = readFile();
    if (mode == CPU) {
        loadDataWithCPU(size, labels, images, stream);
    } else if (mode == GPU) {
        loadDataWithGPU(size, labels, images, stream);
    }
    printData(2, labels, images);
}