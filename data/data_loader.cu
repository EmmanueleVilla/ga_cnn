//
// Created by emman on 08/02/2023.
//

#include "data_loader.cuh"
#include <stdio.h>
#include <corecrt.h>
#include <direct.h>

#define GetCurrentDir _getcwd

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
            //printf("Pixel: %f\n", pixel);
            image[index * 28 * 28 + count - 1] = pixel;
        }
        count++;
        ptr = strtok_s(nullptr, delim, &next_token);
    }
    printf("Count: %d\n", count);
}

void loadData(int size, int *labels, float *images) {
    FILE *stream = nullptr;
    char buff[FILENAME_MAX];
    GetCurrentDir(buff, FILENAME_MAX);
    printf("Current working dir: %s\n", buff);
    errno_t err;
    err = fopen_s(&stream, "../../data/mnist_train.csv", "r");
    if (err == 0) {
        printf("The file '../../data/mnist_train.csv' was opened\n");
    } else {
        printf("The file '../../data/mnist_train.csv' was not opened\n");
    }
    labels = (int *) malloc(size * sizeof(int));
    images = (float *) malloc(size * 28 * 28 * sizeof(float));

    char line[729 * 5];
    int count = 0;
    while (fgets(line, 28 * 28 * 5, stream)) {
        if (count == size) {
            break;
        }
        //printf("Read line #%d: %s\n", count, line);
        loadLine(line, labels, images, count);
        count++;
    }
    fclose(stream);


    for (int i = 0; i < size; i++) {
        printf("Label: %d\n", labels[i]);
        int column = 0;
        for (int j = i * 28 * 28; j < (i + 1) * 28 * 28 - 1; j++) {

            if (column == 28) {
                printf("\n");
                column = 0;
            }
            if (images[j] > 0.5f) {
                printf("X");
            } else {
                printf(" ");
            }
            column++;
        }
        printf("\n\n");
    }
}