//
// Created by emman on 08/02/2023.
//

#include "data_loader.cuh"
#include "file_loader.cuh"
#include <stdio.h>
#include <ctime>

void printData(int size, const int *labels, const float *images) {
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

/**
 * This loads the data from the mnist_train.csv file using the CPU.
 * It parses the file and loads the labels and images into the given arrays.
 * This process takes approximately 7-8 seconds.
 * @param size: the number of entries to be read
 * @param labels: the array to store the labels
 * @param images: the array to store the images
 */
void loadDataWithCPU(int size, int *labels, float *images) {
    clock_t start, stop;

    FILE *stream = readFile();

    start = clock();
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
    stop = clock();

    printf("%6.3ld", start);
    printf("\n\n%6.3ld", stop);
}

/**
 * This loads the data from the mnist_train.csv file using the GPU.
 * It parses the file and loads the labels and images into the given arrays.
 * @param size: the number of entries to be read
 * @param labels: the array to store the labels
 * @param images: the array to store the images
 */
void loadDataWithGPU(int size, int *labels, float *images) {

}

void loadData(int size, int *labels, float *images, MODE mode) {
    if (mode == CPU) {
        loadDataWithCPU(size, labels, images);
    } else if (mode == GPU) {
        loadDataWithGPU(size, labels, images);
    }
}