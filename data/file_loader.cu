//
// Created by emman on 08/02/2023.
//

#include "file_loader.cuh"
#include <stdio.h>

/**
 * This function reads the mnist_train.csv file and return a FILE pointer to it.
 * @return
 */
FILE *readFile() {
    FILE *stream = nullptr;
    errno_t err;
    err = fopen_s(&stream, "C:\\Users\\emman\\CLionProjects\\ga-cnn\\data\\mnist_train.csv", "r");
    if (err == 0) {
        printf("The file '../../data/mnist_train.csv' was opened\n");
    } else {
        printf("The file '../../data/mnist_train.csv' was not opened\n");
    }
    return stream;
}