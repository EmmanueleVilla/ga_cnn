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
    stream = fopen("C:\\Users\\emman\\CLionProjects\\ga-cnn\\data\\mnist_train.csv", "r");
    if (stream != nullptr) {
        printf("The file '../../data/mnist_train.csv' was opened\n");
    } else {
        printf("The file '../../data/mnist_train.csv' was not opened\n");
        stream = fopen(".\\mnist_train.csv", "r");
        if (stream != nullptr) {
            printf("The fallback file 'mnist_train.csv' was opened\n");
        } else {
            stream = fopen("./mnist_train.csv", "r");
            printf("The fallback file 'mnist_train.csv' was not opened\n");
        }
    }
    return stream;
}
