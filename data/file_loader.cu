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
    const char *paths[] = {
            "C:\\Users\\emman\\CLionProjects\\ga-cnn\\data\\mnist_train.csv",
            ".\\mnist_train.csv",
            "/home/emmanuele/ga_cnn/data/mnist_train.csv",
            "./mnist_train.csv"
    };

    //foreach path in paths
    for (int i = 0; i < 4; i++) {
        FILE *stream = fopen(paths[i], "r");
        if (stream != nullptr) {
            printf("The file '%s' was opened\n", paths[i]);
            return stream;
        } else {
            printf("The file '%s' was not opened\n", paths[i]);
        }
    }
    return nullptr;
}
