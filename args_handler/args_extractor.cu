//
// Created by emman on 08/02/2023.
//

#include "args_extractor.cuh"
#include <stdio.h>

void extractArgs(int argc, char **argv, MODE &mode, int &size, int &popSize) {
    for (int i = 1; i < argc; i++) {
        if (strcmp("--mode", argv[i]) == 0 || strcmp("-m", argv[i]) == 0) {
            if (i + 1 < argc) {
                if (strcmp("GPU", argv[i + 1]) == 0) {
                    mode = GPU;
                    printf("Mode set to gpu.\n");
                } else if (strcmp("CPU", argv[i + 1]) == 0) {
                    mode = CPU;
                    printf("Mode set to cpu.\n");
                }
            }
        }
        if (strcmp("--dataSize", argv[i]) == 0 || strcmp("-s", argv[i]) == 0) {
            if (i + 1 < argc) {
                char *unused;
                size = (int) strtol(argv[i + 1], &unused, 10);
                printf("Data size set to %d.\n", size);
            }
        }

        if (strcmp("--popSize", argv[i]) == 0 || strcmp("-p", argv[i]) == 0) {
            if (i + 1 < argc) {
                char *unused;
                popSize = (int) strtol(argv[i + 1], &unused, 10);
                printf("Pop size set to %d.\n", popSize);
            }
        }
    }
}