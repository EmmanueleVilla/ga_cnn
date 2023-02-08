//
// Created by emman on 08/02/2023.
//

#include "args_extractor.cuh"
#include <stdio.h>

void extractMode(int argc, char **argv, MODE &mode) {
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
    }
}

void extractArgs(int argc, char **argv, MODE &mode) {
    extractMode(argc, argv, mode);
}