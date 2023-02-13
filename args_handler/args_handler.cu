//
// Created by emman on 08/02/2023.
//

#include "args_handler.cuh"
#include "args_handler_help.cuh"
#include "args_handler_version.cuh"
#include "args_extractor.cuh"
#include "../data/data_loader.cuh"
#include "../network/init_networks.cuh"
#include <stdio.h>


bool handle(int argc, char **argv) {
    if (argc == 2 && (strcmp("--help", argv[1]) == 0 || strcmp("-h", argv[1]) == 0)) {
        printHelp();
        return true;
    }

    if (argc == 2 && (strcmp("--version", argv[1]) == 0 || strcmp("-v", argv[1]) == 0)) {
        printVersion();
        return true;
    }

    enum MODE argMode = NONE;
    int size = -1;
    extractArgs(argc, argv, argMode, size);

    if (argMode == NONE) {
        printf("Mode not set. Use --mode or -m to set the mode.\n");
        return false;
    }

    if (size < 1 || size > 60000) {
        printf("Size not set or invalid. Must be between 1 and 60000. Use --size or -s to set the size.\n");
        return false;
    }

    int *labels = nullptr;
    float *images = nullptr;
    loadData(size, labels, images, argMode);

    float *networks = nullptr;
    initNetworks(networks, 100);
    return true;
}