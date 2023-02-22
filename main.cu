#include <iostream>
#include "args_handler/args_handler.cuh"

int main(int argc, char **argv) {

    if (!handle(argc, argv)) {
        printf("Parameters not recognized. Use -h or --help for help.\n");
    }
    // reset gpu for profiling
    cudaDeviceReset();
    return 0;
}
