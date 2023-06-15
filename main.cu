#include <iostream>
#include "args_handler/args_handler.cuh"

// TODO use this instead of clock
static double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double )tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
    cudaDeviceReset();
    if (!handle(argc, argv)) {
        printf("Parameters not recognized. Use -h or --help for help.\n");
    }
    // reset gpu for profiling
    cudaDeviceReset();
    return 0;
}
