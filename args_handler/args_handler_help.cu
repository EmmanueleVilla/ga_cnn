//
// Created by emman on 08/02/2023.
//

#include "args_handler_help.cuh"
#include <stdio.h>

void printHelp() {
    printf("This is the program made for the GPU Computing course @ UniMi by Emmanuele Villa.\n");
    printf("usage: ga-cnn [OPTIONS]\n");
    printf("\t-m, --mode <GPU|CPU>\tSelects the mode to run the program in.\n");
    printf("\t[-d, --debug]\t\tEnables the debug mode.\n");
    printf("\t[-h, --help]\t\tPrints the command arguments help.\n");
    printf("\t[-v, --version]\t\tPrints the program version.\n");
}