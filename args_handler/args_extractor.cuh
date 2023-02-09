//
// Created by emman on 08/02/2023.
//

#ifndef GA_CNN_ARGS_EXTRACTOR_CUH
#define GA_CNN_ARGS_EXTRACTOR_CUH

enum MODE {
    NONE = 0,
    GPU = 1,
    CPU = 2
};

void extractArgs(int argc, char **argv, MODE &mode, int &size);


#endif //GA_CNN_ARGS_EXTRACTOR_CUH
