#include <unistd.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "engine/engine.hpp"
#include "utils/utils.hpp"
#include "commmanager/lambdaworker.hpp"
#include "commmanager/lambda_comm.hpp"

#include <chrono>
#include <iomanip>

/**
 *
 * Main entrance of the graph server logic.
 *
 */
int main(int argc, char *argv[])
{   
    // Initialize the engine.
    // The engine object is static and has been substantiated in Engine.cpp.
    engine.init(argc, argv);
    unsigned numEpochs = engine.getNumEpochs();
    unsigned valFreq = 1;

    if (engine.master())
        printLog(engine.getNodeId(),
                 "Number of epochs: %u, validation frequency: %u", numEpochs,
                 valFreq);

    // Sync all nodes before starting computation
    engine.makeBarrier();

    // Do specified number of epochs.
    Timer epochTimer;
    engine.run();

    // Procude the output files.
    engine.output();

    // Destroy the engine.
    engine.destroy();
    
    return 0;
}