#ifndef __LAMBDA_COMM_HPP__
#define __LAMBDA_COMM_HPP__


#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <climits>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "resource_comm.hpp"
#include "lambdaworker.hpp"
#include "../utils/utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"


#define SLEEP_PERIOD 5000   // sleep 5000us and then check the condition.
#define TIMEOUT_PERIOD 5000 // wait for up to 5000ms before relaunching
#define MIN_TIMEOUT 500     // at least wait for 500ms before relaunching
#define EXP_BACKOFF_FACTOR 1.5 // base of exponential backoff

class LambdaWorker;

/**
 *
 * Class of a lambda threads communication handler.
 *
 */
class LambdaComm : public ResourceComm {

public:

    LambdaComm(CommInfo &commInfo);
    ~LambdaComm();

    // For forward-prop.
    void newContextForward(unsigned layer, FeatType *dataBuf, FeatType *zData,
        FeatType *actData, unsigned numLocalVertices, unsigned numFeats,
        unsigned numFeatsNext);
    void requestForward(unsigned layer, bool lastLayer);
    void invokeLambdaForward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void waitLambdaForward(unsigned layer, bool lastLayer);

    // For backward-prop.
    void newContextBackward(unsigned layer, FeatType *oldGradBuf, FeatType *newGradBuf, std::vector<Matrix> *savedTensors, FeatType *targetBuf,
                            unsigned numLocalVertices, unsigned inFeatDim, unsigned outFeatDim, unsigned targetDim);
    void requestBackward(unsigned layer, bool lastLayer);
    void invokeLambdaBackward(unsigned layer, unsigned lambdaId, bool lastLayer);
    void waitLambdaBackward(unsigned layer, bool lastLayer);

    void relaunchLambda(bool forward, unsigned layer, unsigned lambdaId, bool lastLayer);

    // Send a message to the coordination server to shutdown.
    void sendShutdownMessage();

    // simple LambdaWorker initialization
    friend LambdaWorker::LambdaWorker(LambdaComm *manager);

// private:
    unsigned numLambdasForward;
    unsigned numLambdasBackward;
    unsigned numListeners;

    unsigned currLayer;

    bool halt;
    std::vector<bool> trainPartitions;

    double timeoutPeriod;

    // for relaunch timed-out lambdas
    unsigned countForward;
    bool *forwardLambdaTable;
    double forwardTimer;
    unsigned countBackward;
    bool *backwardLambdaTable;
    double backwardTimer;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    zmq::socket_t coordsocket;

    unsigned nodeId;
    unsigned numNodes;
    std::string nodeIp;
    unsigned dataserverPort;

    std::string coordserverIp;
    unsigned coordserverPort;
};


#endif // LAMBDA_COMM_HPP
