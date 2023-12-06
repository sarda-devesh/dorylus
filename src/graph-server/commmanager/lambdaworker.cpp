#include "lambdaworker.hpp"
#include "lambda_comm.hpp"

#include <chrono>
#include <iomanip>

#include <iostream>
#include <map>
#include <sstream>
#include <mutex>

static void nofree(void* data, void* hint) {}

#define BIND_PORT 7000
#define IDENTITY_SIZE (sizeof(Chunk) + sizeof(unsigned))

std::vector<char> constructIdentity(Chunk &chunk) {
    std::vector<char> identity(IDENTITY_SIZE);

    std::random_device rd;
    std::mt19937 generator(rd());

    unsigned rand = generator();
    std::memcpy(identity.data(), &chunk, sizeof(chunk));
    std::memcpy(identity.data() + sizeof(chunk), &rand, sizeof(unsigned));

    return identity;
}

/**
 *
 * LambdaWorker constructor & destructor.
 *
 */
LambdaWorker::LambdaWorker(LambdaComm *manager_) :
  manager(manager_), actual_socket(manager->ctx, ZMQ_REP) {
    // Generate an identity
    Chunk chunk;
    std::vector<char> identity = constructIdentity(chunk);

    // Set the bindings for the current socket
    // printLog(manager->nodeId, "Setting timeout for actual socket");
    actual_socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());

    // Listen for incoming requests
    char actual_port[50];
    // printLog(manager->nodeId, "Actual socket connecting to tcp://*:8000");
    sprintf(actual_port, "tcp://*:8000");
    actual_socket.bind(actual_port);
}

LambdaWorker::~LambdaWorker() {
    actual_socket.setsockopt(ZMQ_LINGER, 0);
    actual_socket.close();
}


/**
 *
 * Lambdaworker is a wrapper over the sender & receiver thread.
 *
 */
void
LambdaWorker::work(unsigned _wid) {
    wid = _wid;
    try {
        while (!(manager->halt)) {
            zmq::message_t identity;

            // recv will return false if timed out.
            if (!actual_socket.recv(&identity)) {
                continue;
            }
            if (identity.size() != IDENTITY_SIZE) {
                printLog(manager->nodeId, "identity size %u", identity.size());
                continue;
            }
            recvTS = timestamp_ms();

            // Read in the operation
            char * data_ptr = (char *) identity.data();
            OP op = parse<OP>(data_ptr, 0);

            // Get the chunk
            Chunk chunk;
            memcpy(&chunk, data_ptr + sizeof(unsigned), sizeof(Chunk));

            switch (op) {
                case (OP::PULL): {
                    printLog(manager->nodeId, "Calling sendTensors");
                    sendTensors(identity, chunk);
                    break;
                }
                case (OP::PULLE): {
                    printLog(manager->nodeId, "Calling sendEdgeTensor");
                    sendEdgeTensor(identity, chunk);
                    break;
                }
                case (OP::PULLEINFO): {
                    printLog(manager->nodeId, "Calling sendEdgeInfo");
                    sendEdgeInfo(identity, chunk);
                    break;
                }
                case (OP::PUSH): {
                    printLog(manager->nodeId, "Calling recvTensors");
                    recvTensors(identity, chunk);
                    break;
                }
                case (OP::PUSHE): {
                    printLog(manager->nodeId, "Calling recvETensors");
                    recvETensors(identity, chunk);
                    break;
                }
                case (OP::EVAL): {
                    printLog(manager->nodeId, "Calling recvEvalData");
                    recvEvalData(identity, chunk);
                    break;
                }
                case (OP::FIN): {
                    printLog(manager->nodeId, "Calling markFinish");
                    markFinish(identity, chunk);
                    break;
                }
                case (OP::TERM): {
                    // terminate by weight server
                    printLog(manager->nodeId, "Weight server convergence");
                    CONVERGE_STATE cs = (CONVERGE_STATE)parse<int>((char *) identity.data(), 1);
                    manager->engine->convergeState = cs;
                    break;
                }
                default: {
                    printLog(manager->nodeId, "unknown op %d, part id %d", op, chunk.localId);
                    break;  /** Not an op that I care about. */
                }
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}

#define MAX_TENSOR_NAME_LEN 5

/**
 *
 * Sending & receiving messages to / from lambda threads.
 *
 */
void LambdaWorker::sendTensors(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();

    // printLog(manager->nodeId, "sendTensors has exist of %d", exist);

    if (exist) {
        // printLog(manager->nodeId, "Inside of exist if statement");
        unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1; // YIFAN: fix this
        // printLog(manager->nodeId, "Getting feat layer %d", featLayer);
        TensorMap& tensorMap = manager->savedNNTensors[featLayer];
        unsigned more = 1;

        while (more) {
            zmq::message_t message;
            actual_socket.recv(&message);
            // printLog(manager->nodeId, "Got message of size %d", message.size());

            std::string tensor_name = std::string(static_cast<char*>(message.data()), message.size());
            // printLog(manager->nodeId, "Got name of %s and of size %d", tensor_name.c_str(), tensor_name.size());

            auto found = tensorMap.find(tensor_name);
            // printLog(manager->nodeId, "Found has val of %d for name %s", found != tensorMap.end(), tensor_name.c_str());

            if (found == tensorMap.end()) {
                // printLog(manager->nodeId, "Requested tensor '%s' not found for layer %u", tensor_name.c_str(), featLayer);
                zmq::message_t errorHeader(TENSOR_HDR_SIZE);
                populateHeader(errorHeader.data(), ERR_HEADER_FIELD, tensor_name.c_str());
                actual_socket.send(client_id, ZMQ_SNDMORE);
                actual_socket.send(errorHeader);
                return;
            } else {
                Matrix& reqMatrix = found->second;
                // printLog(manager->nodeId, "Calling sendTensor with more of %d", more);
                sendTensor(reqMatrix, chunk, more);
            }
        }
    } else {
        size_t usize = sizeof(unsigned);
        unsigned more = 1;
        while (more) {
            // printLog(manager->nodeId, "Not exists expecting header of size %d", TENSOR_HDR_SIZE);
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            actual_socket.recv(&tensorHeader);
            actual_socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        actual_socket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(TENSOR_HDR_SIZE);
        populateHeader((char*) header.data(), ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD, ERR_HEADER_FIELD);
        printLog(manager->nodeId, "Not exists sending error header of size %d", TENSOR_HDR_SIZE);
        actual_socket.send(header);

        char errMsg[1024];
        sprintf(errMsg, "[ ERROR ] when sending chunk: %s %u",
            chunk.str().c_str(), chunk.vertex);
        printLog(manager->nodeId, errMsg);
    }
}

void LambdaWorker::recvTensors(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();

    printLog(manager->nodeId, "recvTensors has exist of %d", exist);

    if (exist) {
        int ret = 0;
        unsigned more = 1;
        size_t usize = sizeof(unsigned);

        while (more && ret == 0) {
            printLog(manager->nodeId, "recvTensors calling recvTensor");
            ret = recvTensor(chunk);
            printLog(manager->nodeId, "recvTensor returned %d", ret);
            actual_socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        if (ret == 0 && manager->NNRecv(chunk)) {
            zmq::message_t ack;
            printLog(manager->nodeId, "Sending sucess ack");
            actual_socket.send(ack);
        } else { // Error, Give up this chunk
            zmq::message_t ack(3 * sizeof(unsigned));
            *(int *)(ack.data()) = -1;
            printLog(manager->nodeId, "Sending failure ack");
            actual_socket.send(ack);
        }
    } else {

        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        unsigned more = 1;
        size_t usize = sizeof(unsigned);

        while (more) {
            zmq::message_t tensorData;
            printLog(manager->nodeId, "Recv tensor header");
            actual_socket.recv(&tensorHeader);
            printLog(manager->nodeId, "Recv tensor data");
            actual_socket.recv(&tensorData);

            actual_socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        zmq::message_t ack(3 * sizeof(unsigned));
        *(int *)(ack.data()) = -1;
        printLog(manager->nodeId, "Send ack");
        actual_socket.send(ack);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::recvETensors(zmq::message_t& client_id, Chunk& chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    
    if (exist) {
        int ret = 0;
        unsigned more = 1;
        size_t usize = sizeof(unsigned);
        while (more && ret == 0) {
            // printLog(manager->nodeId, "recv");
            ret = recvETensor(chunk);
            actual_socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }

        if (ret == 0 && manager->NNRecv(chunk)) {
            zmq::message_t ack;
            actual_socket.send(client_id, ZMQ_SNDMORE);
            actual_socket.send(ack);
        } else { // Error, Give up this chunk
            zmq::message_t ack(3 * sizeof(unsigned));
            *(int *)(ack.data()) = -1;
            actual_socket.send(client_id, ZMQ_SNDMORE);
            actual_socket.send(ack);
        }
    } else {
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        unsigned more = 1;
        size_t usize = sizeof(unsigned);
        while (more) {
            zmq::message_t tensorData;
            actual_socket.recv(&tensorHeader);
            actual_socket.recv(&tensorData);

            actual_socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
        }
        zmq::message_t ack(3 * sizeof(unsigned));
        *(int *)(ack.data()) = -1;
        actual_socket.send(client_id, ZMQ_SNDMORE);
        actual_socket.send(ack);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::recvEvalData(zmq::message_t &client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        zmq::message_t evalMsg(2 * sizeof(float));
        actual_socket.recv(&evalMsg);

        float acc = *((float *)evalMsg.data());
        float loss = *(((float *)evalMsg.data()) + 1);

        manager->accMtx.lock();
        auto &accLoss = manager->accLossTable[chunk.epoch];
        accLoss.acc += acc;
        accLoss.loss += loss;
        accLoss.vtcsCnt += chunk.upBound - chunk.lowBound;
        accLoss.chunkCnt++;
        // printLog(manager->nodeId, "epoch %u, chunk %u/%u, acc %.3f, loss %.3f", chunk.epoch, accLoss.chunkCnt,
        //     manager->engine->numLambdasForward, accLoss.acc / accLoss.vtcsCnt, accLoss.loss / accLoss.vtcsCnt);
        if (accLoss.chunkCnt == manager->engine->numLambdasForward) {
            printLog(manager->nodeId, "epoch %u, acc %.3f, loss %.3f", chunk.epoch,
                accLoss.acc / accLoss.vtcsCnt, accLoss.loss / accLoss.vtcsCnt);
        }
        manager->accMtx.unlock();
    } else {
        zmq::message_t evalMsg(2 * sizeof(float));
        actual_socket.recv(&evalMsg);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate accloss. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::markFinish(zmq::message_t& client_id, Chunk &chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();
    if (exist) {
        zmq::message_t ack(3 * sizeof(unsigned));
        if (manager->NNRecv(chunk)) {
            *(int *)(ack.data()) = 0;
        } else { // Error, Give up this chunk
            *(int *)(ack.data()) = -1;
        }
        actual_socket.send(client_id, ZMQ_SNDMORE);
        actual_socket.send(ack);
    } else {
        zmq::message_t ack(3 * sizeof(unsigned));
        *(int *)(ack.data()) = -1;
        actual_socket.send(client_id, ZMQ_SNDMORE);
        actual_socket.send(ack);

        std::string errMsg = "[ ERROR ] when receiving from " + chunk.str() + ": ";
        errMsg += "Received duplicate results. Discarding...";
        printLog(manager->nodeId, errMsg.c_str());
    }
}

void LambdaWorker::sendTensor(Matrix &tensor, Chunk &chunk, unsigned& more) {
    FeatType *dptr = tensor.get(chunk.lowBound);
    unsigned rows = chunk.upBound - chunk.lowBound;
    unsigned cols = tensor.getCols();
    printLog(manager->nodeId, "Rows of %d and Cols of %d", rows, cols);

    // Send the haeder
    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), OP::PULL, tensor.name().c_str(), rows, cols);
    unsigned bufSize = rows * cols * sizeof(FeatType);
    printLog(manager->nodeId, "Sending tensor header of size %d", responseHeader.size());
    actual_socket.send(responseHeader, ZMQ_SNDMORE);    

    // Send the tensor data
    size_t usize = sizeof(unsigned);
    actual_socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    zmq::message_t tensorData(dptr, bufSize, nofree, NULL);

    printLog(manager->nodeId, "Sending tensor data of size %d", tensorData.size());
    if (!more) {
        actual_socket.send(tensorData);
    } else {
        actual_socket.send(tensorData, ZMQ_SNDMORE);
    }
}

// ASSUMPTION: Only one edge tensor requested at a time
void LambdaWorker::sendEdgeTensor(zmq::message_t& client_id, Chunk& chunk) {
    manager->timeoutMtx.lock();
    bool exist = manager->timeoutTable.find(chunk) != manager->timeoutTable.end();
    manager->timeoutMtx.unlock();

    std::ostringstream combinedKeys;
    for (const auto& pair : manager->timeoutTable) {
        combinedKeys << "Chunk:" << pair.first.localId << "," << pair.first.globalId << "," << pair.first.layer;
        combinedKeys << pair.first.layer << "," << pair.first.dir << "," << pair.first.vertex << ";";
    }

    // printLog(manager->nodeId, "sendEdgeTensor has exist of %d with keys of %s", exist, combinedKeys.str());

    if (exist) {
        unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
        TensorMap& tMap = manager->savedNNTensors[featLayer];
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        actual_socket.recv(&tensorHeader);

        std::string name = parseName((char*)tensorHeader.data());
        auto found = tMap.find(name);
        if (found == tMap.end()) {
            actual_socket.send(client_id, ZMQ_SNDMORE);
            printLog(manager->nodeId, "Requested tensor '%s' not found for layer %u",
                name.c_str(), featLayer);
            zmq::message_t errorHeader(TENSOR_HDR_SIZE);
            populateHeader(errorHeader.data(), ERR_HEADER_FIELD, name.c_str());
            actual_socket.send(client_id, ZMQ_SNDMORE);
            actual_socket.send(errorHeader);
        } else {
            // printLog(manager->nodeId, "SENDING");
            actual_socket.send(client_id, ZMQ_SNDMORE);
            sendEdgeTensorChunk(found->second, chunk);
        }
    } else {
        printLog(manager->nodeId, "Chunk %u DONE", chunk.localId);
        actual_socket.send(client_id, ZMQ_SNDMORE);
        zmq::message_t header(TENSOR_HDR_SIZE);
        populateHeader((char*) header.data(), CHUNK_DNE_ERR, CHUNK_DNE_ERR);
        actual_socket.send(header);

        char errMsg[1024];
        sprintf(errMsg, "[ ERROR ] when sending chunk: %s %u",
            chunk.str().c_str(), chunk.vertex);
        printLog(manager->nodeId, errMsg);
    }
}

void LambdaWorker::sendEdgeTensorChunk(Matrix& eTensor, Chunk& chunk) {
    CSCMatrix<EdgeType>& csc = (manager->engine->graph).forwardAdj;
    unsigned nChunkEdges = csc.columnPtrs[chunk.upBound] - csc.columnPtrs[chunk.lowBound];
    unsigned long long baseIndex = csc.columnPtrs[chunk.lowBound];

    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    zmq::message_t edgeDataMsg(nChunkEdges * sizeof(FeatType));
    populateHeader(responseHeader.data(), nChunkEdges, 1);
    std::memcpy(edgeDataMsg.data(), eTensor.getData() + baseIndex, nChunkEdges * sizeof(FeatType));

    actual_socket.send(responseHeader, ZMQ_SNDMORE);
    actual_socket.send(edgeDataMsg);
}

// JOHN: A lot of information needed for this has to be accessed through engine
//  which is ugly. TODO: Extend matrix class to EdgeMatrix so that all infomration
//  can be encapsulated without accessing engine
void LambdaWorker::sendEdgeInfo(zmq::message_t& client_id, Chunk& chunk) {
    // printLog(manager->nodeId, "SEND EDGE INFO");
    CSCMatrix<EdgeType>& csc = (manager->engine->graph).forwardAdj;
    unsigned numLvids = chunk.upBound - chunk.lowBound;
    unsigned numChunkEdges = csc.columnPtrs[chunk.upBound] - csc.columnPtrs[chunk.lowBound];

    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), numLvids, numChunkEdges);

    zmq::message_t edgeChunkInfoMsg((numLvids + 1) * sizeof(unsigned long long));
    std::memcpy(edgeChunkInfoMsg.data(), csc.columnPtrs + chunk.lowBound, (numLvids + 1) * sizeof(unsigned long long));
    // std::string colPtrsStr = "Actual colPtrs: ";
    // for (unsigned lvid = chunk.lowBound; lvid <= chunk.upBound; ++lvid) {
    //     colPtrsStr += std::to_string(csc.columnPtrs[lvid]) + " ";
    // }
    // unsigned long long* colPtrMsgData = (unsigned long long*) edgeChunkInfoMsg.data();
    // colPtrsStr += "\ncolPtrData colPtrs: ";
    // for (unsigned lvid = 0; lvid <= numLvids; ++lvid) {
    //     colPtrsStr += std::to_string(colPtrMsgData[lvid]) + " ";
    // }
    actual_socket.send(client_id, ZMQ_SNDMORE);
    actual_socket.send(responseHeader, ZMQ_SNDMORE);
    actual_socket.send(edgeChunkInfoMsg);
    // printLog(manager->nodeId, "MESSGAES SENT FOR EDGE INFO");
}

int LambdaWorker::recvTensor(Chunk &chunk) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    printLog(manager->nodeId, "recvTensor trying to read header");
    actual_socket.recv(&tensorHeader);
    zmq::message_t tensorData;
    printLog(manager->nodeId, "recvTensor trying to read data");
    actual_socket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    printLog(manager->nodeId, "recvTensor got name of %s", name.c_str());
    if (!chunk.vertex)
        return 0;

    unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
    if (manager->engine->gnn_type == GNN::GAT && name == "grad") {
        featLayer = chunk.layer - 1;
    }
    TensorMap& tensorMap = manager->savedNNTensors[featLayer];
    auto found = tensorMap.find(name);
    if (found == tensorMap.end()) {
        printLog(manager->nodeId, "Lambda %s returned unknown tensor %u:'%s'. Make sure to allocate it before running lambdas!",
                 chunk.str().c_str(), featLayer, name.c_str());
        return 1;
    }

    // printLog(manager->nodeId, "get Tensor %s (%u, %u) from %s, dst %s",
    //          name.c_str(), chunk.upBound - chunk.lowBound,
    //          tensorData.size() / (chunk.upBound - chunk.lowBound) / 4,
    //          chunk.str().c_str(), found->second.shape().c_str());
    FeatType* dptr = found->second.get(chunk.lowBound);
    std::memcpy(dptr, tensorData.data(), tensorData.size());

    return 0;
}

int LambdaWorker::recvETensor(Chunk& chunk) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    actual_socket.recv(&tensorHeader);
    zmq::message_t tensorData;
    actual_socket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
    TensorMap& tensorMap = manager->savedNNTensors[featLayer];
    auto found = tensorMap.find(name);
    if (found == tensorMap.end()) {
        printLog(manager->nodeId, "Lambda %s returned unknown tensor '%s'. Make sure to allocate it before running lambdas!",
                 chunk.str().c_str(), name.c_str());
        return 1;
    }

    // printLog(manager->nodeId, "Copying edge values");
    CSCMatrix<EdgeType>& csc = (manager->engine->graph).forwardAdj;
    FeatType* eDptr = found->second.get(csc.columnPtrs[chunk.lowBound]);
    std::memcpy(eDptr, tensorData.data(), tensorData.size());

    return 0;
}
