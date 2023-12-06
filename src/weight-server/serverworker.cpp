#include "serverworker.hpp"


static void nofree(void* data, void* hint) {}

/**
 *
 * ServerWorker constructor & destructor.
 *
 */
ServerWorker::ServerWorker(zmq::context_t& ctx_, WeightServer& _ws, unsigned _tid)
    : tid(_tid), ctx(ctx_), workersocket(ctx, ZMQ_DEALER), ws(_ws), lambdasocket(ctx, ZMQ_REP) {
    workersocket.setsockopt(ZMQ_BACKLOG, 500);
    workersocket.connect("inproc://backend");

    // Listen for incoming requests
    if(_tid == 0) {
        char port[50];
        std::cout << "Lambda socket connecting to tcp://*:9000" << std::endl;
        sprintf(port, "tcp://*:9000");
        lambdasocket.bind(port);
    }   
}

ServerWorker::~ServerWorker() {
    workersocket.setsockopt(ZMQ_LINGER, 0);
    workersocket.close();

    lambdasocket.setsockopt(ZMQ_LINGER, 0);
    lambdasocket.close();
}

void ServerWorker::lambda_worker() {
    // std::cout << "[ Weight ] Starts listening for lambdas' requests..." << std::endl;
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;
            // std::cout << "Lambda socket listening for messages " << std::endl;

            lambdasocket.recv(&identity);
            lambdasocket.recv(&header);

            OP op = parse<OP>((char *) header.data(), 0);
            std::cout << "Got OP of " << op << std::endl;

            switch (op) {
                case (OP::PUSH): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    std::cout << "Calling recvTensors " << std::endl;
                    recvTensors(lambdasocket, identity, chunk);
                    break;
                }
                case (OP::PULL): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    std::cout << "Calling sendTensors " << std::endl;
                    sendTensors(lambdasocket, identity, chunk);
                    break;
                }
                case (OP::EVAL): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    std::cout << "Calling recvEvalData " << std::endl;
                    recvEvalData(lambdasocket, identity, chunk);
                    break;
                }
                case (OP::INFO): { // Used to tell how many lambda threads it should expect for this round.
                    unsigned arg = parse<unsigned>((char *)header.data(), 1);
                    std::cout << "Calling setNumLambdas " << std::endl;
                    setNumLambdas(lambdasocket, identity, arg);
                    break;
                }
                case (OP::TERM): {
                    std::cout << "Calling terminateServer " << std::endl;
                    terminateServer(lambdasocket, identity);
                    break;
                }
                default: {
                    std::cout << "Unknown op " << op << std::endl;
                    break;  /** Not an op that I care about. */
                }
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}

/**
 *
 * Listen on lambda threads' requests.
 *
 */
void ServerWorker::work() {
    // std::cout << "[ Weight ] Starts listening for lambdas' requests..." << std::endl;
    try {
        while (true) {
            zmq::message_t identity;
            zmq::message_t header;
            workersocket.recv(&identity);
            workersocket.recv(&header);

            OP op = parse<OP>((char *) header.data(), 0);
            std::cout << "Got OP of " << op << std::endl;

            switch (op) {
                case (OP::PUSH): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    std::cout << "Calling recvTensors " << std::endl;
                    recvTensors(workersocket, identity, chunk);
                    break;
                }
                case (OP::PULL): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    std::cout << "Calling sendTensors " << std::endl;
                    sendTensors(workersocket, identity, chunk);
                    break;
                }
                case (OP::EVAL): {
                    Chunk chunk;
                    memcpy(&chunk, (char *)header.data() + sizeof(OP), sizeof(Chunk));
                    std::cout << "Calling recvEvalData " << std::endl;
                    recvEvalData(workersocket, identity, chunk);
                    break;
                }
                case (OP::INFO): { // Used to tell how many lambda threads it should expect for this round.
                    unsigned arg = parse<unsigned>((char *)header.data(), 1);
                    std::cout << "Calling setNumLambdas " << std::endl;
                    setNumLambdas(workersocket, identity, arg);
                    break;
                }
                case (OP::TERM): {
                    std::cout << "Calling terminateServer " << std::endl;
                    terminateServer(workersocket, identity);
                    break;
                }
                default: {
                    std::cout << "Unknown op " << op << std::endl;
                    break;  /** Not an op that I care about. */
                }
            }
        }
    } catch (std::exception& ex) { /** Context Termintated. */ }
}


void ServerWorker::sendTensors(zmq::socket_t& socket, zmq::message_t& client_id, Chunk &chunk) {
    if (ws.BLOCK && chunk.dir == PROP_TYPE::FORWARD && chunk.epoch * 2 > ws.epoch) {
        while (chunk.epoch * 2 > ws.epoch) {
            usleep(50 * 1000); // sleep 50ms
        }
    }

    unsigned more = 1;
    unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
    // unsigned featLayer = chunk.layer;
    WeightTensorMap& weights = ws.weightsStore[featLayer];

    while (more) {
        // std::cout << "Listening for incoming header" << std::endl;
        zmq::message_t tensorHeader;
        socket.recv(&tensorHeader);
        // std::cout << "Got tensorHeader of size " << tensorHeader.size() << std::endl;

        std::string name = std::string(static_cast<char*>(tensorHeader.data()), tensorHeader.size());
        // std::cout << "Got name of " << name << std::endl;

        auto found = weights.find(name);
        if (found == weights.end()) {
            std::cerr << "Requested tensor '" << name << "' not found" << std::endl;
            zmq::message_t errorHeader(TENSOR_HDR_SIZE);
            populateHeader(errorHeader.data(), ERR_HEADER_FIELD, name.c_str());
            socket.send(errorHeader);
            return;
        } else {
            Matrix& reqMatrix = found->second.getMat(chunk);
            // std::cout << "Calling sendTensor" << std::endl;
            sendTensor(socket, reqMatrix, more);
        }
    }
}

void ServerWorker::recvTensors(zmq::socket_t& socket, zmq::message_t& client_id, Chunk &chunk) {
    unsigned more = 1;
    unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
    // unsigned featLayer = chunk.layer;
    WeightTensorMap& weights = ws.weightsStore[featLayer];
    while (more) {
        recvUpdateTensor(socket, chunk, weights);

        size_t usize = sizeof(more);
        socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    }
}

void ServerWorker::recvEvalData(zmq::socket_t& socket, zmq::message_t& client_id, Chunk &chunk) {
    zmq::message_t evalMsg(2 * sizeof(float));
    socket.recv(&evalMsg);

    float acc = *((float *)evalMsg.data());
    float loss = *(((float *)evalMsg.data()) + 1);

    ws.updateLocalAccLoss(chunk, acc, loss);
}

void ServerWorker::sendTensor(zmq::socket_t& socket, Matrix& tensor, unsigned& more) {
    zmq::message_t responseHeader(TENSOR_HDR_SIZE);
    populateHeader(responseHeader.data(), OP::PULL, tensor.name().c_str(),
      tensor.getRows(), tensor.getCols());
    unsigned bufSize = tensor.getRows() * tensor.getCols() * sizeof(FeatType);
    zmq::message_t tensorData(tensor.getData(), bufSize, nofree, NULL);

    // std::cout << "Sending tensor response header of size " << responseHeader.size() << std::endl;
    socket.send(responseHeader, ZMQ_SNDMORE);

    size_t usize = sizeof(unsigned);
    socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
    // std::cout << "Sending tensor data of size " << tensorData.size() << std::endl;

    if (!more) {
        socket.send(tensorData);
    } else {
        socket.send(tensorData, ZMQ_SNDMORE);
    }
}

void ServerWorker::recvUpdateTensor(zmq::socket_t& socket, Chunk &chunk, WeightTensorMap& weights) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    socket.recv(&tensorHeader);
    socket.recv(&tensorData);

    std::string name = parseName((char*)tensorHeader.data());
    auto found = weights.find(name);
    if (found == weights.end()) {
        std::cerr << "Pushed tensor '" << name
          << "' not found. Make sure to allocate it before starting workers!" << std::endl;
    } else {
        found->second.decRef(chunk);
        FeatType* newUpdate = (FeatType*) tensorData.data();
        unsigned featLayer = chunk.vertex ? chunk.layer : chunk.layer - 1;
        // unsigned featLayer = chunk.layer;
        unsigned localUpdCnt = ws.weightsStore[featLayer][name].localUpdate(newUpdate);

        if (ws.weightsStore[featLayer][name].localUpdTot == localUpdCnt) {
            ws.applyUpdate(featLayer, name);
        }
    }
}

/**
 *
 * Update the weightserver with number of lambdas being called for this iteration.
 * Therefore it knows when to average.
 *
 */
void
ServerWorker::setNumLambdas(zmq::socket_t& socket, zmq::message_t& client_id, unsigned numLambdas) {
    // Send confirm ACK message.
    zmq::message_t confirm;
    socket.send(client_id, ZMQ_SNDMORE);
    socket.send(confirm);

    ws.setLocalUpdTot(numLambdas);
    ws.clearAccLoss();
    std::cout << "[  INFO  ] Number of lambdas set to " << numLambdas << "." << std::endl;
}


/**
 *
 * After receiving the termination message from the graph server alert
 * the main thread that it can shutdown.
 *
 */
void
ServerWorker::terminateServer(zmq::socket_t& socket, zmq::message_t& client_id) {
    std::cerr << "[SHUTDOWN] Server shutting down..." << std::endl;

    std::lock_guard<std::mutex> lk(ws.termMtx);
    ws.term = true;
    ws.termCV.notify_one();
}
