#include "network_ops.hpp"
#include <cmath>

int recvTensor(zmq::socket_t& socket, Matrix &mat) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    std::cout << "Calling recv for tensor header of size " << TENSOR_HDR_SIZE << std::endl;
    if (!socket.recv(&tensorHeader)) {
        return 0;
    }
    std::cout << "Received tensor header" << std::endl;

    unsigned resp = parse<unsigned>((char*)tensorHeader.data(), 0);
    if (resp == ERR_HEADER_FIELD) {
        std::cerr << "Got error from server. Consult graph server output" << std::endl;
        return -1;
    }
    std::string name = parseName((char*)tensorHeader.data());

    std::cout << "Calling receive for tensor data" << std::endl;
    if (!socket.recv(&tensorData)) {
        return 0;
    }
    std::cout << "Received tensor data" << std::endl;

    unsigned rows = parse<unsigned>((char*)tensorHeader.data(), 3);
    unsigned cols = parse<unsigned>((char*)tensorHeader.data(), 4);

    FeatType* data = new FeatType[rows * cols];
    std::memcpy(data, tensorData.data(), tensorData.size());

    mat.setName(name.c_str());
    mat.setRows(rows);
    mat.setCols(cols);
    mat.setData(data);

    return 0;
}

std::vector<Matrix> reqTensors(zmq::socket_t& socket, Chunk &chunk,
                        std::vector<std::string>& tensorRequests) {

#define INIT_PERIOD (5 * 1000u) // 5ms
#define MAX_PERIOD (500 * 1000u)
#define EXP_FACTOR 1.5

    unsigned sleepPeriod = INIT_PERIOD;

    bool empty = true;
    std::vector<Matrix> matrices;
    while (true) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::PULL, chunk);
        std::cout << "Sending header of size " << HEADER_SIZE << std::endl;
        socket.send(header, ZMQ_SNDMORE);
        std::cout << "Socket sent header" << std::endl;

        unsigned numTensors = tensorRequests.size();
        std::cout << "Got numTensors of " << numTensors << std::endl;

        for (unsigned u = 0; u < tensorRequests.size(); ++u) {
            std::string& name = tensorRequests[u];
            zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
            populateHeader(tensorHeader.data(), chunk.localId, name.c_str());
            std::cout << "Populated tensor header of size " << TENSOR_HDR_SIZE << std::endl;

            if (u < numTensors - 1) {
                socket.send(tensorHeader, ZMQ_SNDMORE);
            } else {
                socket.send(tensorHeader);
            }
            std::cout << "Sent tensor header of " << u << std::endl;
        }

        unsigned more = 1;
        empty = false;
        while (more && !empty) {
            Matrix result;
            std::cout << "Calling recv tensor" << std::endl;
            int ret = recvTensor(socket, result);
            std::cout << "recvTensor returned val of " << ret << std::endl;

            if (ret == -1) {
                for (auto& M : matrices) deleteMatrix(M);
                matrices.clear();
                return matrices;
            }
            if (result.empty()) {
                empty = result.empty();

                for (auto& M : matrices) deleteMatrix(M);
                matrices.clear();
                size_t usize = sizeof(more);
                socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
            } else {
                matrices.push_back(result);

                size_t usize = sizeof(more);
                socket.getsockopt(ZMQ_RCVMORE, &more, &usize);
            }
        }

        if (RESEND && empty) {
            usleep(sleepPeriod);
            sleepPeriod *= EXP_FACTOR;
            sleepPeriod = std::min(sleepPeriod, MAX_PERIOD);
        } else {
            break;
        }
    }

    return matrices;

#undef INIT_PERIOD
#undef MAX_PERIOD
#undef EXP_FACTOR
}

int sendTensors(zmq::socket_t& socket, Chunk &chunk,
            std::vector<Matrix>& matrices, bool ack) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::PUSH, chunk);
    socket.send(header, ZMQ_SNDMORE);
    for (uint32_t u = 0; u < matrices.size(); ++u) {
        std::cout << "Sending tensor " << matrices[u].name() << std::endl;
        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        populateHeader(tensorHeader.data(), OP::PUSH, matrices[u].name().c_str(), chunk.layer,
          matrices[u].getRows(), matrices[u].getCols());
        zmq::message_t tensorData(matrices[u].getDataSize());
        std::memcpy(tensorData.data(), matrices[u].getData(), matrices[u].getDataSize());

        socket.send(tensorHeader, ZMQ_SNDMORE);
        if (u < matrices.size() - 1) {
            socket.send(tensorData, ZMQ_SNDMORE);
        } else {
            socket.send(tensorData);
        }
    }

    int ret = 0;
    if (ack) {
        std::cout << "Waiting on ACK" << std::endl;
        zmq::message_t ack;
        socket.recv(&ack);
        if (ack.size() == sizeof(int) * 3) {
            ret = *(int *)ack.data();
        }
        std::cout << "Received ACK" << std::endl;
    }
    return ret;
}

/**
 *
 * Calculate batch loss and accuracy based on local forward predicts and labels.
 */
void sendAccLoss(zmq::socket_t &dsocket, zmq::socket_t &wsocket, Matrix &predicts, Matrix &labels, Chunk &chunk) {
    float acc = 0.0;
    float loss = 0.0;
    const unsigned featDim = labels.getCols();
    const unsigned valStt = (unsigned)(predicts.getRows() * TRAIN_PORTION);
    const unsigned valEnd = valStt + (unsigned)(predicts.getRows() * VAL_PORTION);
    FeatType *currLabel = labels.getData();
    FeatType *currPred = predicts.getData();
    for (unsigned i = valStt; i < valEnd; i++) {
        acc += currLabel[argmax(currPred, currPred + featDim)];
        loss -= std::log(currPred[argmax(currLabel, currLabel + featDim)]);

        currLabel += featDim;
        currPred += featDim;
    }

    // send accloss to graph server
    if (false) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::EVAL, chunk);
        zmq::message_t payload(2 * sizeof(float));
        char *bufPtr = (char *)payload.data();
        memcpy(bufPtr, &acc, sizeof(float));
        bufPtr += sizeof(float);
        memcpy(bufPtr, &loss, sizeof(float));

        dsocket.send(header, ZMQ_SNDMORE);
        dsocket.send(payload);
    }


    // send accloss to weight server
    if (true) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::EVAL, chunk);
        zmq::message_t payload(2 * sizeof(float));
        char *bufPtr = (char *)payload.data();
        memcpy(bufPtr, &acc, sizeof(float));
        bufPtr += sizeof(float);
        memcpy(bufPtr, &loss, sizeof(float));

        wsocket.send(header, ZMQ_SNDMORE);
        wsocket.send(payload);
    }
}

int sendFinMsg(zmq::socket_t& socket, Chunk &chunk) {
    zmq::message_t header(HEADER_SIZE);
    populateHeader(header.data(), OP::FIN, chunk);
    socket.send(header);

    int ret = 0;
    std::cout << "Waiting on ACK" << std::endl;
    zmq::message_t ack;
    socket.recv(&ack);
    if (ack.size() == sizeof(int) * 3) {
        ret = *(int *)ack.data();
    }
    std::cout << "Received ACK" << std::endl;

    return ret;
}
// end named-tensors
