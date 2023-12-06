#include "network_ops.hpp"
#include <cmath>

int recvTensor(zmq::socket_t& socket, Matrix &mat) {
    zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
    zmq::message_t tensorData;

    std::cout << "Listening for tensor header of size " << TENSOR_HDR_SIZE << std::endl;
    if (!socket.recv(&tensorHeader)) {
        return 0;
    }
    std::cout << "Got the tensor header" << std::endl;

    unsigned resp = parse<unsigned>((char*)tensorHeader.data(), 0);
    if (resp == ERR_HEADER_FIELD) {
        std::cerr << "Got error from server. Consult graph server output" << std::endl;
        return -1;
    }
    std::string name = parseName((char*)tensorHeader.data());

    if (!socket.recv(&tensorData)) {
        return 0;
    }

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

std::vector<Matrix> reqTensors(zmq::socket_t& socket, Chunk &chunk, std::vector<std::string>& tensorRequests, bool repeat_overall) {

#define INIT_PERIOD (5 * 1000u) // 5ms
#define MAX_PERIOD (500 * 1000u)
#define EXP_FACTOR 1.5
#define MAX_TENSOR_NAME_LEN 5

    unsigned sleepPeriod = INIT_PERIOD;

    bool empty = true;
    std::vector<Matrix> matrices;
    while (true) {
        zmq::message_t header(HEADER_SIZE);
        populateHeader(header.data(), OP::PULL, chunk);
        std::cout << "Send overall header of size " << header.size() << std::endl;
        socket.send(header, ZMQ_SNDMORE);

        if(repeat_overall) {
            zmq::message_t resend_header(HEADER_SIZE);
            populateHeader(resend_header.data(), OP::PULL, chunk);
            std::cout << "Resending overall header of size " << resend_header.size() << std::endl;
            socket.send(resend_header, ZMQ_SNDMORE);
        }
        
        unsigned numTensors = tensorRequests.size();
        for (unsigned u = 0; u < tensorRequests.size(); ++u) {
            std::cout << "Starting header for " << u << std::endl;
            std::string& name = tensorRequests[u];
            zmq::message_t tensorHeader(name.size());
            std::memcpy(tensorHeader.data(), name.data(), name.size());
            
            std::cout << "Send tensor header of size " << tensorHeader.size() << " for name " << name << std::endl;
            if (u < numTensors - 1) {
                std::cout << "Sending with ZMQ_SNDMORE" << std::endl;
                socket.send(tensorHeader, ZMQ_SNDMORE);
            } else {
                std::cout << "Sending without ZMQ_SNDMORE" << std::endl;
                socket.send(tensorHeader);
            }

            std::cout << "Send output for idx " << u << " with total " << numTensors << std::endl;
        }
        std::cout << "Finished for loop" << std::endl;

        unsigned more = 1;
        empty = false;
        std::cout << "Condition check of " << more << " and " << empty << std::endl;
        while (more && !empty) {
            std::cout << "Enter for loop" << std::endl;
            Matrix result;
            std::cout << "Calling recvTensor" << std::endl;
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
        std::cout << "Exited out of while loop" << std::endl;

        if (RESEND && empty) {
            usleep(sleepPeriod);
            sleepPeriod *= EXP_FACTOR;
            sleepPeriod = std::min(sleepPeriod, MAX_PERIOD);
            std::cout << "Running with sleep period of " << sleepPeriod << std::endl;
        } else {
            std::cout << "Break statement" << std::endl;
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
    std::cout << "sendTensors: Sending overall header" << std::endl;
    socket.send(header, ZMQ_SNDMORE);

    for (uint32_t u = 0; u < matrices.size(); ++u) {
        std::cout << "sendTensors: Sending tensor " << matrices[u].name() << std::endl;

        zmq::message_t tensorHeader(TENSOR_HDR_SIZE);
        populateHeader(tensorHeader.data(), OP::PUSH, matrices[u].name().c_str(), chunk.layer,
          matrices[u].getRows(), matrices[u].getCols());

        zmq::message_t tensorData(matrices[u].getDataSize());
        std::memcpy(tensorData.data(), matrices[u].getData(), matrices[u].getDataSize());

        std::cout << "sendTensors: Sending tensorHeader" << std::endl;
        socket.send(tensorHeader, ZMQ_SNDMORE);

        std::cout << "sendTensors: Sending tensorData" << std::endl;
        if (u < matrices.size() - 1) {
            socket.send(tensorData, ZMQ_SNDMORE);
        } else {
            socket.send(tensorData);
        }
    }

    int ret = 0;
    if (ack) {
        std::cout << "sendTensors: Waiting an ACK" << std::endl;
        zmq::message_t ack;
        socket.recv(&ack);
        if (ack.size() == sizeof(int) * 3) {
            ret = *(int *)ack.data();
        }
        std::cout << "sendTensors: Received an ACK" << std::endl;
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
