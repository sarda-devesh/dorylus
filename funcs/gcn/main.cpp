#include <algorithm>
#include <chrono>
#include <ratio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <random>

#include <cblas.h>
#include <zmq.hpp>

#include <aws/lambda-runtime/runtime.h>
#include <aws/core/Aws.h>
#include <aws/core/utils/json/JsonSerializer.h>

#include "utils.hpp"
#include "../../common/matrix.hpp"
#include "../../common/utils.hpp"

#include "ops/forward_ops.hpp"
#include "ops/backward_ops.hpp"
#include "ops/network_ops.hpp"


using namespace Aws::Utils::Json;
using namespace aws::lambda_runtime;
using namespace std::chrono;

#define IDENTITY_SIZE (sizeof(Chunk) + sizeof(unsigned))
std::vector<char> constructIdentity(Chunk &chunk) {
    std::vector<char> identity(IDENTITY_SIZE);

    std::random_device rd;
    std::mt19937 generator(rd());

    unsigned rand = generator();
    std::cout << "RAND " << rand << std::endl;
    std::memcpy(identity.data(), &chunk, sizeof(chunk));
    std::memcpy(identity.data() + sizeof(chunk), &rand, sizeof(unsigned));

    return identity;
}

invocation_response
finalLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk,
           bool eval, unsigned trainset_size) {
    std::cout << "FINAL LAYER" << std::endl;
    std::vector<std::string> dataRequests{"ah", "lab"};
    std::vector<std::string> weightRequests{"w"};

    std::cout << "Request ah and lab" << std::endl;
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests, false);
    for (auto& M : matrices) {
        if (M.empty()){
            for (auto& M : matrices) deleteMatrix(M);

            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        }
    }

    std::cout << "Request w" << std::endl;
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests, true);
    for (auto& W : weights) {
        if (W.empty()){
            for (auto& M : matrices) deleteMatrix(M);
            for (auto& W : weights) deleteMatrix(W);

            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, W.name() + " is empty");
        }
    }

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.localId, "Got error message from server");
    }
    std::cout << "Fin Request" << std::endl;


    // Forward layer
    Matrix& AH = matrices[0];
    Matrix& W = weights[0];

    Matrix Z = AH.dot(W);

    Matrix preds = softmax(Z);
    deleteMatrix(Z);
    Matrix& labels = matrices[1];

    if (eval) {
        sendAccLoss(data_socket, weights_socket, preds, labels, chunk);
    }

    maskout(preds, labels);

    // Backward computation
    Matrix d_out = preds - labels;
    d_out /= trainset_size;
    deleteMatrix(labels);
    deleteMatrix(preds);

    Matrix interGrad = d_out.dot(W, false, true);
    deleteMatrix(W);
    Matrix d_weights = AH.dot(d_out, true, false);
    deleteMatrix(AH);
    deleteMatrix(d_out);

    std::cout << "Send interGrad" << std::endl;
    interGrad.setName("grad");
    std::vector<Matrix> toSend{interGrad};
    int ret = sendTensors(data_socket, chunk, toSend, true);
    // Clean up data
    for (auto& M : toSend)
        deleteMatrix(M);
    if (ret == -1) {
        return constructResp(false, chunk.localId, "This chunk is already done");
    }

    std::cout << "Send w" << std::endl;
    d_weights.setName("w");
    std::vector<Matrix> weightUpdates{d_weights};
    sendTensors(weights_socket, chunk, weightUpdates);
    std::cout << "Fin send" << std::endl;

    for (auto& M : weightUpdates)
        deleteMatrix(M);
    std::cout << "Data cleaned up" << std::endl;

    return constructResp(true, chunk.localId, "Finished final layer");
}

invocation_response
backwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "BACKWARD LAYER" << std::endl;
    std::cout << "Request ah z and aTg" << std::endl;
    std::vector<std::string> dataReqs{"ah", "z", "aTg"};
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataReqs, false);
    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        } else {
            std::cout << "GOT " << M.name() << std::endl;
        }
    }

    std::cout << "Request w" << std::endl;
    std::vector<std::string> weightReqs{"w"};
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightReqs, true);
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, W.name() + " is empty");
        } else {
            std::cout << "GOT " << W.name() << std::endl;
        }
    }

    if (matrices.empty() || weights.empty()) {
        return constructResp(false, chunk.localId, "Got error message from server");
    }
    std::cerr << "Fin Request" << std::endl;

    Matrix& AH = matrices[0];
    Matrix& Z = matrices[1];
    Matrix& grad = matrices[2];

    Matrix& W = weights[0];

    Matrix actDeriv = tanhDerivative(Z);
    deleteMatrix(Z);

    Matrix interGrad = grad * actDeriv;
    deleteMatrix(grad);
    deleteMatrix(actDeriv);

    Matrix resultGrad = interGrad.dot(W, false, true);
    deleteMatrix(W);

    Matrix d_weights = AH.dot(interGrad, true, false);
    deleteMatrix(AH);
    deleteMatrix(interGrad);

    std::cout << "Send resultGrad" << std::endl;
    resultGrad.setName("grad");
    std::vector<Matrix> toSend{resultGrad};
    int ret = 0;
    if (chunk.layer > 0) { // not the last layer
        ret = sendTensors(data_socket, chunk, toSend, true);
    } else { // the last backward layer (layer 0), skip sending the grad back
        ret = sendFinMsg(data_socket, chunk);
    }
    std::cout << "Fin send" << std::endl;
    for (auto& M : toSend)
        deleteMatrix(M);
    if (ret == -1) {
        return constructResp(false, chunk.localId, "This chunk is already done");
    }

    std::cout << "Send wu" << std::endl;
    d_weights.setName("w");
    std::vector<Matrix> weightUpdates{d_weights};
    sendTensors(weights_socket, chunk, weightUpdates);

    for (auto& M : weightUpdates)
        deleteMatrix(M);

    return constructResp(true, chunk.localId, "Finished backward layer");
}

invocation_response
forwardLayer(zmq::socket_t& data_socket, zmq::socket_t& weights_socket, Chunk &chunk) {
    std::cout << "FORWARD LAYER" << std::endl;

    std::vector<std::string> dataRequests{"ah"};
    std::cerr << "Request AH" << std::endl;
    std::vector<Matrix> matrices = reqTensors(data_socket, chunk, dataRequests, false);
    std::cout << "AH matrix has count " << matrices.size() << std::endl;
    for (auto& M : matrices) {
        if (M.empty()){
            std::cout << M.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, M.name() + " is empty");
        }
    }

    std::vector<std::string> weightRequests{"w"};
    std::cerr << "Request w" << std::endl;
    std::vector<Matrix> weights = reqTensors(weights_socket, chunk, weightRequests, true);
     std::cout << "W matrix has count " << matrices.size() << std::endl;
    for (auto& W : weights) {
        if (W.empty()){
            std::cout << W.name() << " is empty" << std::endl;
            return constructResp(false, chunk.localId, W.name() + " is empty");
        }
    }

    if (matrices.empty() || weights.empty()) {
        std::cout << "Found an empty matrix" << std::endl;
        return constructResp(false, chunk.localId, "Got error message from server");
    }
    std::cerr << "Fin Request" << std::endl;

    std::cout << "Getting AH matrix" << std::endl;
    Matrix& AH = matrices[0];
    std::cout << "Getting W matrix" << std::endl;
    Matrix& W = weights[0];

    std::cout << "Performing dot product" << std::endl;
    Matrix Z = AH.dot(W);
    std::cout << "Setting name to z" << std::endl;
    Z.setName("z");

    std::cout << "deleting AH matrix" << std::endl;
    deleteMatrix(AH);
    std::cout << "deleting W" << std::endl;
    deleteMatrix(W);

    std::cout << "Performing tanh(Z)" << std::endl;
    Matrix H_l = tanh(Z);
    std::cout << "Set name to h" << std::endl;
    H_l.setName("h");

    std::vector<Matrix> toSend;
    toSend.push_back(Z);
    toSend.push_back(H_l);

    std::cout << "Sending tensors Z, H" << std::endl;
    int ret = sendTensors(data_socket, chunk, toSend, true);
    std::cout << "Sent Z and H vectors" << std::endl;
    
    // Clean up data
    for (auto& M : toSend)
        deleteMatrix(M);
    std::cout << "Data cleaned up" << std::endl;
    if (ret == -1) {
        return constructResp(false, chunk.localId, "This chunk is already done.");
    } else {
        return constructResp(true, chunk.localId, "Finished forward layer");
    }
}


/**
 *
 * Main logic:
 *
 *      1. Querying matrix data from dataserver;
 *      2. Querying weight matrix from weightserver;
 *      3. Conduct the matrix multiplication to get Z matrix;
 *      4. Perform activation on Z to get Activated matrix;
 *      5. Send both matrices back to data server.
 *      6. If evaluate is true, check the model precision
 *
 */
invocation_response
apply_phase(std::string dataserver, std::string weightserver, unsigned dport, unsigned wport,
            Chunk &chunk, bool eval, unsigned trainset_size) {
    zmq::context_t ctx(1);

    // Creating identity
    std::vector<char> identity = constructIdentity(chunk);

    zmq::socket_t weights_socket(ctx, ZMQ_REQ);
    zmq::socket_t data_socket(ctx, ZMQ_REQ);
    try {
        weights_socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());
        if (RESEND) {
            weights_socket.setsockopt(ZMQ_RCVTIMEO, TIMEOUT_PERIOD);
        }
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%u", weightserver.c_str(), 9000);
        std::cout << "Connectiong to host port " << whost_port << std::endl;
        weights_socket.connect(whost_port);

        data_socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());
        if (RESEND) {
            data_socket.setsockopt(ZMQ_RCVTIMEO, TIMEOUT_PERIOD);
        }
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%u", dataserver.c_str(), 8000);
        std::cout << "Connectiong to data port " << dhost_port << std::endl;
        data_socket.connect(dhost_port);
    } catch(std::exception& ex) {
        return constructResp(false, chunk.localId, ex.what());
    }

    if (chunk.dir == PROP_TYPE::FORWARD && chunk.layer < 1) {
        std::cout << "Running forward proporgation " << std::endl;
        return forwardLayer(data_socket, weights_socket, chunk);
    } else if (chunk.dir == PROP_TYPE::FORWARD && chunk.layer == 1) {
        std::cout << "Running finalLayer " << std::endl;
        return finalLayer(data_socket, weights_socket, chunk, eval, trainset_size);
    } else if (chunk.dir == PROP_TYPE::BACKWARD) {
        std::cout << "Running backwardLayer " << std::endl;
        return backwardLayer(data_socket, weights_socket, chunk);
    }

    std::cout << "Returning from function" << std::endl;

    weights_socket.setsockopt(ZMQ_LINGER, 0);
    weights_socket.close();
    data_socket.setsockopt(ZMQ_LINGER, 0);
    data_socket.close();
    ctx.close();

    return constructResp(false, chunk.localId, "Didn't run any config");
}


/** Handler that hooks with lambda API. */
invocation_response
my_handler(invocation_request const& request) {
    JsonValue json(request.payload);
    auto v = json.View();

    std::string dataserver = v.GetString("dserver");
    std::string weightserver = v.GetString("wserver");
    unsigned dport = v.GetInteger("dport");
    unsigned wport = v.GetInteger("wport");
    bool eval = v.GetBool("eval");
    unsigned trainset_size = v.GetInteger("trainset_size");

    Chunk chunk;
    chunk.localId = v.GetInteger("id");
    chunk.globalId = v.GetInteger("gid");
    chunk.lowBound = v.GetInteger("lb");
    chunk.upBound = v.GetInteger("ub");
    chunk.layer = v.GetInteger("layer");
    chunk.dir = static_cast<PROP_TYPE>(v.GetInteger("dir"));
    chunk.epoch = v.GetInteger("epoch");
    chunk.vertex = v.GetInteger("vtx");

    std::cout << "[ACCEPTED] Thread " << chunk.str() << " is requested from "
              << dataserver << ":" << dport << ", FORWARD layer " << chunk.layer
              << "." << std::endl;

    return apply_phase(dataserver, weightserver, dport, wport, chunk, eval, trainset_size);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}
