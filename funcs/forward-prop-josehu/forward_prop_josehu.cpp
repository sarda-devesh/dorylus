#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cblas.h>
#include <zmq.hpp>
#include <aws/lambda-runtime/runtime.h>
#include "../../src/utils/utils.hpp"


#define SND_MORE true
#define NO_MORE false


using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using namespace aws::lambda_runtime;
using namespace std::chrono;


/**
 *
 * Request the input matrix data from dataserver.
 * 
 */
static Matrix
requestFeatsMatrix(zmq::socket_t& socket, unsigned id) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_FORWARD, id);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if (layerResp == -1) {      // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrix data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(FeatType));
        socket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}


/**
 *
 * Request the input matrix data from weightserver.
 * 
 */
static Matrix
requestWeightsMatrix(zmq::socket_t& socket, unsigned layer) {
    
    // Send pull request.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PULL_FORWARD, layer);
    socket.send(header);

    // Listen on respond.
    zmq::message_t respHeader;
    socket.recv(&respHeader);

    // Parse the respond.
    unsigned layerResp = parse<unsigned>((char *) respHeader.data(), 1);
    if (layerResp == ERR_HEADER_FIELD) {    // Failed.
        std::cerr << "[ ERROR ] No corresponding matrix chunk!" << std::endl;
        return Matrix();
    } else {                    // Get matrices data.
        unsigned rows = parse<unsigned>((char *) respHeader.data(), 2);
        unsigned cols = parse<unsigned>((char *) respHeader.data(), 3);
        zmq::message_t matxData(rows * cols * sizeof(float));
        socket.recv(&matxData);

        FeatType *matxBuffer = new FeatType[rows * cols];
        std::memcpy(matxBuffer, matxData.data(), matxData.size());

        Matrix m(rows, cols, matxBuffer);
        return m;
    }
}


/**
 *
 * Send multiplied matrix result back to dataserver.
 * 
 */
static void
sendMatrices(Matrix& zResult, Matrix& actResult, zmq::socket_t& socket, unsigned id) {
    
    // Send push header.
    zmq::message_t header(HEADER_SIZE);
    populateHeader((char *) header.data(), OP::PUSH_FORWARD, id, zResult.getRows(), zResult.getCols());
    socket.send(header, ZMQ_SNDMORE);

    // Send zData and actData.
    zmq::message_t zData(zResult.getDataSize());
    std::memcpy(zData.data(), zResult.getData(), zResult.getDataSize());
    zmq::message_t actData(actResult.getDataSize());
    std::memcpy(actData.data(), actResult.getData(), actResult.getDataSize());
    socket.send(zData, ZMQ_SNDMORE);
    socket.send(actData);

    // Wait for data settled reply.
    zmq::message_t confirm;
    socket.recv(&confirm);
}


/**
 *
 * Matrix multiplication function.
 * 
 */
static Matrix
dot(Matrix& features, Matrix& weights) {
    unsigned m = features.getRows(), k = features.getCols(), n = weights.getCols();
    assert(k == weights.getRows());

    FeatType *res = new FeatType[m * n];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                features.getData(), k, weights.getData(), n, 0.0, res, n);

    return Matrix(m, n, res);
}


/**
 *
 * Apply activation function on a matrix.
 * 
 */
static Matrix
activate(Matrix& mat) {
    FeatType *activationData = new FeatType[mat.getNumElemts()];
    FeatType *zData = mat.getData();
    
    for (unsigned i = 0; i < mat.getNumElemts(); ++i)
        activationData[i] = std::tanh(zData[i]);

    return Matrix(mat.getRows(), mat.getCols(), activationData);
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
 * 
 */
static invocation_response
forward_prop_layer(std::string dataserver, std::string weightserver, std::string dport, std::string wport,
                   unsigned id, unsigned layer) {
    zmq::context_t ctx(1);

    //
    // Lambda socket identity is set to:
    //
    //      [ 4 Bytes partId ] | [ n Bytes the string of dataserverIp ]
    //
    // One should extract the partition id by reading the first 4 Bytes, which is simply parse<unsigned>(...).
    //
    size_t identity_len = sizeof(unsigned) + dataserver.length();
    char identity[identity_len];
    memcpy(identity, (char *) &id, sizeof(unsigned));
    memcpy(identity + sizeof(unsigned), (char *) dataserver.c_str(), dataserver.length());

    Timer getWeightsTimer;
    Timer getFeatsTimer;
    Timer computationTimer;
    Timer activationTimer;
    Timer sendResTimer;

    try {
        Matrix weights, feats, z, activations;

        zmq::socket_t weights_socket(ctx, ZMQ_DEALER);
        weights_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char whost_port[50];
        sprintf(whost_port, "tcp://%s:%s", weightserver.c_str(), wport.c_str());
        weights_socket.connect(whost_port);

        zmq::socket_t data_socket(ctx, ZMQ_DEALER);
        data_socket.setsockopt(ZMQ_IDENTITY, identity, identity_len);
        char dhost_port[50];
        sprintf(dhost_port, "tcp://%s:%s", dataserver.c_str(), dport.c_str());
        data_socket.connect(dhost_port);
        
        // Request weights matrix of the current layer.
        std::thread t([&] {     // Weight requests run in a separate thread.
            std::cout << "< FORWARD > Asking weightserver..." << std::endl;
            getWeightsTimer.start();
            weights = requestWeightsMatrix(weights_socket, layer);
            getWeightsTimer.stop();
            std::cout << "< FORWARD > Got data from weightserver." << std::endl;
        });

        // Request feature activation matrix of the current layer.
        std::cout << "< FORWARD > Asking dataserver..." << std::endl;
        getFeatsTimer.start();
        feats = requestFeatsMatrix(data_socket, id);
        getFeatsTimer.stop();
        std::cout << "< FORWARD > Got data from dataserver." << std::endl;

        t.join();

        if (weights.empty())
            return invocation_response::failure("Weights could not be loaded", "application/json");
        if (feats.empty())
            return invocation_response::failure("No chunk corresponding to request", "appliation/json");


        // Multiplication.
        computationTimer.start();
        std::cout << "< FORWARD > Doing the dot multiplication..." << std::endl;
        z = dot(feats, weights);
        computationTimer.stop();

        // Activation.
        activationTimer.start();
        std::cout << "< FORWARD > Doing the activation..." << std::endl;
        activations = activate(z);
        activationTimer.stop();

        // Send back to dataserver.
        sendResTimer.start();
        std::cout << "< FORWARD > Sending results back..." << std::endl;
        sendMatrices(z, activations, data_socket, id);
        std::cout << "< FORWARD > Results sent." << std::endl;
        sendResTimer.stop();

        // Delete malloced spaces.
        delete[] weights.getData();
        delete[] feats.getData();
        delete[] z.getData();
        delete[] activations.getData();

    } catch(std::exception &ex) {
        return invocation_response::failure(ex.what(), "application/json");
    }

    // Couldn't parse JSON with AWS SDK from ptree.
    // For now creating a string with the times to be parsed on server.
    std::string res = std::to_string(id) + ": " +
                      std::to_string(getWeightsTimer.getTime()) + " " +     \
                      std::to_string(getFeatsTimer.getTime())  + " " +      \
                      std::to_string(computationTimer.getTime()) + " " +    \
                      std::to_string(activationTimer.getTime()) + " " +     \
                      std::to_string(sendResTimer.getTime());

    return invocation_response::success(res, "application/json");
}


/** Handler that hooks with lambda API. */
static invocation_response
my_handler(invocation_request const& request) {
    ptree pt;
    std::istringstream is(request.payload);
    read_json(is, pt);

    std::string dataserver = pt.get<std::string>("dataserver");
    std::string weightserver = pt.get<std::string>("weightserver");
    std::string dport = pt.get<std::string>("dport");
    std::string wport = pt.get<std::string>("wport");
    unsigned layer = pt.get<int>("layer");
    unsigned chunkId = pt.get<int>("id");

    std::cout << "[ACCEPTED] Thread " << chunkId << " is requested from " << dataserver << ":" << dport
              << ", FORWARD layer " << layer << "." << std::endl;

    return forward_prop_layer(dataserver, weightserver, dport, wport, chunkId, layer);
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);
    
    return 0;
}
