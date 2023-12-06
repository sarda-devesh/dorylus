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

#define IDENTITY_SIZE (sizeof(Chunk) + sizeof(unsigned))
#define TIMEOUT_PERIOD 100

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

int
main(int argc, char *argv[]) {
    // Start socket
    Chunk chunk;
    zmq::context_t ctx(1);
    std::vector<char> identity = constructIdentity(chunk);
    zmq::socket_t socket(ctx, ZMQ_REP);
    socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());

    // Listen for incoming requests
    char port[50];
    std::cout << "Connecting to tcp://*:55400" << std::endl;
    sprintf(port, "tcp://*:55400");
    socket.bind(port);

    // Listen for incoming message
    std::cout << "Waiting for incoming data " << std::endl;
    zmq::message_t incoming(strlen("Hello World!"));
    if(socket.recv(&incoming) == -1) {
        std::cerr << "Error reading data: " << zmq_strerror(errno) << std::endl;
    }
    
    std::string data_to_send = std::string(static_cast<char*>(incoming.data()), incoming.size());
    std::cout << "Got message of " << data_to_send << std::endl;

    // Send the response back
    zmq::message_t response(data_to_send.size());
    std::memcpy(response.data(), data_to_send.data(), data_to_send.size());
    if (socket.send(response) == -1) {
        std::cerr << "Error sending data: " << zmq_strerror(errno) << std::endl;
    }

    return 0;
}