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

/** Handler that hooks with lambda API. */
invocation_response
my_handler(invocation_request const& request) {
    // Start socket
    Chunk chunk;
    zmq::context_t ctx(1);
    std::vector<char> identity = constructIdentity(chunk);
    zmq::socket_t socket(ctx, ZMQ_REQ);
    socket.setsockopt(ZMQ_IDENTITY, identity.data(), identity.size());

    // Connect to server
    char port[50];
    std::cout << "Connecting to tcp://172.31.21.103:55400" << std::endl;
    sprintf(port, "tcp://172.31.21.103:55400");
    socket.connect(port);

    // Send some data
    std::string data_to_send = "Hello World!";
    zmq::message_t message(data_to_send.size());
    std::memcpy(message.data(), data_to_send.data(), data_to_send.size());
    if (socket.send(message) == -1) {
        std::cerr << "Error sending data: " << zmq_strerror(errno) << std::endl;
    }

    // Ensure that it is echoed back
    zmq::message_t response(data_to_send.size());
    std::cout << "Calling recv" << std::endl;
    if (socket.recv(&response) == -1) {
        std::cerr << "Error reading data: " << zmq_strerror(errno) << std::endl;
    }
    
    std::string output = std::string(static_cast<char*>(response.data()), response.size());
    std::cout << "Text returned by recv is " << output << std::endl;
}

int
main(int argc, char *argv[]) {
    run_handler(my_handler);

    return 0;
}