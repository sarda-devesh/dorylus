#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "../utils/utils.hpp"


/**
 *
 * Wrapper over a server worker thread.
 * 
 */
class ServerWorker {

public:

    ServerWorker(zmq::context_t& ctx_, int sock_type, std::vector<Matrix>& _weights)
        : ctx(ctx_), worker(ctx, sock_type), weight_list(_weights) { }

    // Listens on lambda threads' request for weights.
    void work() {
        worker.connect("inproc://backend");

        std::cout << "[Weight] Starts listening for lambdas' requests..." << std::endl;
        try {
            while (true) {
                zmq::message_t identity;
                zmq::message_t header;
                worker.recv(&identity);
                worker.recv(&header);
                
                int32_t chunkId = parse<int32_t>((char *) identity.data(), 0);
                int32_t op = parse<int32_t>((char *) header.data(), 0);
                int32_t layer = parse<int32_t>((char *) header.data(), 1);

                std::string opStr = op == 0 ? "Push" : "Pull";
                std::string accMsg = "[ACCEPTED] " + opStr + " from thread "
                                   + std::to_string(chunkId) + " for layer "
                                   + std::to_string(layer);
                std::cout << accMsg << std::endl;

                switch (op) {
                    case (OP::PULL):
                        sendWeights(worker, identity, layer);
                        break;
                    case (OP::PUSH):
                        recvUpdates(identity, layer, header);
                        break;
                    default:
                        std::cerr << "ServerWorker: Unknown Op code received." << std::endl;
                }
            }
        } catch (std::exception& ex) {
            std::cerr << ex.what() << std::endl;
        }
    }

private:

    void sendWeights(zmq::socket_t& socket, zmq::message_t& client_id, int32_t layer) {
        Matrix& weights = weight_list[layer];
        zmq::message_t header(HEADER_SIZE);
        populateHeader((char *) header.data(), OP::RESP, 0, weights.rows, weights.cols);
        
        zmq::message_t weightData(weights.getDataSize());
        std::memcpy((char *) weightData.data(), weights.getData(), weights.getDataSize());
        
        // The identity message will be implicitly consumed to route the message to the correct client.
        socket.send(client_id, ZMQ_SNDMORE);
        socket.send(header, ZMQ_SNDMORE);
        socket.send(weightData);
    }

    void recvUpdates(zmq::message_t& client_id, int32_t layer, zmq::message_t& header) {
        // TODO: Receive updates from threads.
    }

    std::vector<Matrix>& weight_list;
    zmq::context_t &ctx;
    zmq::socket_t worker;
};


/**
 *
 * Class of the weightserver. Weightservers are only responsible for replying weight requests from lambdas,
 * and possibly handle weight updates.
 * 
 */
class WeightServer {

public:
    
    WeightServer(unsigned _port, std::string& configFileName)
        : ctx(1), frontend(ctx, ZMQ_ROUTER), backend(ctx, ZMQ_DEALER), port(_port) {

        // Read in layer configurations.
        initializeWeightMatrices(configFileName);

        // Currently using randomly generated weights.
        auto seed = 8888;
        std::default_random_engine dre(seed);
        std::uniform_real_distribution<FeatType> dist(-1.5, 1.5);

        for (uint32_t u = 0; u < dims.size() - 1; ++u) {
            uint32_t dataSize = dims[u] * dims[u + 1];
            FeatType *dptr = new FeatType[dataSize];
            for (uint32_t ui = 0; ui < dataSize; ++ui)
                dptr[ui] = dist(dre);

            layers.push_back(Matrix(dims[u], dims[u + 1], dptr));
        }

        for (uint32_t u = 0; u < layers.size(); ++u)
            fprintf(stdout, "Layer %u Weights: %s\n", u, layers[u].str().c_str());
    }

    // Defines how many concurrent weightserver threads to use.
    enum { kMaxThreads = 2 };

    // Runs the weightserver, start a bunch of worker threads and create a proxy through frontend to
    // backend.
    void run() {
        char host_port[50];
        sprintf(host_port, "tcp://*:%u", port);
        std::cout << "Binding weight server to " << host_port << "..." << std::endl;
        frontend.bind(host_port);
        backend.bind("inproc://backend");

        std::vector<ServerWorker *> workers;
        std::vector<std::thread *> worker_threads;
        for (int i = 0; i < kMaxThreads; ++i) {
            workers.push_back(new ServerWorker(ctx, ZMQ_DEALER, layers));

            worker_threads.push_back(new std::thread(std::bind(&ServerWorker::work, workers[i])));
            worker_threads[i]->detach();
        }

        try {
            zmq::proxy(static_cast<void *>(frontend), static_cast<void *>(backend), nullptr);
        } catch (std::exception& ex) {
            std::cerr << "[ERROR] " << ex.what() << std::endl;
        }

        for (int i = 0; i < kMaxThreads; ++i) {
            delete worker_threads[i];
            delete workers[i];
        }
    }

private:

    // Read in layer configurations.
    void initializeWeightMatrices(std::string& configFileName) {

        std::ifstream infile(configFileName.c_str());
        if (!infile.good())
            fprintf(stderr, "[ERROR] Cannot open layer configuration file: %s [Reason: %s]\n", configFileName.c_str(), std::strerror(errno));

        assert(infile.good());

        // Loop through each line.
        std::string line;
        while (!infile.eof()) {
            std::getline(infile, line);
            boost::algorithm::trim(line);

            if (line.length() > 0)
                dims.push_back(std::stoul(line));
        }

        assert(dims.size() > 1);
    }

    std::vector<uint32_t> dims;
    std::vector<Matrix> layers;

    zmq::context_t ctx;
    zmq::socket_t frontend;
    zmq::socket_t backend;
    unsigned port;
};


/** Main entrance: Starts a weightserver instance and run. */
int
main(int argc, char *argv[]) {
    assert(argc == 3);
    int32_t weightserverPort = std::atoi(argv[1]);
    std::string configFileName = argv[2];

    WeightServer ws(weightserverPort, configFileName);
    ws.run();
    
    return 0;
}
