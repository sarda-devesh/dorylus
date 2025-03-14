#include "engine.hpp"
#include "../utils/utils.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of.
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>

#include "../graph/dataloader.hpp"

#ifdef _GPU_ENABLED_
#include "../commmanager/GPU_comm.hpp"
#endif
#ifdef _CPU_ENABLED_
#include "../commmanager/CPU_comm.hpp"
#endif
#ifdef _LAMBDA_ENABLED_
#include "../commmanager/lambda_comm.hpp"
#endif

/**
 *
 * Initialize the engine with the given command line arguments.
 *
 */
void Engine::init(int argc, char *argv[])
{
    printLog(404, "Engine starts initialization...");
    timeInit = -getTimer();

    parseArgs(argc, argv);

    // Initialize the node manager and communication manager.
    nodeManager.init(dshMachinesFile, myPrIpFile,
                     this); // NodeManger should go first.

    nodeId = nodeManager.getMyNodeId();
    numNodes = nodeManager.getNumNodes();
    assert(numNodes <= 256); // Cluster size limitation.
    outFile += std::to_string(nodeId);
    // Init data ctx with `dThreads` threads for scatter
    commManager.init(nodeManager, mode == LAMBDA ? dThreads : 1);

    // Set number of layers and number of features in each layer. Also store the
    // prefix sum of config for offset querying use.
    readLayerConfigFile(layerConfigFile);
    numLayers = layerConfig.size() - 1;

    std::string graphFile =
        datasetDir + "graph." + std::to_string(nodeId) + ".bin";
    // detect whether preprocessed
    {
        std::ifstream gfile(graphFile.c_str(), std::ios::binary);
        if (!gfile.good() || forcePreprocess)
        {
            DataLoader dl(datasetDir, nodeId, numNodes, undirected);
            dl.preprocess();
        }
    }
    graph.init(graphFile);
    printGraphMetrics();
    printLog(nodeId, "Print graph stats");

    for (unsigned i = 0; i < 2 * numLayers; i++)
    {
        vecTimeAggregate.push_back(0.0);
        vecTimeApplyVtx.push_back(0.0);
        vecTimeScatter.push_back(0.0);
        vecTimeApplyEdg.push_back(0.0);
        vecTimeLambdaInvoke.push_back(0.0);
        vecTimeLambdaWait.push_back(0.0);
    }

    // Save intermediate tensors during forward phase for backward computation.
    savedNNTensors.resize(numLayers + 1);
    savedEdgeTensors.resize(numLayers + 1);

    // Track the number of chunks finished at each epoch;
    if (staleness != UINT_MAX)
    {
        nodesFinishedEpoch.resize(staleness + 1);
        numFinishedEpoch.resize(staleness + 1);
    }

    // Init it here for collecting data when reading files
    forwardVerticesInitData = new FeatType[getFeatDim(0) * graph.localVtxCnt];
    printLog(nodeId, "Created forwardVerticesInitData");
    forwardGhostInitData = new FeatType[getFeatDim(0) * graph.srcGhostCnt];
    printLog(nodeId, "Created forwardGhostInitData");

    // Create labels storage area. Read in labels and store as one-hot format.
    localVerticesLabels = new FeatType[layerConfig[numLayers] * graph.localVtxCnt];
    printLog(nodeId, "Created localVerticesLabels");

    // Read in initial feature values (input features) & labels.
    readFeaturesFile(featuresFile);
    printLog(nodeId, "Created featuresFile");
    readLabelsFile(labelsFile);
    printLog(nodeId, "Created labelsFile");

#ifdef _GPU_ENABLED_
    printLog(nodeId, "Loading SparseMatrices for GPU");
    NormAdjMatrixIn = new CuMatrix();
    NormAdjMatrixOut = new CuMatrix();
    {
        Matrix onenorms(graph.localVtxCnt, 1, graph.vtxDataVec.data());
        OneNorms = new CuMatrix(onenorms, cu.handle);
        CuMatrix::MemoryPool.erase(OneNorms->devPtr); // don't free
    }
    {
        FeatType *zerobuf = new FeatType[graph.localVtxCnt];
        for (unsigned i = 0; i < graph.localVtxCnt; i++)
            zerobuf[i] = 0;
        Matrix zeronorms(graph.localVtxCnt, 1, zerobuf);
        ZeroNorms = new CuMatrix(zeronorms, cu.handle);
        CuMatrix::MemoryPool.erase(ZeroNorms->devPtr); // don't free
        delete[] zerobuf;
    }
    NormAdjMatrixIn->loadSpCSC(cu.spHandle, graph);
    NormAdjMatrixOut->loadSpCSR(cu.spHandle, graph);
#endif

    // Initialize synchronization utilities.
    recvCnt = 0;
    recvCntLock.init();
    recvCntCond.init(recvCntLock);

    if (nodeId == 0)
    {
        printLog(nodeId, "Creating WeightComm with port %d", weightserverPort);
        weightComm = new WeightComm(weightserverIPFile, weightserverPort);
        printLog(nodeId, "Calling updateChunkCnt");
        weightComm->updateChunkCnt(
            numNodes *
            numLambdasForward); // now set up weight servers only once
        printLog(nodeId, "Finished calling updateChunkCnt");
    }
    else
    {
        weightComm = NULL;
    }

#ifdef _LAMBDA_ENABLED_
    if (mode == LAMBDA)
    { // Lambda
        if (gnn_type == GNN::GCN)
        {
            lambdaName = "gcn";
        }
        else if (gnn_type == GNN::GAT)
        {
            lambdaName = "gat";
        }
        else
        {
            lambdaName = "invalid_lambda_name";
        }

        resComm = new LambdaComm(this);
    }
#endif
#ifdef _CPU_ENABLED_
    if (mode == CPU)
    { // CPU
        resComm = new CPUComm(this);
    }
#endif
#ifdef _GPU_ENABLED_
    if (mode == GPU)
    { // GPU
        resComm = new GPUComm(this);
    }
#endif

    timeInit += getTimer();
    printLog(nodeId, "Engine initialization complete.");
}

/**
 *
 * Destroy the engine.
 *
 */
void Engine::destroy()
{
    // printLog(nodeId, "Destroying the engine...");

    nodeManager.destroy();
    commManager.destroy();

    recvCntLock.destroy();
    recvCntCond.destroy();

    if (nodeId == 0)
    {
        weightComm->shutdown();
        delete weightComm;
    }
    delete resComm;

    // delete[] forwardVerticesInitData;
    // delete[] forwardGhostInitData;
    // delete[] localVerticesLabels;
    for (int i = 0; i < numLayers; i++)
    {
        auto &kkv = savedNNTensors[i];
        for (auto &kv : kkv)
        {
            if (kv.first == "A")
            {
                continue;
            }
            kv.second.free();
        }
    }
    for (auto &kkv : savedEdgeTensors)
    {
        for (auto &kv : kkv)
        {
            delete[] kv.second;
        }
    }
}

void Engine::preallocate_tensors(GNN gnn_type)
{
    // printLog(nodeId, "GNN type of %d", gnn_type);
    switch (gnn_type)
    {
    case GNN::GCN:
        // printLog(nodeId, "Preallocating GCN");
        preallocateGCN();
        break;
    case GNN::GAT:
        // printLog(nodeId, "Preallocating GAT");
        preallocateGAT();
        break;
    default:
        printLog(nodeId, "Unrecognized benchmark type");
    }
}

void Engine::run()
{
    // printLog(nodeId, "Going to preallocate tensor of type %d", gnn_type);
    preallocate_tensors(gnn_type);
    // printLog(nodeId, "Allocated tensors", gnn_type);

    start_time = getCurrentTime();
    switch (gnn_type)
    {
    case GNN::GCN:
    case GNN::GAT:
        runPipeline(); // GAT and GCN share the same pipeline
        break;
    default:
        printLog(nodeId, "Unsupported GNN type");
    }
    end_time = getCurrentTime();
}

void Engine::runPipeline()
{
    // printLog(nodeId, "Starting run pipeline");
    using ThreadVector = std::vector<std::thread>;
    pipelineHalt = false;
    unsigned commThdCnt = dThreads;
    // unsigned commThdCnt = std::max(2u, cThreads / 4);

    auto gaWrkrFunc =
        std::bind(&Engine::gatherWorkFunc, this, std::placeholders::_1);
    ThreadVector gaWrkrThds;
    for (unsigned tid = 0; tid < cThreads; ++tid)
    {
        gaWrkrThds.push_back(std::thread(gaWrkrFunc, 2 + tid));
    }
    // printLog(nodeId, "Allocated gaWrkrThds");

    auto avWrkrFunc =
        std::bind(&Engine::applyVertexWorkFunc, this, std::placeholders::_1);
    ThreadVector avWrkrThds;
    for (unsigned tid = 0; tid < 1; ++tid)
    {
        avWrkrThds.push_back(std::thread(avWrkrFunc, tid));
    }
    // printLog(nodeId, "Allocated avWrkrThds");

    auto scWrkrFunc =
        std::bind(&Engine::scatterWorkFunc, this, std::placeholders::_1);

    // std::thread swt(scWrkrFunc, 1);
    ThreadVector scWrkrThds;
    for (unsigned tid = 0; tid < commThdCnt; ++tid)
    {
        scWrkrThds.push_back(std::thread(scWrkrFunc, tid));
    }
    // printLog(nodeId, "Allocated avWrkrThds");

    auto ghstRcvrFunc =
        std::bind(&Engine::ghostReceiverFunc, this, std::placeholders::_1);
    ThreadVector ghstRcvrThds;
    for (unsigned tid = 0; tid < commThdCnt; ++tid)
    {
        ghstRcvrThds.push_back(std::thread(ghstRcvrFunc, tid));
    }
    // printLog(nodeId, "Allocated ghstRcvrThds");

    auto aeWrkrFunc =
        std::bind(&Engine::applyEdgeWorkFunc, this, std::placeholders::_1);
    ThreadVector aeWrkrThds;
    for (unsigned tid = 0; tid < 1; ++tid)
    {
        aeWrkrThds.push_back(std::thread(aeWrkrFunc, tid));
    }
    // printLog(nodeId, "Allocated aeWrkrThds");

    // printLog(nodeId, "Pre barrier 1");
    nodeManager.barrier();
    // printLog(nodeId, "Post barrier 1");

    // printLog(nodeId, "Pre loaded chunks");
    loadChunks();
    // printLog(nodeId, "Post loaded chunks");

    // Start scheduler
    auto schedFunc =
        std::bind(&Engine::scheduleAsyncFunc, this, std::placeholders::_1);
    ThreadVector schedulerThds;
    for (unsigned tid = 0; tid < 1; ++tid)
    {
        schedulerThds.push_back(std::thread(schedFunc, tid));
    }
    // printLog(nodeId, "Allocated schedulerThds");

    for (unsigned tid = 0; tid < 1; ++tid)
    {
        schedulerThds[tid].join();
    }
    // printLog(nodeId, "Joined schedulerThds");

    // Wait for all nodes to finish
    // printLog(nodeId, "Pre barrier 2");
    nodeManager.barrier();
    // printLog(nodeId, "Post barrier 2");

    for (unsigned tid = 0; tid < cThreads; ++tid)
    {
        gaWrkrThds[tid].join();
    }
    // printLog(nodeId, "Joined gaWrkrThds");

    for (unsigned tid = 0; tid < 1; ++tid)
    {
        avWrkrThds[tid].join();
        aeWrkrThds[tid].join();
    }
    // printLog(nodeId, "Joined avWrkrThds and aeWrkrThds");

    for (unsigned tid = 0; tid < commThdCnt; ++tid)
        scWrkrThds[tid].join();
    // printLog(nodeId, "Joined scWrkrThds");

    for (unsigned tid = 0; tid < commThdCnt; ++tid)
        ghstRcvrThds[tid].join();
    // printLog(nodeId, "Joined ghstRcvrThds");

    {
        // clean up
        unsigned sender, topic;
        char *msgBuf = new char[MAX_MSG_SIZE];
        if (commManager.dataPullIn(&sender, &topic, msgBuf,
                                   MAX_MSG_SIZE))
        {
            printLog(nodeId, "CLEAN UP: Still msgs in buffer");
        };
        while (commManager.dataPullIn(&sender, &topic, msgBuf,
                                      MAX_MSG_SIZE))
        {
        };
        delete[] msgBuf;
    }
}

Engine engine;
