#include <iostream>
#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/CycleCount.hpp>
#include <popops/AllTrue.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/codelets.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <poplar/Program.hpp>
#include <cmath>
#include <random>
#include "codelets/ParticleCodeletsCommon.h"

const auto InitialParticles = 1000;
const auto GlobalXMin = 0;
const auto GlobalXMax = 1000;
const auto GlobalYMin = 0;
const auto GlobalYMax = 1000;
const auto MaxIters = 100;
const auto MaxMem = 180 * 1024;
const auto NumIpus = 1;
const auto NumProcessors = 900 * NumIpus;
using namespace poplar;
using namespace poplar::program;

poplar::Device getIpuModel() {
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = 1;
    ipuModel.tilesPerIPU = 4;
    return ipuModel.createDevice();
}

auto getIpuDevice(unsigned int numIpus = 1) -> std::optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();

    // Attempt to connect to a single IPU
    for (auto &d : manager.getDevices(poplar::TargetType::IPU, numIpus)) {
        std::cerr << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cerr << " - attached" << std::endl;
            return {std::move(d)};
        } else {
            std::cerr << std::endl;
        }
    }
    std::cerr << "Error attaching to device" << std::endl;
    return std::nullopt;
}

auto initialiseTileData(char *buf, const size_t numProcessors, const size_t MemSizePerTile) {


    memset(buf, 0, MemSizePerTile * numProcessors);

    std::default_random_engine generator;
    std::uniform_real_distribution<float> speed_distribution(0, 100);
    std::uniform_real_distribution<float> angle_distribution(0, 2 * PI);

    //For now our partitioning is basic, we just assume a square number of processors and do nxn blocks
    auto R = (int) sqrt(numProcessors);
    auto C = (int) sqrt(numProcessors);
//    assert(R * C == numProcessors);
    for (auto tileNum = 0u; tileNum < numProcessors; tileNum++) {
        auto tileOffset = MemSizePerTile * tileNum;
        auto tileData = reinterpret_cast<TileData *const>(&buf[tileOffset]);
        tileData->myRank = tileNum;
        tileData->numProcessors = numProcessors;
        tileData->numParticles = InitialParticles;
        tileData->nextToShed = -1;
        tileData->global.min.x = GlobalXMin;
        tileData->global.min.y = GlobalYMin;
        tileData->global.max.x = GlobalXMax;
        tileData->global.max.y = GlobalYMax;

        auto row = (tileNum / C);
        auto col = (tileNum % C);
        auto heightPerRow = (GlobalYMax - GlobalYMin) / R;
        auto widthPerCol = (GlobalXMax - GlobalXMin) / C;

        tileData->local.min.x = GlobalXMin + col * widthPerCol;
        tileData->local.max.x = tileData->local.min.x + widthPerCol;
        tileData->local.min.y = GlobalXMin + row * heightPerRow;
        tileData->local.max.y = tileData->local.min.y + heightPerRow;


        std::uniform_real_distribution<float> x_distribution(tileData->local.min.x, tileData->local.max.x);
        std::uniform_real_distribution<float> y_distribution(tileData->local.min.y, tileData->local.max.y);

        for (auto i = 0; i < InitialParticles; i++) {
            Particle *const p = &tileData->particles[i];
            p->position = Vector2D{x_distribution(generator), y_distribution(generator)};
            auto speed = speed_distribution(generator);
            auto angle = angle_distribution(generator);
            p->velocity = Vector2D{speed * cosf(angle), speed * sinf(angle)};
        }
    }
}

auto captureProfileInfo(Engine &engine) {
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);

    serializeToJSON(graphOfs, engine.getGraphProfile(), false);
    serializeToJSON(executionOfs, engine.getExecutionProfile(), false);

    graphOfs.close();
    executionOfs.close();


}

template<typename T>
auto norm2(T a, T b) -> T {
    return a * a + b * b;
}

auto angle(float x, float y) -> float {
    return atan2f(y, x);
}

const auto POPLAR_ENGINE_OPTIONS_DEBUG = OptionFlags{
        {"target.saveArchive",                "archive.a"},
        {"debug.instrument",                  "true"},
        {"debug.instrumentCompute",           "true"},
        {"debug.loweredVarDumpFile",          "vars.capnp"},
        {"debug.instrumentControlFlow",       "true"},
        {"debug.computeInstrumentationLevel", "tile"},
        {"debug.outputAllSymbols",            "true"},
        {"autoReport.all",                    "true"},
        {"autoReport.outputSerializedGraph",  "true"},
        {"debug.retainDebugInformation",      "true"},
//        {"debug.runtimeVerify","true"},
//        {"debug.trace", "true"},
//        {"debug.traceFile", "debug.trace.out"},
//        {"debug.verify", "true"}
};

const auto POPLAR_ENGINE_OPTIONS_RELEASE = OptionFlags{};

auto printTileData(const TileData &tileData, bool ignoreGlobals, FILE *fptr) -> void {
    fprintf(fptr, "{");
    fprintf(fptr, "\"%s\":%d,", "rank", tileData.myRank);
    if (!ignoreGlobals) {
        fprintf(fptr, "\"%s\":%d,", "numProcessors", tileData.numProcessors);
        fprintf(fptr, "\"globalBounds\":{");
        fprintf(fptr, "\"%s\":%f,", "x_min", tileData.global.min.x);
        fprintf(fptr, "\"%s\":%f,", "y_min", tileData.global.min.x);
        fprintf(fptr, "\"%s\":%f,", "x_max", tileData.global.max.x);
        fprintf(fptr, "\"%s\":%f", "y_max", tileData.global.max.y);
        fprintf(fptr, "},");
    }
    fprintf(fptr, "\"localBounds\":{");
    fprintf(fptr, "\"%s\":%f,", "x_min", tileData.local.min.x);
    fprintf(fptr, "\"%s\":%f,", "y_min", tileData.local.min.y);
    fprintf(fptr, "\"%s\":%f,", "x_max", tileData.local.max.x);
    fprintf(fptr, "\"%s\":%f", "y_max", tileData.local.max.y);
    fprintf(fptr, "},");
    fprintf(fptr, "\"%s\":%d,", "numParticles", tileData.numParticles);
    fprintf(fptr, "\"%s\":%d,", "nextToShed", tileData.nextToShed);
    fprintf(fptr, "\"%s\":%d,", "shedThisIter", tileData.particlesShedThisIter);
    fprintf(fptr, "\"%s\":%d,", "acceptedThisIter", tileData.particlesAcceptedThisIter);
    fprintf(fptr, "\"%s\":%d,", "offeredToMeThisIter", tileData.offeredToMeThisIter);
    fprintf(fptr, "\"particles\":[");
    for (auto i = 0; i < tileData.numParticles; i++) {
        auto printParticle = [fptr](const Particle &particle) {
            fprintf(fptr, "{");
            fprintf(fptr, "\"%s\":%f,", "x", particle.position.x);
            fprintf(fptr, "\"%s\":%f,", "y", particle.position.y);
            fprintf(fptr, "\"%s\":%f,", "speed", norm2(particle.velocity.x, particle.velocity.y));
            fprintf(fptr, "\"%s\":%f", "angle", angle(particle.velocity.x, particle.velocity.y));
            fprintf(fptr, "}");
        };
        printParticle(tileData.particles[i]);
        if (i != tileData.numParticles - 1) fprintf(fptr, ",");
    }

    fprintf(fptr, "]}");
}

void deserialiseToFile(char *buf, int iter, size_t numProcessors, size_t SizeOfMemBlockPerTile) {
    FILE *fptr = fopen(("data." + std::to_string(iter) + ".json").c_str(), "w");
    fprintf(fptr, "{\"data\":[\n");
    for (auto tileNum = 0u; tileNum < numProcessors; tileNum++) {
        auto tileOffset = SizeOfMemBlockPerTile * tileNum;
        auto tileData = reinterpret_cast<TileData *const>(&buf[tileOffset]);
        printTileData(*tileData, tileNum != 0, fptr);
        if (tileNum != numProcessors - 1) fprintf(fptr, ",");
        fprintf(fptr, "\n");
    }
    fprintf(fptr, "]}\n");
    fclose(fptr);
}


int main() {

//    auto device = std::optional<Device>{getIpuModel()};
    auto device = getIpuDevice(NumIpus);
    device = device->createVirtualDevice(NumProcessors / NumIpus);
    if (!device.has_value()) {
        std::cerr << "Could not attach to IPU device. Aborting" << std::endl;
    }

    auto graph = poplar::Graph(device->getTarget());


    const long unsigned int NUM_PROCESSORS = device->getTarget().getNumTiles();

    auto mapNPerTile = [&](Tensor &t, int n) {
        for (auto tileNum = 0u; tileNum < NUM_PROCESSORS; tileNum++) {
            graph.setTileMapping(t.slice(tileNum * n, (tileNum + 1) * n), tileNum);
        }
    };

    popops::addCodelets(graph);
    graph.addCodelets({"codelets/ParticleSimCodelet.cpp"}, "-O0 -I codelets");

    auto particleToShed = graph.addVariable(poplar::FLOAT, {NUM_PROCESSORS, PARTICLE_DIM}, "particlesToShed");
    mapNPerTile(particleToShed, 1);

    auto hasParticlesToShed = graph.addVariable(
            poplar::BOOL,
            {NUM_PROCESSORS},
            "hasParticlesToShed");
    mapNPerTile(hasParticlesToShed, 1);


    auto memories = graph.addVariable(poplar::CHAR, {NUM_PROCESSORS, MaxMem}, "memories");
    mapNPerTile(memories, 1);

    Sequence findAlienParticle;
    {
        auto cs = graph.addComputeSet("findFirstParticleToShed");
        for (auto tileNum = 0u; tileNum < NUM_PROCESSORS; tileNum++) {
            auto v = graph.addVertex(cs, "FindFirstAlienParticle", {
                    {"data",               memories[tileNum]},
                    {"hasParticlesToShed", hasParticlesToShed[tileNum]}

            });
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tileNum);
        }
        findAlienParticle = Execute(cs);
    }


    Sequence exchangeParticles = {};
    {
        auto csOffer = graph.addComputeSet("offerParticles");
        for (auto tileNum = 0u; tileNum < NUM_PROCESSORS; tileNum++) {
            auto v = graph.addVertex(csOffer, "OfferNextAlienParticle",
                                     {
                                             {"data",           memories[tileNum]},
                                             {"particleToShed", particleToShed[tileNum]}
                                     });
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tileNum);
        }
        exchangeParticles.add(Execute(csOffer));

        auto csAccept = graph.addComputeSet("acceptParticles");
        // We only wire up neighbours (assume you can't pass through a neighbour in 1 timestep)
        auto findNeighbours = [&](int tileNum) -> std::vector<int> {
            auto rowsOfTiles = (int) sqrt(NumProcessors);
            auto colsOfTiles = (int) sqrt(NumProcessors);
            int myRow = tileNum / colsOfTiles;
            int myCol = tileNum - myRow * colsOfTiles;
            auto north = std::optional<int>{
                    myRow > 0 ?
                    std::optional{(myRow - 1) * colsOfTiles + myCol} :
                    std::nullopt
            };
            auto south = std::optional<int>{
                    myRow < rowsOfTiles - 1 ?
                    std::optional{(myRow + 1) * colsOfTiles + myCol} :
                    std::nullopt
            };
            auto east = std::optional<int>{
                    (myCol < colsOfTiles - 1) && (tileNum != NumProcessors - 1) ?
                    std::optional{myRow * colsOfTiles + myCol + 1} :
                    std::nullopt
            };
            auto west = std::optional<int>{
                    myCol > 0 ?
                    std::optional{myRow * colsOfTiles + myCol - 1} :
                    std::nullopt
            };
            auto nw = std::optional<int>{
                    (myRow > 0) && (myCol > 0) ?
                    std::optional{(myRow - 1) * colsOfTiles + myCol - 1} :
                    std::nullopt
            };
            auto ne = std::optional<int>{
                    (myRow > 0) && (myCol < colsOfTiles - 1) ?
                    std::optional{(myRow - 1) * colsOfTiles + myCol + 1} :
                    std::nullopt};
            auto sw = std::optional<int>{
                    (myRow < rowsOfTiles - 1) && (myCol > 0) ?
                    std::optional{(myRow + 1) * colsOfTiles + myCol - 1} :
                    std::nullopt};
            auto se = std::optional<int>{
                    (myRow < rowsOfTiles - 1) && (myCol < colsOfTiles - 1) ?
                    std::optional{(myRow + 1) * colsOfTiles + myCol + 1} :
                    std::nullopt};
            auto result = std::vector<int>{};
            auto options = std::vector<std::optional<int>>{north, south, east, west, nw, ne, sw, se};
            for (auto item : options) {
                if (item.has_value()) {
                    result.push_back(item.value());
                }
            }
            std::sort(result.begin(), result.end());
            return result;
        };

        for (auto tileNum = 0u; tileNum < NUM_PROCESSORS; tileNum++) {
            auto neighbours = findNeighbours(tileNum);
            assert(!neighbours.empty());
            Tensor particleToShedSlices = particleToShed[neighbours[0]].flatten();
            Tensor isOfferingSlices = hasParticlesToShed[neighbours[0]].flatten();
            for (auto i = 1u; i < neighbours.size(); i++) {
                particleToShedSlices = poplar::concat(particleToShedSlices,
                                                      particleToShed[neighbours[i]].flatten());
                isOfferingSlices = poplar::concat(isOfferingSlices, hasParticlesToShed[neighbours[i]].flatten());
            }
            auto v = graph.addVertex(csAccept, "AcceptAlienParticles",
                                     {
                                             {"data",                  memories[tileNum]},
                                             {"potentialNewParticles", particleToShedSlices},
                                             {"isOfferingParticle",    isOfferingSlices},
                                     });
            graph.setInitialValue(v["numNeighbours"], neighbours.size());
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tileNum);
        }
        exchangeParticles.add(Execute(csAccept));

        auto csFindNext = graph.addComputeSet("findNextAlientParticle");
        for (auto tileNum = 0u; tileNum < NUM_PROCESSORS; tileNum++) {
            auto v = graph.addVertex(csFindNext, "FindNextAlienParticle",
                                     {
                                             {"data",               memories[tileNum]},
                                             {"hasParticlesToShed", hasParticlesToShed[tileNum]}

                                     });
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tileNum);
        }
        exchangeParticles.add(Execute(csFindNext));
    }

    Sequence reduceHasParticlesToShed;
    auto stillParticlesToShed = popops::logicalNot(graph,
                                                   popops::allTrue(graph,
                                                                   popops::logicalNot(graph, hasParticlesToShed,
                                                                                      reduceHasParticlesToShed),
                                                                   reduceHasParticlesToShed),
                                                   reduceHasParticlesToShed);


    auto updatePositionsCs = graph.addComputeSet("updatePositions");
    auto updateTimestepCs = graph.addComputeSet("timestep");
    for (
            auto tileNum = 0u;
            tileNum < NUM_PROCESSORS;
            tileNum++) {
        auto v = graph.addVertex(updatePositionsCs, "CalculateNextPositions",
                                 {
                                         {"data", memories[tileNum]},
                                 });
        graph.setCycleEstimate(v,100);
        graph.setTileMapping(v, tileNum);
    }
    Sequence updateParticlePositions;
    updateParticlePositions.add(Execute(updatePositionsCs));
    updateParticlePositions.add(Execute(updateTimestepCs));

    Sequence loopUntilAllParticlesExchanged = RepeatWhileTrue(
            reduceHasParticlesToShed,
            stillParticlesToShed,
            exchangeParticles
    );
    const auto memoryOut = graph.addDeviceToHostFIFO("<<data", CHAR,
                                                     NUM_PROCESSORS * MaxMem);
    const auto memoryIn = graph.addHostToDeviceFIFO(">>data", CHAR,
                                                    NUM_PROCESSORS * MaxMem);
    auto copyBackToHost = Copy(memories, memoryOut);


    Sequence timestepProgram = Sequence{
            findAlienParticle,
            loopUntilAllParticlesExchanged,
            updateParticlePositions
    };

    Program copyInitialData = Copy(memoryIn, memories);

    char *dataBuf = new char[MaxMem * NUM_PROCESSORS];

    std::cout << "Compiling..." <<
              std::endl;
    auto tic = std::chrono::high_resolution_clock::now();

    auto progressFunc = [tic](int a, int b) {
        auto toc = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
        std::cout << " ...stage " << a << " of " << b << " after " << std::right << std::setw(6)
                  << std::setprecision(2)
                  << diff << "s" <<
                  std::endl;
    };


    auto engine = Engine(graph, {copyInitialData, timestepProgram, copyBackToHost},
                         POPLAR_ENGINE_OPTIONS_RELEASE, progressFunc);
    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    engine.load(*device);
    engine.disableExecutionProfiling();


    initialiseTileData(dataBuf, NUM_PROCESSORS, MaxMem);
    engine.connectStream("<<data", dataBuf);
    engine.connectStream(">>data", dataBuf);


    std::cout << "Sending initial data..." << std::endl;

    engine.run(0);

    engine.run(2); // Copy back

    deserialiseToFile(dataBuf, 0, NUM_PROCESSORS, MaxMem);


    for (auto iter = 1; iter <= MaxIters; iter++) {
        std::cout << "Running iteration " << iter << ":" << std::endl;
        tic = std::chrono::high_resolution_clock::now();
        if (iter == 2) {
            engine.resetExecutionProfile();
            engine.enableExecutionProfiling();

        }
        engine.run(1);
        if (iter == 2) {
            engine.disableExecutionProfiling();
            captureProfileInfo(engine);
        }

        toc = std::chrono::high_resolution_clock::now();


        diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;
        engine.run(2); // Copy back

//        deserialiseToFile(dataBuf, iter, NUM_PROCESSORS, MaxMem);
    }

    engine.printProfileSummary(std::cout,
                               OptionFlags{
//                                       {"showVarStorage", "true"},
//                                       {"showOptimizations", "true"},
//                                       { "showExecutionSteps", "false" }
                               }
    );


//    for (auto i = 0; i < NUM_PROCESSORS; i++) {
//        std::cout << dataBuf[i] << std::endl;
//    }




    return 0;
}

