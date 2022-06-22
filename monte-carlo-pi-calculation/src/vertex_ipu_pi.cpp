// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "pi_options.hpp"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>

using ::std::map;
using ::std::vector;
using ::std::string;
using ::std::optional;

using ::poplar::BOOL;
using ::poplar::FLOAT;
using ::poplar::UNSIGNED_INT;
using ::poplar::OptionFlags;
using ::poplar::Tensor;
using ::poplar::Graph;
using ::poplar::Engine;
using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::TargetType;
using ::poplar::program::Program;
using ::poplar::program::Sequence;
using ::poplar::program::Copy;
using ::poplar::program::Repeat;
using ::poplar::program::Execute;

static const auto MAX_TENSOR_SIZE = 55000000ul;

auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        } else {
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

auto createGraphAndAddCodelets(const optional<Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());

    graph.addCodelets({"pi_vertex.cpp"}, "-O3");
    return graph;
}

auto serializeGraph(const Graph &graph) {
    std::ofstream graphSerOfs;
    graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

    graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
}

auto captureProfileInfo(Engine &engine) {
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);
}

int main(int argc, char *argv[]) {
    pi_options options = parse_options(argc, argv, "IPU PI Iterative");
    auto precision = options.precision;
    auto iterations = options.iterations;

    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    auto device = getIpuDevice(options.num_ipus);
    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);

    std::cout << "STEP 4: Define data streams" << std::endl;
    size_t numTiles = device->getTarget().getNumTiles();
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", UNSIGNED_INT, numTiles * 6);

    std::cout << "STEP 3: Building the compute graph" << std::endl;
    auto counts = graph.addVariable(UNSIGNED_INT, {numTiles * 6}, "counts");
    poputil::mapTensorLinearly(graph, counts);

    const auto NumElemsPerTile = iterations / (numTiles * 6);
    auto cs = graph.addComputeSet("loopBody");
    std::cout << "numTiles = " << numTiles << std::endl;
    for (auto tileNum = 0u; tileNum < numTiles; tileNum++) {
        const auto sliceStart = tileNum * 6;
        const auto sliceEnd = (tileNum + 1) * 6; 

        auto v = graph.addVertex(cs, "PiVertex", {
                {"hits", counts.slice(sliceStart, sliceEnd)}
        });
        graph.setInitialValue(v["iterations"], NumElemsPerTile);
        graph.setPerfEstimate(v, 10); // Ideally you'd get this as right as possible
        graph.setTileMapping(v, tileNum);
    }
 
    std::cout << "STEP 5: Create engine and compile graph" << std::endl;
    auto ENGINE_OPTIONS = OptionFlags{
            {"target.saveArchive",                "archive.a"},
            {"debug.instrument",                  "true"},
            {"debug.instrumentCompute",           "true"},
            {"debug.loweredVarDumpFile",          "vars.capnp"},
            {"debug.instrumentControlFlow",       "true"},
            {"debug.computeInstrumentationLevel", "tile"},
            {"debug.outputAllSymbols",            "true"},
            {"autoReport.all",                    "true"},
            {"autoReport.outputSerializedGraph",  "true"},
            {"debug.retainDebugInformation",      "true"}
    };
    auto engine = Engine(graph, Sequence({Execute(cs), Copy(counts, fromIpuStream)}), ENGINE_OPTIONS);
        
    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*device);
    engine.enableExecutionProfiling();

    std::cout << "STEP 7: Attach data streams" << std::endl;
    auto results = std::vector<unsigned int>(numTiles * 6);
    engine.connectStream("FROM_IPU", results.data(), results.data() + results.size());

    std::cout << "STEP 8: Run programs" << std::endl;
    auto hits = 0ull;
    
    auto start = std::chrono::steady_clock::now();
    engine.run(0, "main"); // Main program
    auto stop = std::chrono::steady_clock::now();
    for (size_t i = 0; i < results.size(); i++) {
        hits += results[i];
        //std::cout << results[i] << " ";
    }
    //std::cout << std::endl;

    std::cout << "STEP 9: Capture debug and profile info" << std::endl;
    serializeGraph(graph);
    captureProfileInfo(engine);
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});
    std::cout << std::endl;
    std::cout << *std::max_element(results.begin(), results.end()) << std::endl;
    std::cout << "chunk_size = " << numTiles * 6 << " repeats = " << iterations / numTiles * 6 << std::endl; 
    std::cout << "tests = " << iterations << " took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl; 
    std::cout << "pi = " << std::setprecision(precision) << (4. * hits/(iterations)) << std::endl;

    return EXIT_SUCCESS;
}
