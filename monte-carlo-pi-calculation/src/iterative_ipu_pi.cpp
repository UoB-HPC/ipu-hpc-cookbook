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
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/Fill.hpp>
#include <popops/Expr.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <poprand/codelets.hpp>

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

    popops::addCodelets(graph);
    return graph;
}

auto buildComputeGraph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int numTiles, unsigned long chunk_size) {
    Sequence prog;
    
    tensors["x"] = graph.addVariable(FLOAT, {chunk_size}, "x");
    poputil::mapTensorLinearly(graph, tensors["x"]);

    tensors["y"] = graph.addVariable(FLOAT, {chunk_size}, "y");
    poputil::mapTensorLinearly(graph, tensors["y"]);

    tensors["data"] = graph.addVariable(UNSIGNED_INT, {chunk_size}, "fit");
    poputil::mapTensorLinearly(graph, tensors["data"]);
    //poputil::mapTensorLinearly(graph, fit);
    ///popops::zero(graph, fit, prog);
    //prog.add(poplar::program::PrintTensor("x-debug", tensors["data"]));
    tensors["data"] = popops::map(graph, popops::expr::Add(popops::expr::_3, popops::expr::Cast(popops::expr::Lte(
                                                                popops::expr::Sqrt(
                                                                    popops::expr::Add(
                                                                        popops::expr::Square(popops::expr::_1), 
                                                                        popops::expr::Square(popops::expr::_2)
                                                                    )
                                                                ), popops::expr::Const(1.f)
                                                            ),
                                         UNSIGNED_INT)), {tensors["x"], tensors["y"], tensors["data"]}, prog);

    programs["main"] = prog;
}

auto defineDataStreams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, unsigned long chunk_size) {
    auto xIpuStream = graph.addHostToDeviceFIFO("in_x", FLOAT, chunk_size);
    auto yIpuStream = graph.addHostToDeviceFIFO("in_y", FLOAT, chunk_size);
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", UNSIGNED_INT, chunk_size);

    auto copyToIpuProgram = Sequence({Copy(xIpuStream, tensors["x"]), Copy(yIpuStream, tensors["y"])});
    auto copyToHostProgram = Copy(tensors["data"], fromIpuStream);

    programs["copy_to_ipu"] = copyToIpuProgram;
    programs["copy_to_host"] = copyToHostProgram;
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
    auto chunk_size = options.chunk_size;
    iterations -= iterations % chunk_size;

    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    auto device = getIpuDevice(1);
    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);

    std::cout << "STEP 3: Building the compute graph" << std::endl;
    auto tensors = map<string, Tensor>{};
    auto programs = map<string, Program>{};
    buildComputeGraph(graph, tensors, programs, device->getTarget().getNumTiles(), chunk_size);

    std::cout << "STEP 4: Define data streams" << std::endl;
    defineDataStreams(graph, tensors, programs, chunk_size);

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
            {"debug.retainDebugInformation",      "true"},
            {"exchange.enablePrefetch", "true"}
    };

    Program s = Repeat(iterations / chunk_size, Sequence{programs["copy_to_ipu"], programs["main"], programs["copy_to_host"]});
    auto engine = Engine(graph, {s}, ENGINE_OPTIONS);
        
    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*device);
    engine.enableExecutionProfiling();

    std::cout << "STEP 7: Attach data streams" << std::endl;
    std::random_device dev;
    std::mt19937 mt(dev());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    auto vx = std::vector<float>(iterations);
    auto vy = std::vector<float>(iterations);
    auto results = std::vector<unsigned int>(iterations);
    engine.connectStream("in_x", vx.data(), vx.data() + vx.size());
    engine.connectStream("in_y", vy.data(), vy.data() + vy.size());
    engine.connectStream("FROM_IPU", results.data(), results.data() + results.size());

    std::cout << "STEP 8: Run programs" << std::endl;
    auto hits = 0ul;
    std::generate(vx.begin(), vx.end(), [&mt, &dist]{return dist(mt);});
    std::generate(vy.begin(), vy.end(), [&mt, &dist]{return dist(mt);});
    
    //for (auto & i : vx)
    //    std::cout << i << " ";
    //std::cout << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    engine.run(0, "main"); // Main program
    auto stop = std::chrono::steady_clock::now(); 
    hits += std::accumulate(results.begin(), results.end(), 0);

    std::cout << "STEP 9: Capture debug and profile info" << std::endl;
    serializeGraph(graph);
    captureProfileInfo(engine);
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});

    //for (auto & i : results)
    //    std::cout << i << " ";
    //std::cout << std::endl;

    std::cout << "tests = " << iterations << " took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl; 
    std::cout << "pi = " << std::setprecision(precision) << (4. * hits/(iterations)) << std::endl;

    return EXIT_SUCCESS;
}
