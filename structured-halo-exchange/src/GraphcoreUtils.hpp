//
// Created by Thorben Louw on 25/06/2020.
//
#ifndef LBM_GRAPHCORE_GRAPHCOREUTILS_H
#define LBM_GRAPHCORE_GRAPHCOREUTILS_H

#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <chrono>

#include <poplar/DeviceManager.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "StructuredGridUtils.hpp"

using namespace poplar;
using namespace poplar::program;

namespace utils {

    typedef std::map <std::string, Tensor> TensorMap;


    const auto POPLAR_ENGINE_OPTIONS_DEBUG = OptionFlags{
            {"target.saveArchive",                "archive.a"},
            {"debug.instrument",                  "true"},
            {"debug.instrumentCompute",           "true"},
            {"debug.loweredVarDumpFile",          "vars.capnp"},
            {"debug.instrumentControlFlow",       "true"},
            {"debug.computeInstrumentationLevel", "tile"}};

    const auto POPLAR_ENGINE_OPTIONS_NODEBUG = OptionFlags{};

    auto getIpuModel(const unsigned short numIpus = 1) -> std::optional <Device> {
        IPUModel ipuModel;
        ipuModel.numIPUs = numIpus;
        ipuModel.tilesPerIPU = 1216;
        return {ipuModel.createDevice()};
    }

    auto serializeGraph(const Graph &graph) {
        std::ofstream graphSerOfs;
        graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

        graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
        graphSerOfs.close();
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

    auto getIpuDevice(unsigned int numIpus = 1) -> std::optional <Device> {
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

    auto createDebugEngine(Graph &graph, ArrayRef <Program> programs) -> Engine {
        return Engine(graph, programs, POPLAR_ENGINE_OPTIONS_DEBUG);
    }

    auto createReleaseEngine(Graph &graph, ArrayRef <Program> programs) -> Engine {
        return Engine(graph, programs, POPLAR_ENGINE_OPTIONS_NODEBUG);
    }

    auto mapCellsToTiles(Graph &graph, Tensor &cells, const grids::GridPartitioning &tileMappings, bool print = false) {
        const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();
        for (const auto&[target, slice]: tileMappings) {
            const auto tile = target.virtualTile(numTilesPerIpu);

            if (print) {
                std::cout << "tile: " << tile << " ipu: " << target.ipu() << ":" << target.tile() << ":"
                          << target.worker() <<
                          "(r: " << slice.rows().from() << ",c: " << slice.cols().from() << ",w: " << slice.width() <<
                          ",h: " << slice.height() << std::endl;
            }
            graph.setTileMapping(cells
                                         .slice(slice.rows().from(), slice.rows().to(), 0)
                                         .slice(slice.cols().from(), slice.cols().to(), 1),
                                 tile);
        }
    }


    auto applySlice(Tensor &tensor, grids::Slice2D slice) -> Tensor {
        return
                tensor.slice(slice.rows().from(), slice.rows().to(), 0)
                        .slice(slice.cols().from(), slice.cols().to(),
                               1);
    };

    auto stitchHalos(const Tensor &nw, const Tensor &n, const Tensor &ne,
                     const Tensor &w, const Tensor &m, const Tensor &e,
                     const Tensor &sw, const Tensor &s, const Tensor &se) -> Tensor {
        return concat({
                              concat({nw, w, sw}),
                              concat({n, m, s}),
                              concat({ne, e, se})
                      }, 1);
    }


    const auto timedStep = [](const std::string description, auto f) -> double {
        std::cerr << std::setw(60) << description;
        auto tic = std::chrono::high_resolution_clock::now();
        f();
        auto toc = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();
        std::cerr << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;
        return diff;
    };


}

#endif //LBM_GRAPHCORE_GRAPHCOREUTILS_H
