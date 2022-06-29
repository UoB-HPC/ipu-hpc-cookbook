#ifndef IPUCOMMONUTILS_HPP
#define IPUCOMMONUTILS_HPP

#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <popops/ElementWiseUtil.hpp>
#include <poplar/Program.hpp>


namespace ipu {
    using namespace poplar;
    using namespace poplar::program;
    using namespace std;

    Device getIpuModel() {
        IPUModel ipuModel;
        ipuModel.numIPUs = 1;
        ipuModel.tilesPerIPU = 4;
        return ipuModel.createDevice();
    }

    auto getIpuDevice(unsigned int numIpus = 1) -> optional <Device> {
        DeviceManager manager = DeviceManager::createDeviceManager();

        for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
            cerr << "Trying to attach to IPU " << d.getId();
            if (d.attach()) {
                cerr << " - attached" << endl;
                return {move(d)};
            } else {
                cerr << endl;
            }
        }
        cerr << "Error attaching to device" << endl;
        return nullopt;
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
            {"debug.retainDebugInformation",      "true"}
    };

    const auto POPLAR_ENGINE_OPTIONS_RELEASE = OptionFlags{};


    template<class UnaryPredicate>
    void assertThat(const string &&msg, const UnaryPredicate p) {
        if (!p())
            throw std::runtime_error(msg);
    };


/**
 * Similar to Graphcore's poputils Linear tile mapping, but restricts the tensor to live in just 1 IPU's memories
 */
    const auto mapLinearlyOnOneIpu = [&](Tensor &tensor, const int ipuNum, Device& device, Graph &graph) {
        auto totalElements = 1;
        for (auto dim: tensor.shape()) {
            totalElements *= dim;
        }
        int numTilesPerIpu = device.getTarget().getNumTiles() / device.getTarget().getNumIPUs();
        auto itemsPerTile = totalElements / numTilesPerIpu;
        auto numTilesWithExtraItem = totalElements % numTilesPerIpu;
        auto from = 0;
        for (auto tileNum = 0; tileNum < numTilesPerIpu; tileNum++) {
            const auto itemsForThisTile = (tileNum < numTilesWithExtraItem) ? itemsPerTile + 1 : itemsPerTile;
            const auto to = from + itemsForThisTile;
            graph.setTileMapping(tensor.slice(from, to), tileNum + (ipuNum * numTilesPerIpu));
            from = to;
        }
    };

    /** Starts a timer and outputs a message */
    auto startTimer(const string &title) -> auto {
        cout << "Starting [" << title << "]..." << endl;
        return make_pair(title, chrono::high_resolution_clock::now());
    }

    /** Ends a timer and shows the time taken */
    auto endTimer(const auto &timer) -> void {
        auto &[title, tic] = timer;
        auto toc = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast < chrono::duration < double >> (toc - tic).count();
        cout << "[" << title << "] took " << right << setw(12) << setprecision(5) << diff << "s" <<
             endl;
    }

    /**
     * Compiles the graph with the given programs, creates an Engine and loads the engine onto the device
     */
    auto prepareEngine(Graph &graph, ArrayRef <Program> programs, Device &device) -> Engine {
        auto timer = startTimer("Compiling graph, creating engine, and loading to device");
        auto tic = std::get<1>(timer);

        auto progressFunc = [tic](int a, int b) {
            auto toc = chrono::high_resolution_clock::now();
            auto diff = chrono::duration_cast < chrono::duration < double >> (toc - tic).count();
            cout << " ...stage " << a << " of " << b << " after " << right << setw(6)
                 << setprecision(2)
                 << diff << "s" <<
                 endl;
        };

        auto engine = Engine(graph, programs,
                             POPLAR_ENGINE_OPTIONS_DEBUG, progressFunc);
        engine.load(device);
        endTimer(timer);
        return engine;
    }



}

#endif
