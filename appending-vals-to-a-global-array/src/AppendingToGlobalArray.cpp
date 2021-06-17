#include <iostream>
#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplar/Program.hpp>
#include <algorithm>

#include "CommonIpuUtils.hpp"

constexpr auto NumIterations = 1000;

using namespace poplar;
using namespace poplar::program;

int main() {

    auto device = ipu::getIpuDevice();
    if (!device.has_value()) {
        std::cerr << "Could not attach to IPU device. Aborting" << std::endl;
    }

    auto graph = poplar::Graph(device->getTarget());

    graph.addCodelets({"codelets/AppendingValsToGlobal.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);

    auto data = graph.addVariable(FLOAT, {NumIterations}, "data"); // Where we will store the results
    poputil::mapTensorLinearly(graph, data);

    auto latestResult = graph.addVariable(FLOAT, {}, "latestResult");
    graph.setTileMapping(latestResult, 0); // Store latest result on tile 0, it will be broadcast to all others

    // A dummy operation where we calculate a new value for
    // the latest result and store it in latestResult.
    auto calculateLatestResult = [&](Tensor &latestResult) -> auto {
        auto cs = graph.addComputeSet("calcNextResult");
        auto tileMapping = graph.getTileMapping(data);
        auto v = graph.addVertex(cs, "CalculateNextResult", {
                {"result", latestResult},
        });
        graph.setTileMapping(v, 0);
        return Execute(cs);
    };


    // Add an AppendValToGlobalArray to each tile. Only the one holding the slice of data that
    // matches the current iteration will write the latest value to the array
    auto appendResult = [&](Tensor &data, Tensor &latestResult) -> auto {
        auto cs = graph.addComputeSet("appendLatest");
        auto tileMapping = graph.getTileMapping(data);
        auto tileNum = 0;
        for (auto &tile : tileMapping) {
            for (auto chunk: tile) {
                auto from = chunk.begin();
                auto to = chunk.end();
                auto v = graph.addVertex(cs, "AppendValToGlobalArray", {
                        {"results",       data.slice(from, to)},
                        {"currentResult", latestResult}
                });
                graph.setTileMapping(v, tileNum);
                graph.setInitialValue(v["index"], 0);
                graph.setInitialValue(v["myStartIndex"], from);
            }
            tileNum++;
        }
        return Execute(cs);
    };

    const auto program = Repeat(
            NumIterations,
            Sequence{calculateLatestResult(latestResult),
                     appendResult(data, latestResult)
            }
    );


    auto engine = ipu::prepareEngine(graph, {program}, *device);

    auto timer = ipu::startTimer("Running append program");
    engine.run(0);
    ipu::endTimer(timer);

    return EXIT_SUCCESS;
}

