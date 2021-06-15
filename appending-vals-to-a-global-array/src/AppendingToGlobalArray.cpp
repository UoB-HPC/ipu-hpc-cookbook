#include <iostream>
#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplar/Program.hpp>
#include <algorithm>

#include "CommonIpuUtils.hpp"

constexpr auto NumIterations = 32000;

using namespace poplar;
using namespace poplar::program;

int main() {

    auto device = ipu::getIpuDevice(2);
    if (!device.has_value()) {
        std::cerr << "Could not attach to IPU device. Aborting" << std::endl;
    }

    auto graph = poplar::Graph(device->getTarget());

    graph.addCodelets({"codelets/AppendingValsToGlobal.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);

    auto data = graph.addVariable(FLOAT, {NumIterations}, "data"); // Where we will store the results
    poputil::mapTensorLinearly(graph, data);

    auto latestResult = graph.addVariable(FLOAT, {NumIterations}, "latestResult");
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
        auto cs = graph.addComputeSet("processData");
        auto tileMapping = graph.getTileMapping(data);
        auto tileNum = 0;
        for (auto &tile : tileMapping) {
            for (auto chunk: tile) {
                auto from = chunk.begin();
                auto to = chunk.end();
                auto v = graph.addVertex(cs, "AppendValToGlobalArray", {
                        {"results", data.slice(from, to)},
                        {"currentResult", latestResult}
                });
                graph.setTileMapping(v, tileNum);
                graph.setInitalValue(v["index"], 0);
                graph.setInitalValue(v["myStartIndex"], from);
            }
            tileNum++;
        }
        return Execute(cs);
    };

    const auto program =
            Repeat(NumIterations ,
                   Sequence{calculateLatestResult(latestResult),
                            appendResult(data, latestResult)};
            );


    auto engine = ipu::prepareEngine(graph, {program}, *device);

    return EXIT_SUCCESS;
}

