#include <iostream>
#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplar/Program.hpp>
#include <algorithm>
#include <map>

#include "CommonIpuUtils.hpp"

#define UNUSED(expr) (void)(expr);


constexpr auto NumNodes = 6;
constexpr auto NumEdges = 8;
constexpr auto NumWorkers = 3;
constexpr auto NodeDim = 3;
const auto NumIterations = 100;

using namespace poplar;
using namespace poplar::program;


/**
 * We'll use this static structure instead of the output from a tool like Metis in this example
 */
using Edge = std::tuple<int, int>;
using Node = int;

// The index of edge weights in the flattened tensor
std::map<Edge, int> edgeToIdx = {
        {{0, 1}, 0},
        {{1, 2}, 1},
        {{1, 3}, 2},
        {{2, 3}, 3},
        {{2, 5}, 4},
        {{3, 5}, 5},
        {{4, 5}, 6},
        {{0, 4}, 7},
        // And for convenience also the other way round since our edges are undirected
        {{1, 0}, 0},
        {{2, 1}, 1},
        {{3, 1}, 2},
        {{3, 2}, 3},
        {{5, 2}, 4},
        {{5, 3}, 5},
        {{5, 4}, 6},
        {{4, 0}, 7},
};

struct Partitioning {
    std::vector<Node> localNodes;
    std::vector<Node> foreignNodes;
    std::vector<Edge> localEdges;
    std::vector<Edge> foreignEdges;
};

const Partitioning worker1 = {
        .localNodes = {0, 1},
        .foreignNodes = {2, 3, 4},
        .localEdges =  {{0, 1},
                        {1, 2},
                        {1, 3}},
        .foreignEdges = {{0, 4}}
};

const Partitioning worker2 = {
        .localNodes = {2, 3},
        .foreignNodes = {1, 5},
        .localEdges =  {{2, 3},
                        {3, 5}},
        .foreignEdges = {{2, 5},
                         {1, 3},
                         {1, 2}}
};

const Partitioning worker3 = {
        .localNodes = {4, 5},
        .foreignNodes = {2, 3, 0},
        .localEdges =  {{2, 5},
                        {4, 5},
                        {0, 4}},
        .foreignEdges = {{3, 5}}
};

int numNeighbours(int node) {
    auto count = 0;
    for (const auto &[k, v]: edgeToIdx) {
        UNUSED(v);
        auto[from, to] = k;
        UNUSED(to);
        if (from == node) {
            count++;
        }
    }
    return count;
}

// Set MS bit
auto markLocal = [](unsigned int in) -> unsigned int {
    return in | 0x8000000u;
};

// Clear MS bit
auto markForeign = [](unsigned int in) -> unsigned int {
    return in & 0x7FFFFFFFu;
};

std::vector<unsigned int> edgeMapForNode(const Node node, const Partitioning &partitioning) {
    auto result = std::vector<unsigned int>();
    for (const auto &[k, v]: edgeToIdx) {
        UNUSED(v);
        auto[from, to] = k;
        if (from == node) {
            std::cout << "(" << from << "," << to << "):" << std::endl;
            auto encodedToNodeVal = [&](const unsigned int toNodeId) -> unsigned int {
                auto localNodeIdx = std::find(std::begin(partitioning.localNodes),
                                              std::end(partitioning.localNodes),
                                              toNodeId);
                auto foreignNodeIdx = std::find(std::begin(partitioning.foreignNodes),
                                                std::end(partitioning.foreignNodes),
                                                toNodeId);
                auto toNodeIsLocal = (localNodeIdx != std::end(partitioning.localNodes));
                auto toNodeIsForeign = (foreignNodeIdx != std::end(partitioning.foreignNodes));
                assert(toNodeIsLocal || toNodeIsForeign);
                if (toNodeIsLocal) {
                    return markLocal(*localNodeIdx);
                } else {
                    return markForeign(*foreignNodeIdx);
                }
            }(to);

            auto encodedEdgeVal = [&](const Edge &edge) -> unsigned int {
                auto localEdgeIdx = std::find(std::begin(partitioning.localEdges),
                                              std::end(partitioning.localEdges),
                                              edge);
                auto reversedEdge = Edge(std::get<1>(edge), std::get<0>(edge));
                if (localEdgeIdx == std::end(partitioning.localEdges)) {
                    localEdgeIdx = std::find(std::begin(partitioning.localEdges),
                                             std::end(partitioning.localEdges),
                                             reversedEdge);
                }
                auto foreignEdgeIdx = std::find(std::begin(partitioning.foreignEdges),
                                                std::end(partitioning.foreignEdges),
                                                edge);
                if (foreignEdgeIdx == std::end(partitioning.foreignEdges)) {
                    foreignEdgeIdx = std::find(std::begin(partitioning.foreignEdges),
                                               std::end(partitioning.foreignEdges),
                                               reversedEdge);
                }
                auto edgeIsLocal = (localEdgeIdx != std::end(partitioning.localEdges));
                auto edgeIsForeign = (foreignEdgeIdx != std::end(partitioning.foreignEdges));
                assert(edgeIsLocal || edgeIsForeign);
                if (edgeIsLocal) {
                    return markLocal(localEdgeIdx - std::begin(partitioning.localEdges));
                } else {
                    return markForeign(foreignEdgeIdx - std::begin(partitioning.foreignEdges));
                }
            }(k);

            result.push_back(encodedToNodeVal);
            result.push_back(encodedEdgeVal);
        }
    }
    return result;
}

int main() {

    auto device = ipu::getIpuDevice();
    if (!device.has_value()) {
        std::cerr << "Could not attach to IPU device. Aborting" << std::endl;
    }

    auto graph = poplar::Graph(device->getTarget());

    graph.addCodelets({"codelets/UnstructuredCodelets.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);

    auto emptyList = graph.addVariable(FLOAT, {},
                                       "empty"); // We *might* have empty node or edge lists but still need to wire up something
    graph.setTileMapping(emptyList, 100);
    auto nodeValuesA = graph.addVariable(FLOAT, {NumNodes, NodeDim}, "nodes"); // Nodal values (float[3]'s)
    auto edgeWeightsA = graph.addVariable(FLOAT, {NumEdges, 1}, "edgeWeights"); // Edge weights (float)

    auto nodeValuesB = graph.addVariable(FLOAT, {NumNodes, NodeDim},
                                         "nodesB"); // Double buffered tmp of Nodal values
    auto edgeWeightsB = graph.addVariable(FLOAT, {NumEdges, 1},
                                          "edgeWeightsB"); // Double buffered tmp of Edge weights (float)

    auto workerVertex = [&](ComputeSet &cs, Tensor &nodesIn, Tensor &nodesOut, Tensor &edgesIn,
                            Tensor &edgesOut, const Partitioning &partitioning,
                            const std::string &name, const int tileToPlaceNewVars) -> auto {
        std::cout << "Setting up vertex for " << name << " on tile " << tileToPlaceNewVars << std::endl;
        auto localNodes = [&]() {
            if (partitioning.localNodes.size() > 0) {
                auto tensors = std::vector<Tensor>(partitioning.localNodes.size());

                std::transform(partitioning.localNodes.begin(), partitioning.localNodes.end(), tensors.begin(),
                               [&](int a) -> Tensor { return nodesIn[a]; });
                for (auto &t: tensors) {
                    graph.setTileMapping(t, tileToPlaceNewVars);
                }
                return concat(tensors);
            } else {
                return emptyList;
            }
        }();

        auto updatedLocalNodes = [&]() {
            if (partitioning.localNodes.size() > 0) {
                auto tensors = std::vector<Tensor>(partitioning.localNodes.size());

                std::transform(partitioning.localNodes.begin(), partitioning.localNodes.end(), tensors.begin(),
                               [&](int a) -> Tensor { return nodesOut[a]; });
                for (auto &t: tensors) {
                    graph.setTileMapping(t, tileToPlaceNewVars);
                }
                return concat(tensors);
            } else {
                return emptyList;
            }
        }();

        auto foreignNodes = [&]() {
            if (partitioning.foreignNodes.size() > 0) {
                auto tensors = std::vector<Tensor>(partitioning.foreignNodes.size());
                std::transform(partitioning.foreignNodes.begin(), partitioning.foreignNodes.end(), tensors.begin(),
                               [&](int a) -> Tensor { return nodesIn[a]; });
                return concat(tensors);
            } else {
                return emptyList;
            }
        }();

        auto localEdges = [&]() {
            if (partitioning.localEdges.size() > 0) {
                auto tensors = std::vector<Tensor>(partitioning.localEdges.size());
                std::transform(partitioning.localEdges.begin(), partitioning.localEdges.end(), tensors.begin(),
                               [&](Edge a) -> Tensor {
                                   const auto idx = edgeToIdx[a];
                                   return edgesIn[idx];
                               });
                for (auto &t: tensors) {
                    graph.setTileMapping(t, tileToPlaceNewVars);
                }
                return concat(tensors);
            } else {
                return emptyList;
            }
        }();

        auto updatedLocalEdges = [&]() {
            if (partitioning.localEdges.size() > 0) {
                auto tensors = std::vector<Tensor>(partitioning.localEdges.size());
                std::transform(partitioning.localEdges.begin(), partitioning.localEdges.end(), tensors.begin(),
                               [&](Edge a) -> Tensor {
                                   const auto idx = edgeToIdx[a];
                                   return edgesOut[idx];
                               });
                for (auto &t: tensors) {
                    graph.setTileMapping(t, tileToPlaceNewVars);
                }
                return concat(tensors);
            } else {
                return emptyList;
            }
        }();

        auto foreignEdges = [&]() {
            if (partitioning.foreignEdges.size() > 0) {
                auto tensors = std::vector<Tensor>(partitioning.foreignEdges.size());
                std::transform(partitioning.foreignEdges.begin(), partitioning.foreignEdges.end(), tensors.begin(),
                               [&](Edge a) -> Tensor {
                                   const auto idx = edgeToIdx[a];
                                   return edgesIn[idx];
                               });
                return concat(tensors);
            } else {
                return emptyList;
            }
        }();

        auto connectivityIdxVec = std::vector<unsigned int>(partitioning.localNodes.size() * 2); // two uint32s per node
        auto connectivityMapVec = std::vector<unsigned int>(32); // two uint32s per local node
        auto indexInMap = 0;
        for (auto i = 0u; i < partitioning.localNodes.size(); i++) {

            connectivityIdxVec[i * 2] = indexInMap;
            connectivityIdxVec[i * 2 + 1] = numNeighbours(partitioning.localNodes[i]);

            auto thisNodesEdgeMap = edgeMapForNode(partitioning.localNodes[i], partitioning);
            connectivityMapVec.insert(std::end(connectivityMapVec), std::begin(thisNodesEdgeMap),
                                      std::end(thisNodesEdgeMap));
            indexInMap += 2 * thisNodesEdgeMap.size();
        }
        auto connectivityIdxName = std::string("connectivityIndex").append(name);
        auto connectivityMapName = std::string("connectivityMap").append(name);

        auto connectivityIndex = graph.addConstant(UNSIGNED_INT, {connectivityIdxVec.size()},
                                                   connectivityIdxVec.data(),
                                                   connectivityIdxName);
        auto connectivityMap = graph.addConstant(UNSIGNED_INT, {connectivityMapVec.size()},
                                                 connectivityMapVec.data(),
                                                 connectivityMapName);
        graph.setTileMapping(connectivityIndex, tileToPlaceNewVars);
        graph.setTileMapping(connectivityMap, tileToPlaceNewVars);

        auto v = graph.addVertex(cs, "UpdateVertex", {
                {"localNodes",        localNodes},
                {"foreignNodes",      foreignNodes},
                {"localEdges",        localEdges},
                {"foreignEdges",      foreignEdges},
                {"connectivityMap",   connectivityMap},
                {"connectivityIndex", connectivityIndex},
                {"updatedLocalEdges", updatedLocalEdges},
                {"updatedLocalNodes", updatedLocalNodes}
        });
        graph.setInitialValue(v["numLocalNodes"], partitioning.localNodes.size());
        graph.setInitialValue(v["numForeignNodes"], partitioning.foreignNodes.size());
        graph.setInitialValue(v["numLocalEdges"], partitioning.localEdges.size());
        graph.setInitialValue(v["numForeignEdges"], partitioning.foreignEdges.size());
        return v;
    };

    // Read from ValueA and write to ValueB
    auto updateAToB = [&]() -> auto {
        auto cs = graph.addComputeSet("updateAToB");
        auto v1 = workerVertex(cs, nodeValuesA, nodeValuesB, edgeWeightsA, edgeWeightsB, worker1, "worker1A2B", 1);
        auto v2 = workerVertex(cs, nodeValuesA, nodeValuesB, edgeWeightsA, edgeWeightsB, worker2, "worker2A2B", 2);
        auto v3 = workerVertex(cs, nodeValuesA, nodeValuesB, edgeWeightsA, edgeWeightsB, worker3, "worker3A2B", 3);
        graph.setTileMapping(v1, 1);
        graph.setTileMapping(v2, 2);
        graph.setTileMapping(v3, 3);
        return Execute(cs);
    };

    // Read from ValuesB and write to ValuesA
    auto updateBToA = [&]() -> auto {
        auto cs = graph.addComputeSet("updateBToA");
        auto v1 = workerVertex(cs, nodeValuesB, nodeValuesA, edgeWeightsB, edgeWeightsA, worker1, "worker1B2A", 1);
        auto v2 = workerVertex(cs, nodeValuesB, nodeValuesA, edgeWeightsB, edgeWeightsA, worker2, "worker1B2A", 2);
        auto v3 = workerVertex(cs, nodeValuesB, nodeValuesA, edgeWeightsB, edgeWeightsA, worker3, "worker1B2A", 3);
        graph.setTileMapping(v1, 1);
        graph.setTileMapping(v2, 2);
        graph.setTileMapping(v3, 3);
        return Execute(cs);
    };


    const auto program = Repeat(
            NumIterations,
            Sequence{updateAToB(),
                     updateBToA()
            }
    );


    auto engine = ipu::prepareEngine(graph, {program}, *device);

    auto timer = ipu::startTimer("Running append program");
    engine.run(0);
    ipu::endTimer(timer);

    return EXIT_SUCCESS;
}

