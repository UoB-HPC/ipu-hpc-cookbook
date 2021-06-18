# Unstructured neighbour lists

In this example, we'll use the simple graph (in the sense of 'unstructured mesh', not 'Poplar compute graph') shown below, and show how to solve a problem that
is partitioned over 3 "workers" (which we can say correspond to tiles for now), and
which involved updating both node and edge values based on neighbour node and edge values.

![The simple graph and its decomposition][UnstructuredGraph]

[UnstructuredGraph]: ./UnstructuredGraphForIpuCookbook.png "Unstructured graph partitioned over 3 workers"

The graph is partitioned so that nodes 1 and 2 are assigned to worker 1,
nodes 3 and 4 to worker 2, and nodes 5 and 6 to worker 3. Similarly, the 
edges {(1,2), (2,3), (2,4)} are "owned" by worker 1, edges {(3,4) and (4,6)}
are "owned" by worker 2 and edges {(1,5), (3,6), (5,6)} are owned by worker 3. 

```C++
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

```

Note that, in contrast to the structured grid example, nodes have variable numbers of neighbours, and neighbours are in no 
particular order.

In our problem, nodes have values that are a vector of 3 `floats`, which edge
weights are scalar `float`s. 

At each iteration we update the node value such that
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{n}_i^{t%2b1} = 0.8\mathbf{n}_i^t %2b 0.2(\sum_{j \in D}\mathbf{n}_j^te_{ij}^t)">
and the edge weights so that <img src="https://render.githubusercontent.com/render/math?math=e_{ij}^{t%2b 1} = \frac{(\mathbf{n}_i^t)^T (\mathbf{n}_j^t)}{6}">

In other words, nodes are updated according the value of nodes they are connected to, weighted with the edge
strength, and edges are redefined as a scaled dot product of the incident nodes. This is an arbitrary example: we just wanted enough complexity that we have to show
managing and updating both node and edge values, where some nodes and edges are "owned" by other workers.

At each iteration, a worker only updates the nodes and edges it "owns", even though
it uses information from neighbouring workers' edges and nodes. Neighbouring 
node and edge values are sent from their owners at the start of every iteration.

## Partitioning graphs
You can use existing software such as Scotch or Metis to partition graphs onto workers (tiles, or threads on a tile).
 If you use `popsparse`, you probably have these libraries available already.
 Then we exploit the compute graph in Poplar allows us to capture the communication that will occur between iterations
an _compile_ it upfront, allowing the Graph compiler to optimise it. 

In the most extreme case you could partition down to 1 graph node per worker, but this would
be very suboptimal in terms of memory use and communication.


This is a static C++ representation of the partitioning in the diagram we showed previously. Note
that node numbering starts at 0, so is 1 less than the node ids shown in the diagram.
```C++

/**
 * We'll use this static structure instead of the output from a
 * tool like Metis in this example
 */
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

```

## What vertexes look like

Since we can't represent complex data structures like maps
easily in a codelet, we will need to keep lists of pointers to neighbours and edges
for each node as follows:

* (Input) `localNodes`: a 2D tensor of my node values (locally renumbered): 3 floats for each node
* (Input) `foreignNodes` a 2D tensor of node values I use but don't own (locally renumbered): 3 floats for each node
* (Input) `localEdges`: a 1D tensor of edge values I own 
* (Input) `foreignEdges`: a 1D tensor of edge values I use but don't own
* (Input) `connectivityIndex`: 2 ints for each local node, a specifying the offset index in `connectivityMap` and its number of edges
* (Input) `connectivityMap`: A list of unsigned ints, being tuples that represent: the node array index (and local indicator) and edge array index (and local indicator).
 The local indicator is the highest bit of the unsigned int, indicating whether this is a reference into
 the local or foreign list.
* (Output) `updatedLocalNodes` The updated value of locally owned nodes
* (Output) `updatedLocalEdgees` The updated value of locally owned edges

Vertexes are only aware of *local* graph numberings (i.e. indexes in their local arrays). 
The global numbering and broadcasts of updated values between tiles is taken care of by the compute graph structure
and the graph compiler. 


```C++

const static auto NodeDim = 3;

class UpdateVertex : public Vertex {
public:
    Input <Vector<float, VectorLayout::ONE_PTR>> localNodes;
    Input <Vector<float, VectorLayout::ONE_PTR>> foreignNodes;
    Input <Vector<float, VectorLayout::ONE_PTR>> localEdges;
    Input <Vector<float, VectorLayout::ONE_PTR>> foreignEdges;
    Input <Vector<unsigned int, VectorLayout::ONE_PTR>> connectivityIndex;
    Input <Vector<unsigned int, VectorLayout::ONE_PTR>> connectivityMap;
    Output <Vector<float, VectorLayout::ONE_PTR>> updatedLocalNodes;
    Output <Vector<float, VectorLayout::ONE_PTR>> updatedLocalEdges;

    int numLocalNodes;
    int numLocalEdges;
    int numForeignNodes;
    int numForeignEdges;

    auto compute() -> bool {
        for (auto i = 0; i < numLocalNodes; i++) {
            updatedLocalEdges[i] = 0.f;
        }

        for (int i = 0; i < numLocalNodes; i++) {
            const auto thisNode = &localNodes[i * NodeDim];
            auto updatedNode = &updatedLocalNodes[i * NodeDim];

            updatedNode[0] = thisNode[0] * 0.8f;
            updatedNode[1] = thisNode[1] * 0.8f;
            updatedNode[2] = thisNode[2] * 0.8f;


            auto thisNodeConnectivity = &connectivityMap[i];
            const auto numEdges = thisNodeConnectivity[1];
            const auto mapIndex = thisNodeConnectivity[0];

            for (auto edge = 0; edge < numEdges; edge++) {
                auto incidentMapAddr = &connectivityMap[mapIndex + edge * 2];
                auto incidentNodeIdx = incidentMapAddr[0];
                auto edgeWeightIdx = incidentMapAddr[1];

                const bool incidentNodeIsLocal = (incidentNodeIdx & 0x80000000) == 0x80000000;
                const unsigned int incidentNodeIndex = (incidentNodeIdx & 0x7FFFFFFF);
                const bool edgeIsLocal = (edgeWeightIdx & 0x80000000) == 0x80000000;
                const unsigned int edgeIndex = (edgeWeightIdx & 0x7FFFFFFF);

                const auto incidentNodeAddr = incidentNodeIsLocal ?
                                              &localNodes[incidentNodeIndex * NodeDim] :
                                              &foreignNodes[incidentNodeIndex * NodeDim];

                const auto edgeWeight = edgeIsLocal ? localEdges[edgeIndex] : foreignEdges[edgeIndex];

                updatedNode[0] += incidentNodeAddr[0] * 0.2f * edgeWeight;
                updatedNode[1] += incidentNodeAddr[1] * 0.2f * edgeWeight;
                updatedNode[2] += incidentNodeAddr[2] * 0.2f * edgeWeight;

                if (edgeIsLocal &&
                    updatedLocalEdges[edgeIndex] != 0.f) { // We don't want to update twice if both incident
                    // nodes are local
                    updatedLocalEdges[edgeIndex] = (
                                                           thisNode[0] * incidentNodeAddr[0] +
                                                           thisNode[1] * incidentNodeAddr[1] +
                                                           thisNode[2] * incidentNodeAddr[2]) / 6.f;
                }

            }

        }

        return true;
    }
};
```


We calculate and wire up the static `connectivityIndex` and `connectivityMap` for each
vertex when we build the compute graph.

## The compute graph
We use a double-buffered approach and have two sets of nodes and edges (A and B).
To avoid graph cycles, we can't read from A and write to A in the same compute set
from different vertex instances. So we run iterations as

* Vertex calculation reading from A and writing to B
* Vertex calculation reading from B and writing to A

This approach does use double the required memory.

For unstructured graph communication, the complexity is in building the graph
and the calculations for the connectivityIndexes and Maps to be wired. During
graph compilation, the compiler generates the necessary communication between
neighbours:

```C++

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

...
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
        auto connectivityMapVec = std::vector<unsigned int>(); // will be two uint32s per local node
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
...
```


We can see from the execution trace that the compiler has automatically generated the correct communication to update 
all the "foreign" lists before vertexes execute:
![Execution trace of unstructured][UnstructuredGraphExec]

[UnstructuredGraphExec]: ./unstructured-neighbour-list-execution.png "Unstructured neighbour list execution"


## Notes:
* We used a hardcoded toy 3-partition graph here. For a real application, use partitioning software 
like ParMetis to partition graphs/unstructured meshes between workers
* There's no difference between 'tiles' and worker instances
 in this example, but you could schedule 6 workers per tile (or hierarchically partition the graph)
* You want to maximise parallelism and need to spread the graph between the
  memory of the tiles, so should partition to as many of the tiles as possible. 
  Remember that communication between the tiles on the IPU
  is cheaper than on other platforms!
* We haven't look at cases where the graph is bigger than can fit on the (multi)-IPU
memory yet. Presumably for bipartite graphs swapping out to `RemoteBuffer`s is
a feasible approach?
* There's probably loads of potential for optimising the implementation we discuss here!
