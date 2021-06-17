#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <print.h>
#include <math.h>
#include <ipudef.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;
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