

#include <cstdlib>
#include <cxxopts.hpp>
#include "StructuredGridUtils.hpp"
#include <chrono>
#include "GraphcoreUtils.hpp"
#include <poplar/IPUModel.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <poplar/Program.hpp>

#include <sstream>

constexpr auto NumTilesInIpuCol = 2u;

auto fill(Graph &graph, const Tensor &tensor, const float value, const unsigned tileNumber, ComputeSet &cs) -> void {
    auto v = graph.addVertex(cs,
                             "Fill<float>",
                             {
                                     {"result", tensor.flatten()},
                                     {"val",    value}
                             }
    );
    graph.setCycleEstimate(v, 100);
    graph.setTileMapping(v, tileNumber);
}

auto implicitStrategy(Graph &graph, const unsigned numTiles,
                      const unsigned blockSizePerTile, const unsigned numIters) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    auto in = graph.addVariable(FLOAT, {NumTilesInIpuRow * blockSizePerTile, NumTilesInIpuCol * blockSizePerTile},
                                "in");
    auto out = graph.addVariable(FLOAT, {NumTilesInIpuRow * blockSizePerTile, NumTilesInIpuCol * blockSizePerTile},
                                 "out");

    // Place the blocks of in and out on the right tiles
    auto z = std::vector<float>(blockSizePerTile, 0.f);

    auto initCs = graph.addComputeSet("init");
    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;
        auto startRowInTensor = ipuRow * blockSizePerTile;
        auto endRowInTensor = startRowInTensor + blockSizePerTile;
        auto startColInTensor = ipuCol * blockSizePerTile;
        auto endColInTensor = startColInTensor + blockSizePerTile;
        auto block = [=](const Tensor &t) -> Tensor {
            return t.slice({startRowInTensor, startColInTensor}, {endRowInTensor, endColInTensor});
        };
        graph.setTileMapping(block(in), tile);
        graph.setTileMapping(block(out), tile);
        fill(graph, block(in), (float) tile + 1, tile, initCs);
    }

    auto stencilProgram = [&]() -> Program {
        ComputeSet compute1 = graph.addComputeSet("implicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("implicitCompute2");
        for (auto tile = 0u; tile < numTiles; tile++) {


            auto ipuRow = tile / NumTilesInIpuCol;
            auto ipuCol = tile % NumTilesInIpuCol;
            auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

            auto maybeZerosVector = std::optional<Tensor>{};
            auto maybeZeroScalar = std::optional<Tensor>{};

            if (ipuRow == 0 || ipuRow == NumTilesInIpuRow - 1 || ipuCol == 0 || ipuCol == NumTilesInIpuCol - 1) {
                maybeZerosVector = {graph.addConstant(FLOAT, {blockSizePerTile}, z.data(), "{0...}")};
                graph.setTileMapping(*maybeZerosVector, tile);
                maybeZeroScalar = {graph.addConstant(FLOAT, {1, 1}, 0.f, "0")};
                graph.setTileMapping(*maybeZeroScalar, tile);
            }

            const auto block = [&](const Tensor &t, const int rowOffsetBlocks,
                                   const int colOffsetBlocks) -> Tensor {
                const auto startRow = ipuRow * blockSizePerTile + rowOffsetBlocks * blockSizePerTile;
                const auto startCol = ipuCol * blockSizePerTile + colOffsetBlocks * blockSizePerTile;
                return t.slice({startRow, startCol}, {startRow + blockSizePerTile, startCol + blockSizePerTile});
            };

            const auto n = [&](const Tensor &t) -> Tensor {
                return ipuRow > 0
                       ? block(t, -1, 0).slice({blockSizePerTile - 1, 0},
                                               {blockSizePerTile, blockSizePerTile})
                       : maybeZerosVector->reshape({1, blockSizePerTile});
            };
            const auto s = [&](const Tensor &t) -> Tensor {
                return ipuRow < NumTilesInIpuRow - 1
                       ? block(t, 1, 0).slice({0, 0},
                                              {1, blockSizePerTile})
                       : maybeZerosVector->reshape({1, blockSizePerTile});
            };
            const auto e = [&](const Tensor &t) -> Tensor {
                return ipuCol < NumTilesInIpuCol - 1
                       ? block(t, 0, 1).slice({0, 0},
                                              {blockSizePerTile, 1})
                       : maybeZerosVector->reshape({blockSizePerTile, 1});
            };
            const auto w = [&](const Tensor &t) -> Tensor {
                return ipuCol > 0
                       ? block(t, 0, -1).slice({0, blockSizePerTile - 1},
                                               {blockSizePerTile, blockSizePerTile})
                       : maybeZerosVector->reshape({blockSizePerTile, 1});
            };
            const auto nw = [&](const Tensor &t) -> Tensor {
                return ipuCol > 0 && ipuRow > 0
                       ? block(t, -1, -1)[blockSizePerTile - 1][blockSizePerTile - 1].reshape({1, 1})
                       : maybeZeroScalar->reshape({1, 1});
            };
            const auto ne = [&](const Tensor &t) -> Tensor {
                return ipuCol < NumTilesInIpuCol - 1 && ipuRow > 0
                       ? block(t, -1, 1)[blockSizePerTile - 1][0].reshape({1, 1})
                       : maybeZeroScalar->reshape({1, 1});
            };
            const auto sw = [&](const Tensor &t) -> Tensor {
                return ipuCol > 0 && ipuRow < NumTilesInIpuRow - 1
                       ? block(t, 1, -1)[0][blockSizePerTile - 1].reshape({1, 1})
                       : maybeZeroScalar->reshape({1, 1});
            };
            const auto se = [&](const Tensor &t) -> Tensor {
                return ipuCol < NumTilesInIpuCol - 1 && ipuRow < NumTilesInIpuRow - 1
                       ? block(t, 1, 1)[0][0].reshape({1, 1})
                       : maybeZeroScalar->reshape({1, 1});
            };


            const auto stitchHalos = [&](const Tensor b) -> Tensor {
                return concat({
                                      concat({nw(b), w(b), sw(b)}),
                                      concat({n(b), block(b, 0, 0), s(b)}),
                                      concat({ne(b), e(b), se(b)})
                              }, 1);
            };


            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  stitchHalos(in)},
                                             {"out", block(out, 0, 0)}
                                     }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
            v = graph.addVertex(compute2,
                                "IncludedHalosApproach<float>",
                                {
                                        {"in",  stitchHalos(out)},
                                        {"out", block(in, 0, 0)}
                                }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
        }
        return Sequence(Execute(compute1), Execute(compute2));
    };

    return {Execute(initCs), Repeat{numIters, stencilProgram()}};
}

auto explicitManyTensorStrategy(Graph &graph, const unsigned numTiles,
                                const unsigned blockSizePerTile, const unsigned numIters) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    // Place the blocks of in and out on the right tiles

    auto blocksForIncludedHalosIn = std::vector<Tensor>{numTiles};
    auto blocksForIncludedHalosOut = std::vector<Tensor>{numTiles};

    auto initialiseProgram = Sequence{};
    auto initialiseCs = graph.addComputeSet("init");

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        blocksForIncludedHalosIn[tile] = graph.addVariable(FLOAT, {blockSizePerTile + 2, blockSizePerTile + 2},
                                                           "in" + std::to_string(tile));
        blocksForIncludedHalosOut[tile] = graph.addVariable(FLOAT, {blockSizePerTile + 2, blockSizePerTile + 2},
                                                            "in" + std::to_string(tile));
        graph.setTileMapping(blocksForIncludedHalosIn[tile], tile);
        graph.setTileMapping(blocksForIncludedHalosOut[tile], tile);
        fill(graph, blocksForIncludedHalosIn[tile].slice({1, 1}, {blockSizePerTile + 1, blockSizePerTile + 1}),
             (float) tile + 1, tile, initialiseCs);
        fill(graph, blocksForIncludedHalosOut[tile].slice({1, 1}, {blockSizePerTile + 1, blockSizePerTile + 1}),
             (float) tile + 1, tile, initialiseCs);
        //  zero out the tlbr grids' halos appropriately
        if (ipuRow == 0) {
            popops::zero(graph, blocksForIncludedHalosIn[tile][0], initialiseProgram, "zeroTopHaloEdge");
            popops::zero(graph, blocksForIncludedHalosOut[tile][0], initialiseProgram, "zeroTopHaloEdge");
        }
        if (ipuRow == NumTilesInIpuRow - 1) {
            popops::zero(graph, blocksForIncludedHalosIn[tile][blockSizePerTile + 1], initialiseProgram,
                         "zeroBottomEdge");
            popops::zero(graph, blocksForIncludedHalosOut[tile][blockSizePerTile + 1], initialiseProgram,
                         "zeroBottomEdge");
        }
        if (ipuCol == 0) {
            popops::zero(graph, blocksForIncludedHalosIn[tile].slice({0, 0}, {blockSizePerTile + 2, 1}),
                         initialiseProgram,
                         "zeroLeftHaloEdge");
            popops::zero(graph, blocksForIncludedHalosOut[tile].slice({0, 0}, {blockSizePerTile + 2, 1}),
                         initialiseProgram,
                         "zeroLeftHaloEdge");
        }
        if (ipuCol == NumTilesInIpuCol - 1) {
            popops::zero(graph, blocksForIncludedHalosIn[tile].slice({0, blockSizePerTile + 1},
                                                                     {blockSizePerTile + 2, blockSizePerTile + 2}),
                         initialiseProgram, "zeroRightHaloEdge");
            popops::zero(graph, blocksForIncludedHalosOut[tile].slice({0, blockSizePerTile + 1},
                                                                      {blockSizePerTile + 2, blockSizePerTile + 2}),
                         initialiseProgram, "zeroRightHaloEdge");
        }
    }

    auto stencilProgram = [&]() -> Sequence {
        ComputeSet compute1 = graph.addComputeSet("explicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("explicitCompute2");
        auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

        const auto haloExchangeFn = [&](std::vector<Tensor> &t) -> Sequence {
            auto s = Sequence{};
            for (auto tile = 0u; tile < numTiles; tile++) {
                const auto ipuRow = tile / NumTilesInIpuCol;
                const auto ipuCol = tile % NumTilesInIpuCol;

                const auto ghostRegionWidth = (blockSizePerTile + 2);
                const auto ghostRegionHeight = (blockSizePerTile + 2);

                const auto ghostTopRow = 0;
                const auto ghostBottomRow = ghostRegionHeight - 1;
                const auto ghostLeftCol = 0;
                const auto ghostRightCol = ghostRegionWidth - 1;

                const auto northTile = tile - ipuCol;
                const auto southTile = tile + ipuCol;
                const auto eastTile = tile + 1;
                const auto westTile = tile - 1;
                const auto northWestTile = northTile - 1;
                const auto southWestTile = southTile - 1;
                const auto northEastTile = northTile + 1;
                const auto southEastTile = southTile + 1;

                const auto borderBottomRow = ghostBottomRow - 1;
                const auto borderLeftCol = ghostLeftCol + 1;
                const auto borderTopRow = ghostTopRow + 1;
                const auto borderRightCol = ghostRightCol - 1;


                // copy my north neighbour's bottom border to my top ghost
                if (ipuRow > 0) {
                    s.add(
                            Copy(t[northTile].slice({borderBottomRow, borderLeftCol},
                                                    {borderBottomRow + 1,
                                                     borderRightCol + 1}),
                                 t[tile].slice({ghostTopRow, ghostLeftCol + 1}, {ghostTopRow + 1, ghostRightCol})));

                    // copy my northEast neighbour's bottom left cell to my top right ghost cell
                    if (ipuCol < NumTilesInIpuCol - 1) {
                        s.add(Copy(t[northEastTile][borderBottomRow][borderLeftCol],
                                   t[tile][ghostTopRow][ghostRightCol]));
                    }

                    // copy my northWest neighbour's bottom right cell to my top left ghost cell
                    if (ipuCol > 0) {
                        s.add(Copy(t[northWestTile][borderBottomRow][borderRightCol],
                                   t[tile][ghostTopRow][ghostLeftCol]));
                    }


                }
                // copy my south neighbour's top border to my bottom ghost
                if (ipuRow < NumTilesInIpuRow - 1) {
                    s.add(
                            Copy(t[southTile].slice({borderTopRow, borderLeftCol},
                                                    {borderTopRow + 1,
                                                     borderRightCol + 1}),
                                 t[tile].slice({ghostBottomRow, ghostLeftCol + 1},
                                               {ghostBottomRow + 1, ghostRightCol})));

                    // copy my southEast neighbour's top left cell to my bottom right ghost cell
                    if (ipuCol < NumTilesInIpuCol - 1) {
                        s.add(Copy(t[southEastTile][borderTopRow][borderLeftCol],
                                   t[tile][ghostBottomRow][ghostRightCol]));
                    }

                    // copy my southWest neighbour's top right cell to my bottom left ghost cell
                    if (ipuCol > 0) {
                        s.add(Copy(t[southWestTile][borderTopRow][borderRightCol],
                                   t[tile][ghostBottomRow][ghostLeftCol]));
                    }
                }

//                 copy my east neighbour's left border to my right ghost
                if (ipuCol < NumTilesInIpuCol - 1) {
                    s.add(
                            Copy(t[eastTile].slice({borderTopRow, borderLeftCol},
                                                   {borderBottomRow + 1,
                                                    borderLeftCol + 1}),
                                 t[tile].slice({ghostTopRow + 1, ghostRightCol}, {ghostBottomRow, ghostRightCol + 1})));
                }
                // copy my west neighbour's right border to my left ghost region
                if (ipuCol > 0) {
                    s.add(
                            Copy(t[westTile].slice({borderTopRow, borderRightCol},
                                                   {borderBottomRow + 1,
                                                    borderRightCol + 1}),
                                 t[tile].slice({ghostTopRow + 1, ghostLeftCol}, {ghostBottomRow, ghostLeftCol + 1})));
                }

            }
            return s;
        };

        auto haloExchange1 = haloExchangeFn(blocksForIncludedHalosIn);
        auto haloExchange2 = haloExchangeFn(blocksForIncludedHalosOut);

        for (auto tile = 0u; tile < numTiles; tile++) {
            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  blocksForIncludedHalosIn[tile]},
                                             {"out", blocksForIncludedHalosOut[tile]},
                                     }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
            v = graph.addVertex(compute2,
                                "IncludedHalosApproach<float>",
                                {
                                        {"in",  blocksForIncludedHalosOut[tile]},
                                        {"out", blocksForIncludedHalosIn[tile]},
                                }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
        }


        return Sequence(haloExchange1, Execute(compute1), haloExchange2, Execute(compute2));
    };
    Sequence printTensors;
    for (auto i = 0u; i < numTiles; i++) {
        printTensors.add(PrintTensor(blocksForIncludedHalosIn[i]));
    }
    for (auto i = 0u; i < numTiles; i++) {
        printTensors.add(PrintTensor(blocksForIncludedHalosOut[i]));
    }
    return {Sequence{initialiseProgram, Execute(initialiseCs)},
            Repeat{numIters, stencilProgram()}};

}

auto explicitOneTensorStrategy2Wave(Graph &graph, const unsigned numTiles,
                                    const unsigned blockSizePerTile, const unsigned numIters) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    auto expandedIn = graph.addVariable(FLOAT,
                                        {NumTilesInIpuRow * (blockSizePerTile + 2),
                                         NumTilesInIpuCol * (blockSizePerTile + 2)},
                                        "expandedIn");
    auto expandedOut = graph.addVariable(FLOAT,
                                         {NumTilesInIpuRow * (blockSizePerTile + 2),
                                          NumTilesInIpuCol * (blockSizePerTile + 2)},
                                         "expandedOut");


    auto initialiseProgram = Sequence{};
    auto initialiseCs = graph.addComputeSet("init");

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        const auto blockWithHalo = [&](const Tensor &t) -> Tensor {
            const auto startRow = ipuRow * (blockSizePerTile + 2);
            const auto startCol = ipuCol * (blockSizePerTile + 2);
            return t.slice({startRow, startCol},
                           {startRow + blockSizePerTile + 2, startCol + blockSizePerTile + 2});
        };
        graph.setTileMapping(blockWithHalo(expandedIn), tile);
        graph.setTileMapping(blockWithHalo(expandedOut), tile);

    }
    popops::zero(graph, expandedIn, initialiseProgram);
    popops::zero(graph, expandedOut, initialiseProgram);

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        const auto block = [&](const Tensor &t) -> Tensor {
            const auto startRow = ipuRow * (blockSizePerTile + 2) + 1;
            const auto startCol = ipuCol * (blockSizePerTile + 2) + 1;
            return t.slice({startRow, startCol}, {startRow + blockSizePerTile, startCol + blockSizePerTile});
        };
        fill(graph, block(expandedIn), (float) tile + 1, tile, initialiseCs);
        fill(graph, block(expandedOut), (float) tile + 1, tile, initialiseCs);
    }

    auto stencilProgram = [&]() -> Sequence {
        ComputeSet compute1 = graph.addComputeSet("explicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("explicitCompute2");
        auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

        const auto haloExchangeFn = [&](Tensor &t) -> Sequence {
            auto northSouthWave = Sequence{};
            auto eastWestWave = Sequence{};
            for (auto tile = 0u; tile < numTiles; tile++) {
                auto ipuRow = tile / NumTilesInIpuCol;
                auto ipuCol = tile % NumTilesInIpuCol;

                const auto borderWidth = blockSizePerTile;
                const auto borderHeight = blockSizePerTile;
                const auto ghostRegionWidth = (blockSizePerTile + 2);
                const auto ghostRegionHeight = (blockSizePerTile + 2);

                const auto myGhostTopRow = ipuRow * ghostRegionHeight;
                const auto myGhostBottomRow = myGhostTopRow + ghostRegionHeight - 1;
                const auto myGhostLeftCol = ipuCol * ghostRegionWidth;
                const auto myGhostRightCol = myGhostLeftCol + ghostRegionWidth - 1;

                const auto northNeighbourBorderBottomRow = myGhostTopRow - 2;
                const auto northNeighbourBorderLeftCol = myGhostLeftCol + 1;
                const auto southNeighbourBorderTopRow = myGhostBottomRow + 2;
                const auto southNeighbourBorderLeftCol = northNeighbourBorderLeftCol;
                const auto westNeighbourBorderRightCol = myGhostLeftCol - 2;
                const auto westNeighbourBorderTopRow = myGhostTopRow + 1;
                const auto eastNeighbourBorderTopRow = myGhostTopRow + 1;
                const auto eastNeighbourBorderLeftCol = myGhostRightCol + 2;


                // copy my north neighbour's bottom border (+ one more cell) to my top ghost ( less left cell)
                if (ipuRow > 0) {
                    northSouthWave.add(
                            Copy(t.slice({northNeighbourBorderBottomRow, northNeighbourBorderLeftCol},
                                         {northNeighbourBorderBottomRow + 1,
                                          northNeighbourBorderLeftCol + borderWidth + 1}),
                                 t.slice({myGhostTopRow, myGhostLeftCol + 1},
                                         {myGhostTopRow + 1, myGhostRightCol + 1})));
                }
                // copy my south neighbour's top border (+ one more cell)  to my bottom ghost ( less left cell)
                if (ipuRow < NumTilesInIpuRow - 1) {
                    northSouthWave.add(
                            Copy(t.slice({southNeighbourBorderTopRow, southNeighbourBorderLeftCol},
                                         {southNeighbourBorderTopRow + 1,
                                          southNeighbourBorderLeftCol + borderWidth + 1}),
                                 t.slice({myGhostBottomRow, myGhostLeftCol + 1},
                                         {myGhostBottomRow + 1, myGhostRightCol + 1})));
                }

                // copy my east neighbour's left border + one cell to top and bottom to my right ghost
                if (ipuCol < NumTilesInIpuCol - 1) {
                    eastWestWave.add(
                            Copy(t.slice({eastNeighbourBorderTopRow - 1, eastNeighbourBorderLeftCol},
                                         {eastNeighbourBorderTopRow + borderHeight + 1,
                                          eastNeighbourBorderLeftCol + 1}),
                                 t.slice({myGhostTopRow, myGhostRightCol},
                                         {myGhostBottomRow + 1, myGhostRightCol + 1})));
                }
                // copy my west neighbour's right   border + one cell to top and bottom to my left ghost region
                if (ipuCol > 0) {
                    eastWestWave.add(Copy(t.slice({westNeighbourBorderTopRow - 1, westNeighbourBorderRightCol},
                                                  {westNeighbourBorderTopRow + borderHeight + 1,
                                                   westNeighbourBorderRightCol + 1}),
                                          t.slice({myGhostTopRow, myGhostLeftCol},
                                                  {myGhostBottomRow + 1, myGhostLeftCol + 1})));
                }
            }
            return Sequence(northSouthWave, eastWestWave);
        };

        auto haloExchange1 = haloExchangeFn(expandedIn);
        auto haloExchange2 = haloExchangeFn(expandedOut);

        for (auto tile = 0u; tile < numTiles; tile++) {
            auto ipuRow = tile / NumTilesInIpuCol;
            auto ipuCol = tile % NumTilesInIpuCol;

            const auto topHaloRow = ipuRow * (blockSizePerTile + 2);
            const auto bottomHaloRow = topHaloRow + blockSizePerTile + 1;
            const auto leftHaloCol = ipuCol * (blockSizePerTile + 2);
            const auto rightHaloCol = leftHaloCol + blockSizePerTile + 1;

            const auto block = [&](const Tensor &t) -> Tensor {
                return t.slice({topHaloRow, leftHaloCol}, {bottomHaloRow + 1, rightHaloCol + 1});
            };
            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  block(expandedIn)},
                                             {"out", block(expandedOut)},
                                     }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
            v = graph.addVertex(compute2,
                                "IncludedHalosApproach<float>",
                                {
                                        {"out", block(expandedIn)},
                                        {"in",  block(expandedOut)},
                                }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
        }


        return Sequence(haloExchange1, Execute(compute1), haloExchange2, Execute(compute2));
    };
    return {Sequence{initialiseProgram, Execute(initialiseCs)},
            Repeat{numIters, stencilProgram()}
    };
}


auto explicitOneTensorStrategy(Graph &graph, const unsigned numTiles,
                               const unsigned blockSizePerTile, const unsigned numIters,
                               bool groupDirs = false) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    auto expandedIn = graph.addVariable(FLOAT,
                                        {NumTilesInIpuRow * (blockSizePerTile + 2),
                                         NumTilesInIpuCol * (blockSizePerTile + 2)},
                                        "expandedIn");
    auto expandedOut = graph.addVariable(FLOAT,
                                         {NumTilesInIpuRow * (blockSizePerTile + 2),
                                          NumTilesInIpuCol * (blockSizePerTile + 2)},
                                         "expandedOut");


    auto initialiseProgram = Sequence{};
    auto initialiseCs = graph.addComputeSet("init");

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        const auto blockWithHalo = [&](const Tensor &t) -> Tensor {
            const auto startRow = ipuRow * (blockSizePerTile + 2);
            const auto startCol = ipuCol * (blockSizePerTile + 2);
            return t.slice({startRow, startCol},
                           {startRow + blockSizePerTile + 2, startCol + blockSizePerTile + 2});
        };
        graph.setTileMapping(blockWithHalo(expandedIn), tile);
        graph.setTileMapping(blockWithHalo(expandedOut), tile);

    }
    popops::zero(graph, expandedIn, initialiseProgram);
    popops::zero(graph, expandedOut, initialiseProgram);

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        const auto block = [&](const Tensor &t) -> Tensor {
            const auto startRow = ipuRow * (blockSizePerTile + 2) + 1;
            const auto startCol = ipuCol * (blockSizePerTile + 2) + 1;
            return t.slice({startRow, startCol}, {startRow + blockSizePerTile, startCol + blockSizePerTile});
        };
        fill(graph, block(expandedIn), (float) tile + 1, tile, initialiseCs);
        fill(graph, block(expandedOut), (float) tile + 1, tile, initialiseCs);
    }

    auto stencilProgram = [&]() -> Sequence {
        ComputeSet compute1 = graph.addComputeSet("explicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("explicitCompute2");
        auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

        const auto haloExchangeFn = [&](Tensor &t) -> Sequence {
            auto s = Sequence{};
            for (auto copyType = 0; copyType < 8; copyType++) {
                for (auto tile = 0u; tile < numTiles; tile++) {
                    const auto ipuRow = tile / NumTilesInIpuCol;
                    const auto ipuCol = tile % NumTilesInIpuCol;

                    const auto borderWidth = blockSizePerTile;
                    const auto borderHeight = blockSizePerTile;
                    const auto ghostRegionWidth = (blockSizePerTile + 2);
                    const auto ghostRegionHeight = (blockSizePerTile + 2);

                    const auto myGhostTopRow = ipuRow * ghostRegionHeight;
                    const auto myGhostBottomRow = myGhostTopRow + ghostRegionHeight - 1;
                    const auto myGhostLeftCol = ipuCol * ghostRegionWidth;
                    const auto myGhostRightCol = myGhostLeftCol + ghostRegionWidth - 1;

                    const auto northNeighbourBorderBottomRow = myGhostTopRow - 2;
                    const auto northNeighbourBorderLeftCol = myGhostLeftCol + 1;
                    const auto southNeighbourBorderTopRow = myGhostBottomRow + 2;
                    const auto southNeighbourBorderLeftCol = northNeighbourBorderLeftCol;
                    const auto westNeighbourBorderRightCol = myGhostLeftCol - 2;
                    const auto westNeighbourBorderTopRow = myGhostTopRow + 1;
                    const auto eastNeighbourBorderTopRow = myGhostTopRow + 1;
                    const auto eastNeighbourBorderLeftCol = myGhostRightCol + 2;
                    const auto northWestNeighbourBorderBottomRow = northNeighbourBorderBottomRow;
                    const auto northWestNeighbourBorderRightCol = northNeighbourBorderLeftCol - 2;
                    const auto northEastNeighbourBorderBottomRow = northNeighbourBorderBottomRow;
                    const auto northEastNeighbourBorderLeftCol = northNeighbourBorderLeftCol + ghostRegionWidth;
                    const auto southWestNeighbourBorderTopRow = southNeighbourBorderTopRow;
                    const auto southWestNeighbourBorderRightCol = northWestNeighbourBorderRightCol;
                    const auto southEastNeighbourBorderLeftCol = northEastNeighbourBorderLeftCol;
                    const auto southEastNeighbourBorderTopRow = southNeighbourBorderTopRow;

                    // copy my north neighbour's bottom border to my top ghost
                    if (ipuRow > 0) {
                        if (!groupDirs || copyType == 0)
                            s.add(
                                    Copy(t.slice({northNeighbourBorderBottomRow, northNeighbourBorderLeftCol},
                                                 {northNeighbourBorderBottomRow + 1,
                                                  northNeighbourBorderLeftCol + borderWidth}),
                                         t.slice({myGhostTopRow, myGhostLeftCol + 1},
                                                 {myGhostTopRow + 1, myGhostRightCol})));

                        // copy my northEast neighbour's bottom left cell to my top right ghost cell
                        if (ipuCol < NumTilesInIpuCol - 1) {
                            if (!groupDirs || copyType == 1)

                                s.add(Copy(t[northEastNeighbourBorderBottomRow][northEastNeighbourBorderLeftCol],
                                           t[myGhostTopRow][myGhostRightCol]));
                        }

                        // copy my northWest neighbour's bottom right cell to my top left ghost cell

                        if (ipuCol > 0) {
                            if (!groupDirs || copyType == 2)

                                s.add(Copy(t[northWestNeighbourBorderBottomRow][northWestNeighbourBorderRightCol],
                                           t[myGhostTopRow][myGhostLeftCol]));
                        }


                    }
                    // copy my south neighbour's top border to my bottom ghost
                    if (ipuRow < NumTilesInIpuRow - 1) {
                        if (!groupDirs || copyType == 3)

                            s.add(
                                    Copy(t.slice({southNeighbourBorderTopRow, southNeighbourBorderLeftCol},
                                                 {southNeighbourBorderTopRow + 1,
                                                  southNeighbourBorderLeftCol + borderWidth}),
                                         t.slice({myGhostBottomRow, myGhostLeftCol + 1},
                                                 {myGhostBottomRow + 1, myGhostRightCol})));

                        // copy my southEast neighbour's top left cell to my bottom right ghost cell
                        if (ipuCol < NumTilesInIpuCol - 1) {
                            if (!groupDirs || copyType == 4)

                                s.add(Copy(t[southEastNeighbourBorderTopRow][southEastNeighbourBorderLeftCol],
                                           t[myGhostBottomRow][myGhostRightCol]));
                        }

                        // copy my southWest neighbour's top right cell to my bottom left ghost cell
                        if (ipuCol > 0) {
                            if (!groupDirs || copyType == 5)

                                s.add(Copy(t[southWestNeighbourBorderTopRow][southWestNeighbourBorderRightCol],
                                           t[myGhostBottomRow][myGhostLeftCol]));
                        }
                    }

                    // copy my east neighbour's left border to my right ghost
                    if (ipuCol < NumTilesInIpuCol - 1) {
                        if (!groupDirs || copyType == 6)

                            s.add(
                                    Copy(t.slice({eastNeighbourBorderTopRow, eastNeighbourBorderLeftCol},
                                                 {eastNeighbourBorderTopRow + borderHeight,
                                                  eastNeighbourBorderLeftCol + 1}),
                                         t.slice({myGhostTopRow + 1, myGhostRightCol},
                                                 {myGhostBottomRow, myGhostRightCol + 1})));
                    }
                    // copy my west neighbour's right border to my left ghost region
                    if (ipuCol > 0) {
                        if (!groupDirs || copyType == 7)

                            s.add(Copy(t.slice({westNeighbourBorderTopRow, westNeighbourBorderRightCol},
                                               {westNeighbourBorderTopRow + borderHeight,
                                                westNeighbourBorderRightCol + 1}),
                                       t.slice({myGhostTopRow + 1, myGhostLeftCol},
                                               {myGhostBottomRow, myGhostLeftCol + 1})));
                    }


                }
                if (!groupDirs) {
                    break;
                }
            }
            return s;
        };

        auto haloExchange1 = haloExchangeFn(expandedIn);
        auto haloExchange2 = haloExchangeFn(expandedOut);

        for (auto tile = 0u; tile < numTiles; tile++) {
            auto ipuRow = tile / NumTilesInIpuCol;
            auto ipuCol = tile % NumTilesInIpuCol;

            const auto topHaloRow = ipuRow * (blockSizePerTile + 2);
            const auto bottomHaloRow = topHaloRow + blockSizePerTile + 1;
            const auto leftHaloCol = ipuCol * (blockSizePerTile + 2);
            const auto rightHaloCol = leftHaloCol + blockSizePerTile + 1;

            const auto block = [&](const Tensor &t) -> Tensor {
                return t.slice({topHaloRow, leftHaloCol}, {bottomHaloRow + 1, rightHaloCol + 1});
            };
            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  block(expandedIn)},
                                             {"out", block(expandedOut)},
                                     }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
            v = graph.addVertex(compute2,
                                "IncludedHalosApproach<float>",
                                {
                                        {"out", block(expandedIn)},
                                        {"in",  block(expandedOut)},
                                }
            );
            graph.setCycleEstimate(v, 100);
            graph.setTileMapping(v, tile);
        }


        return Sequence(haloExchange1, Execute(compute1), haloExchange2, Execute(compute2));
    };
    return {Sequence{initialiseProgram, Execute(initialiseCs)},
            Repeat{numIters, stencilProgram()}
    };
}

int main(int argc, char *argv[]) {
    unsigned numIters = 1u;
    unsigned numIpus = 1u;
    unsigned blockSizePerTile = 100;
    std::string strategy = "implicit";
    bool compileOnly = false;
    bool debug = false;
    bool useIpuModel = false;

    cxxopts::Options options(argv[0],
                             " - Prints timing for a run of a simple Moore neighbourhood average stencil ");
    options.add_options()
            ("h,halo-exhange-strategy",
             "{implicit,explicitManyTensors,explicitOneTensor,explicitOneTensor2Wave,explicitOneTensorGroupedDirs}",
             cxxopts::value<std::string>(strategy)->default_value("implicit"))
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("b,block-size", "Block size per Tile",
             cxxopts::value<unsigned>(blockSizePerTile)->default_value("100"))
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)",
             cxxopts::value<unsigned>(numIpus)->default_value("1"))
            ("d,debug", "Run in debug mode (capture profiling information)")
            ("compile-only", "Only compile the graph and write to stencil_<width>x<height>.exe, don't run")
            ("m,ipu-model", "Run on IPU model (emulator) instead of real device");

    try {
        auto opts = options.parse(argc, argv);
        debug = opts["debug"].as<bool>();
        compileOnly = opts["compile-only"].as<bool>();
        useIpuModel = opts["ipu-model"].as<bool>();
        if (opts.count("n") + opts.count("b") < 2) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
        if (!(strategy == "implicit" || strategy == "explicitManyTensors"
              || strategy == "explicitOneTensor" || strategy == "explicitOneTensor2Wave" ||
              strategy == "explicitOneTensorGroupedDirs")) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    auto device = useIpuModel ? utils::getIpuModel(numIpus) : utils::getIpuDevice(numIpus);
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    auto graph = poplar::Graph(*device);
    const auto numTiles = graph.getTarget().getNumTiles();

    std::cout << "Using " << numIpus << " IPUs for " << blockSizePerTile << "x" << blockSizePerTile
              << " blocks on each of "
              << numTiles
              << " tiles, running for " << numIters << " iterations using the " << strategy << " strategy" << ". ("
              << (blockSizePerTile * blockSizePerTile * 4 * numTiles * 2.f) / 1024.f / 1024.f
              << "MB min memory required)" <<
              std::endl;

    std::cout << "Building graph";
    auto tic = std::chrono::high_resolution_clock::now();


    auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;
    assert(NumTilesInIpuCol * NumTilesInIpuRow == numTiles);

    graph.addCodelets("codelets/HaloRegionApproachesCodelets.cpp");
    popops::addCodelets(graph);


    auto programs = std::vector<Program>{};
    if (strategy == "implicit") {
        programs = implicitStrategy(graph, numTiles, blockSizePerTile, numIters);
    } else if (strategy == "explicitManyTensors") {
        programs = explicitManyTensorStrategy(graph, numTiles, blockSizePerTile, numIters);
    } else if (strategy == "explicitOneTensor") {
        programs = explicitOneTensorStrategy(graph, numTiles, blockSizePerTile, numIters, false);
    } else if (strategy == "explicitOneTensorGroupedDirs") {
        programs = explicitOneTensorStrategy(graph, numTiles, blockSizePerTile, numIters, true);
    } else if (strategy == "explicitOneTensor2Wave") {
        programs = explicitOneTensorStrategy2Wave(graph, numTiles, blockSizePerTile, numIters);
    } else {
        return EXIT_FAILURE;
    }


    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    if (debug) {
        utils::serializeGraph(graph);
    }
    if (auto dumpGraphVisualisations =
                std::getenv("DUMP_GRAPH_VIZ") != nullptr;  dumpGraphVisualisations) {
        ofstream vertexGraph;
        vertexGraph.open("vertexgraph.dot");
        graph.outputVertexGraph(vertexGraph,
                                programs);
        vertexGraph.close();

        ofstream computeGraph;
        computeGraph.open("computegraph.dot");
        graph.outputComputeGraph(computeGraph,
                                 programs);
        computeGraph.close();
    }
    std::cout << "Compiling graph";
    tic = std::chrono::high_resolution_clock::now();
    if (compileOnly) {
        auto exe = poplar::compileGraph(graph, programs, debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG
                                                               : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);
        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;

        const auto filename = "graph.exe";
        ofstream exe_file;
        exe_file.open(filename);
        exe.serialize(exe_file);
        exe_file.close();

        return EXIT_SUCCESS;
    } else {
        auto engine = Engine(graph, programs,
                             utils::POPLAR_ENGINE_OPTIONS_DEBUG);

        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;

        engine.load(*device);

        //  engine.run(0);

        utils::timedStep("Running halo exchange iterations", [&]() -> void {
            engine.run(1);
        });


        if (debug) {
            utils::captureProfileInfo(engine);

            engine.printProfileSummary(std::cout,
                                       OptionFlags{{"showExecutionSteps", "false"}});
        }
    }

    return EXIT_SUCCESS;
}
