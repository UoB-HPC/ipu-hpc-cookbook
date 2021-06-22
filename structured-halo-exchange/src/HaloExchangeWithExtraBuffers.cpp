#include <iostream>
#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/CycleCount.hpp>
#include <popops/AllTrue.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/codelets.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <poplar/Program.hpp>
#include <cmath>
#include <random>
#include <string>
#include "codelets/HaloExchangeCommon.h"
#include "CommonIpuUtils.hpp"
#include <exception>

using namespace std;

using namespace poplar;
using namespace poplar::program;


const auto MaxIters = 200;
const auto NumCellElements = 1; //Data structure is just 1 float per cell in this demo
const auto NumIpus = 1;
const auto TotalNumTilesToUse = 1216 * NumIpus;
const int NumWorkers = 6;

static_assert(TotalNumTilesToUse % NumIpus == 0);

auto initialiseAllTileData(char *buf, const int numProcessors) {
    auto data = reinterpret_cast<TileData *>(buf);
    for (auto i = 0; i < numProcessors; i++) {
        data[i].numRows = NumCellsInTileSide;
        data[i].numCols = NumCellsInTileSide;
    }
}

auto createAndMapTensors(Graph &graph) -> std::map<std::string, Tensor> {
    auto tensors = std::map<std::string, Tensor>{};

    auto mapNPerTile = [&](Tensor &t, int n) {
        for (auto tileNum = 0; tileNum < TotalNumTilesToUse; tileNum++) {
            graph.setTileMapping(t.slice(tileNum * n, (tileNum + 1) * n), tileNum);
        }
    };

    // The byte[] block of memory that we cast to be the structure per core that we want
    tensors["tileData"] = graph.addVariable(poplar::CHAR, {TotalNumTilesToUse, BufferSize}, "data");
    mapNPerTile(tensors["tileData"], 1);

    tensors["haloForNeighbours"] = graph.addVariable(poplar::FLOAT, {TotalNumTilesToUse, HaloSizeToNeighbours},
                                                     "haloToNeighbours");
    mapNPerTile(tensors["haloForNeighbours"], 1);
    tensors["haloFromNeighbours"] = graph.addVariable(poplar::FLOAT, {TotalNumTilesToUse, HaloSizeFromNeighbours},
                                                      "haloFromNeighbours");
    mapNPerTile(tensors["haloFromNeighbours"], 1);

    tensors["chunk"] = graph.addVariable(poplar::CHAR, {TotalNumTilesToUse, 100},
                                         "chunk");
    mapNPerTile(tensors["chunk"], 1);

    return tensors;
}


// We only wire up neighbours
auto findNeighbours(const int tileNum, const int numProcessors) -> std::map<Direction, std::optional<int>> {
    auto rowsOfTiles = (int) sqrt(numProcessors);
    auto colsOfTiles = (int) sqrt(numProcessors);
    int myRow = tileNum / colsOfTiles;
    int myCol = tileNum - myRow * colsOfTiles;
    auto north = std::optional<int>{
            myRow > 0 ?
            std::optional{(myRow - 1) * colsOfTiles + myCol} :
            std::nullopt
    };
    auto south = std::optional<int>{
            myRow < rowsOfTiles - 1 ?
            std::optional{(myRow + 1) * colsOfTiles + myCol} :
            std::nullopt
    };
    auto east = std::optional<int>{
            (myCol < colsOfTiles - 1) && (tileNum != numProcessors - 1) ?
            std::optional{myRow * colsOfTiles + myCol + 1} :
            std::nullopt
    };
    auto west = std::optional<int>{
            myCol > 0 ?
            std::optional{myRow * colsOfTiles + myCol - 1} :
            std::nullopt
    };
    auto nw = std::optional<int>{
            (myRow > 0) && (myCol > 0) ?
            std::optional{(myRow - 1) * colsOfTiles + myCol - 1} :
            std::nullopt
    };
    auto ne = std::optional<int>{
            (myRow > 0) && (myCol < colsOfTiles - 1) ?
            std::optional{(myRow - 1) * colsOfTiles + myCol + 1} :
            std::nullopt};
    auto sw = std::optional<int>{
            (myRow < rowsOfTiles - 1) && (myCol > 0) ?
            std::optional{(myRow + 1) * colsOfTiles + myCol - 1} :
            std::nullopt};
    auto se = std::optional<int>{
            (myRow < rowsOfTiles - 1) && (myCol < colsOfTiles - 1) ?
            std::optional{(myRow + 1) * colsOfTiles + myCol + 1} :
            std::nullopt};
    auto options = std::map<Direction, std::optional<int>>{
            {Directions::n,  north},
            {Directions::s,  south},
            {Directions::e,  east},
            {Directions::w,  west},
            {Directions::nw, nw},
            {Directions::ne, ne},
            {Directions::sw, sw},
            {Directions::se, se}
    };
    return options;
};

template<typename _>
auto containsKey(std::map<int, _> map, const Directions &key) -> bool {
    return map.count(key) > 0;
}


auto haloExchange(Graph &graph, std::map<std::string, Tensor> tensors) -> Sequence {
    Sequence result;

    auto packHaloCs = graph.addComputeSet("packHalo");
    for (auto tileNum = 0; tileNum < TotalNumTilesToUse; tileNum++) {
        auto v = graph.addVertex(packHaloCs, "PackHalo",
                                 {
                                         {"data", tensors["tileData"][tileNum]},
                                         {"halo", tensors["haloForNeighbours"][tileNum]}
                                 });
        graph.setPerfEstimate(v, NumCellsInTileSide * 4);
        graph.setTileMapping(v, tileNum);
    }
    result.add(Execute(packHaloCs));

    Sequence copyToN, copyToS, copyToE, copyToW, copyToNW, copyToNE, copyToSW, copyToSE;
    for (auto tileNum = 0; tileNum < TotalNumTilesToUse; tileNum++) {
        auto neighbours = findNeighbours(tileNum, TotalNumTilesToUse);
        if (neighbours[Directions::n].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToTopNeighbourHaloIndex, ToTopNeighbourHaloIndex + NumCellsInTileSide);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::n]].slice(
                    FromBottomNeighbourHaloIndex, FromBottomNeighbourHaloIndex + NumCellsInTileSide);
            copyToN.add(Copy(src, dst));
        }
        if (neighbours[Directions::s].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToBottomNeighbourHaloIndex, ToBottomNeighbourHaloIndex + NumCellsInTileSide);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::s]].slice(
                    FromTopNeighbourHaloIndex, FromTopNeighbourHaloIndex + NumCellsInTileSide);
            copyToS.add(Copy(src, dst));
        }
        if (neighbours[Directions::w].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToLeftNeighbourHaloIndex, ToLeftNeighbourHaloIndex + NumCellsInTileSide);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::w]].slice(
                    FromRightNeighbourHaloIndex, FromRightNeighbourHaloIndex + NumCellsInTileSide);
            copyToW.add(Copy(src, dst));
        }
        if (neighbours[Directions::e].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToRightNeighbourHaloIndex, ToRightNeighbourHaloIndex + NumCellsInTileSide);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::e]].slice(
                    FromLeftNeighbourHaloIndex, FromLeftNeighbourHaloIndex + NumCellsInTileSide);
            copyToE.add(Copy(src, dst));
        }
        if (neighbours[Directions::nw].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToTopLeftNeighbourHaloIndex, ToTopLeftNeighbourHaloIndex + 1);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::nw]].slice(
                    FromBottomRightNeighbourHaloIndex, FromBottomRightNeighbourHaloIndex + 1);
            copyToNW.add(Copy(src, dst));
        }
        if (neighbours[Directions::ne].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToTopRightNeighbourHaloIndex, ToTopRightNeighbourHaloIndex + 1);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::ne]].slice(
                    FromBottomLeftNeighbourHaloIndex, FromBottomLeftNeighbourHaloIndex + 1);
            copyToNE.add(Copy(src, dst));
        }
        if (neighbours[Directions::sw].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToBottomLeftNeighbourHaloIndex, ToBottomLeftNeighbourHaloIndex + 1);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::sw]].slice(
                    FromTopRightNeighbourHaloIndex, FromTopRightNeighbourHaloIndex + 1);
            copyToSW.add(Copy(src, dst));
        }
        if (neighbours[Directions::se].has_value()) {
            auto src = tensors["haloForNeighbours"][tileNum].slice(
                    ToBottomRightNeighbourHaloIndex, ToBottomRightNeighbourHaloIndex + 1);
            auto dst = tensors["haloFromNeighbours"][*neighbours[Directions::se]].slice(
                    FromTopLeftNeighbourHaloIndex, FromTopLeftNeighbourHaloIndex + 1);
            copyToSE.add(Copy(src, dst));
        }
    }
    result.add(copyToN);
    result.add(copyToNW);
    result.add(copyToW);
    result.add(copyToSW);
    result.add(copyToS);
    result.add(copyToSE);
    result.add(copyToE);
    result.add(copyToNE);

    // TODO Special cases for top and bottom of this "CHUNK"
    // (Would also be for E and W, in which case also corners!)


    auto unpackHaloCs = graph.addComputeSet("unpackHalo");
    for (auto tileNum = 0; tileNum < TotalNumTilesToUse; tileNum++) {
        auto v = graph.addVertex(unpackHaloCs, "UnpackHalo",
                                 {
                                         {"data", tensors["tileData"][tileNum]},
                                         {"halo", tensors["haloFromNeighbours"][tileNum]}
                                 });
        graph.setPerfEstimate(v, NumCellsInTileSide * 3);
        graph.setTileMapping(v, tileNum);
        v = graph.addVertex(unpackHaloCs, "UnpackHaloTop",
                            {
                                    {"data", tensors["tileData"][tileNum]},
                                    {"halo", tensors["haloFromNeighbours"][tileNum]}
                            });
        graph.setPerfEstimate(v, NumCellsInTileSide);
        graph.setTileMapping(v, tileNum);
    }
    result.add(Execute(unpackHaloCs));

    return result;
}


auto initialise(Graph &graph, std::map<std::string, Tensor> tensors) -> Sequence {
    Sequence result;
    auto initCs = graph.addComputeSet("init");
    for (auto tileNum = 0; tileNum < TotalNumTilesToUse; tileNum++) {
        auto v = graph.addVertex(initCs, "Initialise",
                                 {
                                         {"data", tensors["tileData"][tileNum]}
                                 });
        graph.setInitialValue(v["numRows"], NumCellsInTileSide);
        graph.setInitialValue(v["numCols"], NumCellsInTileSide);
        graph.setPerfEstimate(v, 2);
        graph.setTileMapping(v, tileNum);
    }
    result.add(Execute(initCs));
    return result;
}


auto stencil(Graph &graph, std::map<std::string, Tensor> tensors) -> Sequence {
    Sequence result;
    auto initCs = graph.addComputeSet("stencil");
    for (auto tileNum = 0; tileNum < TotalNumTilesToUse; tileNum++) {
        for (auto worker = 0; worker < NumWorkers; worker++) {
            auto v = graph.addVertex(initCs, "Stencil",
                                     {
                                             {"data", tensors["tileData"][tileNum]}
                                     });
            auto cellsPerWorker = NumCellsInTileSide / NumWorkers;
            auto from = cellsPerWorker * worker;
            auto to = worker == NumWorkers - 1 ? NumCellsInTileSide : from + cellsPerWorker;

            graph.setInitialValue(v["threadRowFrom"], from);
            graph.setInitialValue(v["threadRowTo"], to);
            graph.setPerfEstimate(v, NumCellsInTileSide * NumCellsInTileSide * 4 / NumWorkers);
            graph.setTileMapping(v, tileNum);
        }
    }
    result.add(Execute(initCs));
    return result;
}

int main(int argc, char *argv[]) {

//    auto device = std::optional<Device>{getIpuModel()};
    auto device = ipu::getIpuDevice(NumIpus);
    device = device->createVirtualDevice(TotalNumTilesToUse / NumIpus);
    if (!device.has_value()) {
        std::cerr << "Could not attach to IPU device. Aborting" << std::endl;
    }

    auto graph = poplar::Graph(device->getTarget());

    popops::addCodelets(graph);
    graph.addCodelets({"codelets/HaloExchangeCodelets.cpp"}, "-O3 -I codelets");

    auto tensors = createAndMapTensors(graph);

    auto dataToDevice = graph.addHostToDeviceFIFO(">>data", CHAR, BufferSize * TotalNumTilesToUse,
                                                  ReplicatedStreamMode::REPLICATE, {

//                {"bufferingDepth", "100"},
//                {"splitLimit", "0"},

                                                  });
    auto dataFromDevice = graph.addDeviceToHostFIFO("<<data", CHAR, BufferSize *
                                                                    TotalNumTilesToUse); // Maybe we want lots of different ones of these?


    auto copyBackToHost = Copy(tensors["tileData"], dataFromDevice);
    auto copyToDevice = Copy(dataToDevice, tensors["tileData"]);


    Sequence initProgram = initialise(graph, tensors);

    Sequence timestepProgram = Repeat{20, Sequence{haloExchange(graph, tensors), stencil(graph, tensors)}};

    std::cout << "Compiling..." <<
              std::endl;
    auto tic = std::chrono::high_resolution_clock::now();


    auto engine = Engine(graph, {copyToDevice, initProgram, timestepProgram, copyBackToHost},
                         ipu::POPLAR_ENGINE_OPTIONS_DEBUG);

    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    engine.load(*device);
    engine.disableExecutionProfiling();

    auto dataBuf = std::make_unique<std::vector<char>>(BufferSize * TotalNumTilesToUse * 20);

    initialiseAllTileData(dataBuf->data(), TotalNumTilesToUse);
    std::cout << "Sending initial data..." <<
              std::endl;
    engine.run(0); // Copy to device

    engine.run(1);



    for (int iter = 1; iter <= MaxIters; iter++) {
        std::cout << "Running iteration " << iter << ":" << std::endl;
        if (iter == 2) {
            engine.resetExecutionProfile();
            engine.enableExecutionProfiling();

        }
        tic = std::chrono::high_resolution_clock::now();
        engine.run(2);
        toc = std::chrono::high_resolution_clock::now();
        if (iter == 2) {
            engine.disableExecutionProfiling();
            ipu::captureProfileInfo(engine);
        }


        diff = std::chrono::duration_cast<std::chrono::duration<double >>(toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;
        engine.run(3); // Copy back

    }

    return EXIT_SUCCESS;
}

