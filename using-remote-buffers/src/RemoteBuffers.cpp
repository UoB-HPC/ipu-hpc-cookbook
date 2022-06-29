#include <iostream>
#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplar/Program.hpp>
#include <algorithm>

#include "CommonIpuUtils.hpp"

constexpr auto NumElemsToTransfer = 28000000;
constexpr auto NumDataRepeats = 32;

using namespace poplar;
using namespace poplar::program;

int main() {

    auto device = ipu::getIpuDevice(2);
    if (!device.has_value()) {
        std::cerr << "Could not attach to IPU device. Aborting" << std::endl;
    }

    // We'll demo remote buffers with 2 IPUs
    // IPU 0 will load data while IPU 1 processes and vice versa
    // We create 2x RemoteBuffers, one per IPU
    // PHASE     ACTION
    // STEP1     Stream 128MiB of data to IPU0
    //           Stream 128MiB of data from IPU1 back to RB
    // STEP2     Process data on IPU0 (dummy, just increment value of byte)
    //           Stream 128MiB of data from RB to IPU1
    // STEP3     Stream 128MiB of data back to remote from IPU0
    //           Process 128MiB of data on IPU1 (dummy, just increment value of byte)
    // The first and last time are slightly different (we want IPU0 and IPU1 to start staggered)


    auto graph = poplar::Graph(device->getTarget());

    graph.addCodelets({"codelets/RemoteBuffers.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);

    auto data0 = graph.addVariable(poplar::INT, {NumElemsToTransfer}, "data0");
    auto data1 = graph.addVariable(poplar::INT, {NumElemsToTransfer}, "data1");
    ipu::mapLinearlyOnOneIpu(data0, 0, *device, graph);
    ipu::mapLinearlyOnOneIpu(data1, 1, *device, graph);

    auto One = graph.addVariable(INT, {}, "1");
    graph.setTileMapping(One, 0);
    graph.setInitialValue(One, 1);

    auto remoteBuffer0Index = graph.addVariable(INT, {}, "offset0");
    auto remoteBuffer1Index = graph.addVariable(INT, {}, "offset1");
    graph.setTileMapping(remoteBuffer0Index, 0);
    auto numTilesPerIpu = device->getTarget().getNumTiles() / device->getTarget().getNumIPUs();
    graph.setTileMapping(remoteBuffer1Index, numTilesPerIpu);
    graph.setInitialValue(remoteBuffer0Index, 0);
    graph.setInitialValue(remoteBuffer0Index, 0);

    auto remoteBuffer0 = graph.addRemoteBuffer("remoteBuffer0", INT, NumElemsToTransfer, NumDataRepeats);
    auto remoteBuffer1 = graph.addRemoteBuffer("remoteBuffer1", INT, NumElemsToTransfer, NumDataRepeats);


    auto processDataProgram = [&](Tensor &data, const int ipuNum) -> auto {
        auto cs = graph.addComputeSet("processData");
        auto tileMapping = graph.getTileMapping(data);
        auto tileNum = 0;
        for (auto &tile : tileMapping) {
            for (auto chunk: tile) {
                auto from = chunk.begin();
                auto to = chunk.end();
                auto v = graph.addVertex(cs, "ProcessData", {
                        {"data", data.slice(from, to)}
                });
                graph.setTileMapping(v, tileNum);
            }
            tileNum++;
        }
        return Execute(cs);
    };

    const auto copyFromRbToIpu0 = Copy(remoteBuffer0, data0, remoteBuffer0Index);
    const auto copyFromRbToIpu1 = Copy(remoteBuffer1, data1, remoteBuffer1Index);
    const auto copyFromIpu0ToRb = Copy(data0, remoteBuffer0, remoteBuffer0Index);
    const auto copyFromIpu1ToRb = Copy(data1, remoteBuffer1, remoteBuffer1Index);
    const auto processDataOnIpu0 = Sequence{processDataProgram(data0, 0)};
    const auto processDataOnIpu1 = Sequence{processDataProgram(data1, 1)};

    const auto increment = [&](Tensor &t) -> Sequence {
        Sequence s;
        popops::addInPlace(graph, t, One, s, "t++");
        return s;
    };

    const auto program = Sequence{
            copyFromRbToIpu0,
            Sequence{processDataOnIpu0, copyFromRbToIpu1},
            Sequence{copyFromIpu0ToRb, processDataOnIpu1},
            Repeat(NumDataRepeats - 1,
                   Sequence{
                           increment(remoteBuffer0Index),
                           Sequence{copyFromRbToIpu0, copyFromIpu1ToRb},
                           increment(remoteBuffer1Index),
                           Sequence{processDataOnIpu0, copyFromRbToIpu1},
                           Sequence{copyFromIpu0ToRb, processDataOnIpu1}}
            ),
            copyFromIpu1ToRb
    };


    auto engine = ipu::prepareEngine(graph, {program}, *device);

    // We use engine.copyToRemoteBuffer() to store some initial data in the remote buffer
    // We're just going to copy 0...,1...,2...,.... to the data0 chunks and 100...,101.... etc to the data1 chunks
    std::cout << "Copy initial data to remote buffer:" << std::endl;


    auto dataInKernelMemory = new int[NumElemsToTransfer];
    auto fillBufferWith = [&dataInKernelMemory](const int val) -> auto {
        for (int i = 0; i < (int) NumElemsToTransfer; i++) {
            dataInKernelMemory[i] = val;
        }
    };
    for (auto i = 0; i < NumDataRepeats; i++) {
        fillBufferWith(i);
        engine.copyToRemoteBuffer(dataInKernelMemory, remoteBuffer0.handle(), i);
        fillBufferWith(100 + i);
        engine.copyToRemoteBuffer(dataInKernelMemory, remoteBuffer1.handle(), i);
    }

    engine.disableExecutionProfiling();

    auto timer = ipu::startTimer("Running Program");
    engine.run(0);
    ipu::endTimer(timer);

    // We use engine.copyFromRemoteBuffer() to copy the final data in the remote buffer back to kernel memory
    // And check that it's the expected value: every byte should have the value 1 now
    std::cout << "Copy final data from remote buffer and check:" << std::endl;
    const auto everyValueInChunkIs = [&](const int value) -> auto {
        return [&dataInKernelMemory, value]() -> auto {
            return std::all_of(dataInKernelMemory,
                               dataInKernelMemory + NumElemsToTransfer,
                               [value](int x) -> auto { return x == value; });
        };
    };

//    engine.printProfileSummary(std::cout,
//                               OptionFlags{
//                                       //  {"showVarStorage", "true"}
//                               }
//    );

    for (auto i = 0; i < NumDataRepeats; i++) {
        engine.copyFromRemoteBuffer(remoteBuffer0.handle(), dataInKernelMemory, i);
        ipu::assertThat("chunk " + std::to_string(i) + " remoteBuffer 0 did not have the expected value everywhere",
                        everyValueInChunkIs(i + 1));
        engine.copyFromRemoteBuffer(remoteBuffer1.handle(), dataInKernelMemory, i);
        ipu::assertThat("chunk " + std::to_string(i) + " remoteBuffer 1 did not have the expected value everywhere",
                        everyValueInChunkIs(101 + i));
    }
    //ipu::captureProfileInfo(engine);

    delete[] dataInKernelMemory;
    return EXIT_SUCCESS;
}

