#A skeleton program for the IPU

Here we present all the code for a bare-bones Poplar IPU program that includes the following,
often-repeated steps:
* connecting to an IPU `Device`
* creating a compute `Graph`
* adding codelets to the graph
* building a compute graph 
* defining some `DataStream`s for input and output
* creating an `Engine`  
* compiling the graph
* copying data to the IPU
* running programs in the graph
* copying data back from the IPU


This is a great starting point for a project, since any useful program
will have to perform at least these steps.

In most of our code samples we have abstracted these steps away in
a common library, but here we will show them in full.

## Step 0. Includes
You'll probably want to include these common headers and use the poplar namespaces 
to reduce code verbosity. We'll also add the "popops" library headers as an 
example of including some poplar libraries (we wouldn't normally bother 
writing codelets for the things that already exist in the library and have
been heavily optimised!)

```C++
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Program.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
    
using ::poplar::FLOAT;
using ::poplar::OptionFlags;
using ::poplar::Tensor;
using ::poplar::Graph;
using ::poplar::Engine;
using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::TargetType;
using ::poplar::program::Program;
using ::poplar::program::Sequence;
using ::poplar::program::Copy;
using ::poplar::program::Repeat;
using ::poplar::program::Execute;


```

## Step 1. Connecting to an IPU device

```C++


auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        } else {
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }
    }
    return device;
}
...


auto device = getIpuDevice(1);
if (!device.has_value()) {
    std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
    return EXIT_FAILURE;
} 
```

## Step 2. Creating a compute graph and adding codelets

Here we want to add any codelets to the graph, including library
and our custom codelets. This causes their sources to be compiled. 
```C++

auto createGraphAndAddCodelets(const optional<Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());

    // Add our custom codelet, building from CPP source
    // with the given popc compiler options
    graph.addCodelets({"codelets/SkeletonCodelets.cpp"}, "-O3 -I codelets");

    // Add the codelets for the popops librarys
    popops::addCodelets(graph);
    return graph;
}
```
### The custom codelet
The custom codelet we added contains a simple `Increment` vertex, shown
below, and stored in a different source file [codeletes/SkeletonCodelets.cpp](skeleton-program/src/codelets/SkeletonCodelets.cpp)

Note that 
* the C++ for codelets is "sort-of C++11", but there is no
dynamic memory (you can't `new` or `delete`) and limited standard 
library support
* We include a bunch of useful headers (IPU-specific), which you can
  find in the SDK at `poplar-*/lib/graphcore/include/`. It's worth a peek 
  at these sources, since they're what's  available to codelets to call
  and not well documented in the Graphcore API docs.


```C++
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

class IncrementData : public Vertex {
public:
    InOut <Vector<float>> data;

    auto compute() -> bool {
        for (auto i = 0; i < data.size(); i++) {
            data[i]++;
        }
        return true;
    }
};
```


## Step 3. Building the compute graph
In this step, we define all the data upfront as `Tensor`s, and specify 
how the data will be laid out on the IPU tile memories. 
In our simple example, we just create a large 1-D Tensor of `floats`,
and spread it linearly over the tile memories. 

We find it useful to keep a map of tensor name to tensors, since 
you end up referring to tensors everywhere.

```C++
const auto NUM_DATA_ITEMS = 200000;

auto tensors = map<string, Tensor>{};
auto programs = map<string, Program>{};
...

auto buildComputeGraph(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs, const int numTiles) {
    // Add tensors
    tensors["data"] = graph.addVariable(poplar::FLOAT, {NUM_DATA_ITEMS}, "data");
    poputil::mapTensorLinearly(graph, tensors["data"]);


    // Add programs and wire up data
    const auto NumElemsPerTile = NUM_DATA_ITEMS / numTiles;
    auto cs = graph.addComputeSet("loopBody");
    for (auto tileNum = 0; tileNum < numTiles; tileNum++) {
        const auto sliceEnd = std::min((tileNum + 1) * NumElemsPerTile, (int)   NUM_DATA_ITEMS);
        const auto sliceStart = tileNum * NumElemsPerTile;

        auto v = graph.addVertex(cs, "SkeletonVertex", {
                {"data", tensors["data"].slice(sliceStart, sliceEnd)}
        });
        graph.setPerfEstimate(v, 100); // Ideally you'd get this as right as possible
        graph.setTileMapping(v, tileNum);
    }
    auto executeIncrementVertex = Execute(cs);

    auto mainProgram = Repeat(10, executeIncrementVertex, "repeat10x");
    programs["main"] = mainProgram; // Program 0 will be the main program
}
```


We also define the `Program`s which will be available 
execute on the IPU once the graph is compiled. We build 
a compute graph by composing `Sequence`s of building blocks, like `Repeat`s,
and blocks of `Execute`s of the `ComputeSet`s which define the computation.
We have to manually specify which tiles will be computing what compute sets.
We also have to "wire up" inputs and outputs to vertexes.

Our main `Program` will repeat our custom `IncrementData` vertex operation
10 times, with 2 `IncrementData`s scheduled per tile. Each vertex gets a slice
of the Tensor we created as its input/output in our example.

The compiler inserts `Copy`s for any necessary communication between tiles.
In this example we haven't been too careful about making sure that 
data is laid out on the tiles that will be computing it, so some communication
will happen between tiles. Generally, you will want to avoid unnecessary communication, but since IPULink
communication is cheap, don't optimise this prematurely.

## Step 4. Defining datastreams

Now we define "FIFOs" (DataStreams) 
that encapsulate movement to and from the host and add them to the graph. This
just creates a named handle to a FIFO.

We also add two more programs to the `Graph`, which define 
data movement from the host FIFO to the tensor on the IPU, 
and from the tensor on the IPU to the host FIFO:

```

auto defineDataStreams(Graph &graph, map<string, Tensor> &tensors, map<string, Program> &programs) {
    auto toIpuStream = graph.addHostToDeviceFIFO("TO_IPU", FLOAT, NUM_DATA_ITEMS);
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", FLOAT, NUM_DATA_ITEMS);

    auto copyToIpuProgram = Copy(toIpuStream, tensors["data"]);
    auto copyToHostProgram = Copy(tensors["data"], fromIpuStream);

    programs["copy_to_ipu"] = copyToIpuProgram;
    programs["copy_to_host"] = copyToHostProgram;
}
```

Note that we haven't actually connected these to any actual
host memory yet. We'll do that in Step 5! First we need 
to create an `Engine` that can actually execute the compute
graph we just defined.


 
## Step 5. Creating the Engine and compiling the graph
```C++
   auto ENGINE_OPTIONS = OptionFlags{
              {"target.saveArchive",                "archive.a"},
              {"debug.instrument",                  "true"},
              {"debug.instrumentCompute",           "true"},
              {"debug.loweredVarDumpFile",          "vars.capnp"},
              {"debug.instrumentControlFlow",       "true"},
              {"debug.computeInstrumentationLevel", "tile"},
              {"debug.outputAllSymbols",            "true"},
              {"autoReport.all",                    "true"},
              {"autoReport.outputSerializedGraph",  "true"},
              {"debug.retainDebugInformation",      "true"},
      };
  
      auto programIds = map<string, int>();
      auto programsList = vector<Program>(programs.size());
      int index = 0;
      for (auto &nameToProgram: programs) {
          programIds[nameToProgram.first] = index;
          programsList[index] = nameToProgram.second;
          index++;
      }
      auto engine = Engine(graph, programsList, ENGINE_OPTIONS);
```

We like to create a more flexible mapping from programs to index
in the list we pass to the engine so that we can later invoke 
programs by name, making code less brittle.

### Note:
* You can also compile the graph separately, serialize the executables,
and load the graph into an engine (see the Graphcore docs). This
avoids the need for expensive graph recompilation if you're re-running
an experiment with many different parameters
* We can pass options to the Engine during creation. We
  like to allow running in two modes: release mode has no debug and no
  profiling instrumentation. Debug mode captures information for
  debugging and inspection using the Poplar Graph Vision tools.

## Step 6: Load compiled graph onto the IPU tiles
This step just copies over the executables on the IPU, ready for
invocation.

```C++
engine.load(*device);
engine.enableExecutionProfiling();
```
## Step 7. Setting up actual data transfers
We define a host-side array of data that we will populate 
with data to send to the IPU, and which we'll fill with the
final values from the IPU. There is no Poplar here - just standard C++.

```C++
auto hostData = vector<float>(NUM_DATA_ITEMS, 0.0f);
```

Lastly, we attach the data streams we created in Step 3 to these
actual arrays on the host:
```C++
engine.connectStream("TO_IPU", hostData.data());
engine.connectStream("FROM_IPU", hostData.data());

```   

## Step 8. Running programs
To run the three programs (copy data to IPU, main program, copy data from IPU),
we just use
```C++
    std::cout << "STEP 8: Run programs" << std::endl;
    engine.run(programIds["copy_to_ipu"]); // Copy to IPU
    engine.run(programIds["main"]); // Main program
    engine.run(programIds["copy_to_host"]); // Copy from IPU
```

You can see this reflected in the Execution trace view
of the PopVision Graph Analyser tool:


![A screenshot from PopVision Graph Analyser showing the execution trace of the Skeleton program on 1 IPUs][skeleton-program-trace]

[skeleton-program-trace]: ./skeleton-execution-trace.png "Execution trace from the Skeleton program"


## Step 9: Capture debug and profile info
```C++
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

...
 engine.printProfileSummary(std::cout,
                                   OptionFlags{{"showExecutionSteps", "false"}});
```
