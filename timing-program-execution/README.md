#Timing program execution

You're probably familiar with the simple method
of timing program execution using usi ng some variant of:

```C++
#include <chrono>
using chrono::high_resolution_clock::now;

...
auto tic = now();
theThingYouWantToMeasure();
auto toc = now();

auto diff = chrono::duration_cast<chrono::duration<double> > (toc - tic).count();
std::cout << "theThingYouWantToMeasure took " << diff << "s" << endl;
...
```
or perhaps using libraries that use more exact performance 
counters for a specific architecture. 


While using `<chrono>` is a perfectly sensible way to time things, on the IPU, there
are a few other considerations.

## Be sure you know what you want to time
Bear in mind that there are several parts to an IPU program, and
you might not want to include them all in timings, especially if 
you're going to be comparing processor performance between
different architectures.

For example, an IPU program involves building a compute graph,
which is then compiled ( by the Graph Compiler) when you load it into the `Engine`.
The length of Graph compilation can be a nasty surprise and potentially take
tens of minutes for large programs, even if the program runs
in seconds.

There's also the (much faster) `popc` Codelet compilation when you add codelets to the graph.

Note that, for both of these forms of compilation, a 'real' application might choose to use ahead of time
compilation and just load the executables, so it's questionable whether 
these timings should be included in any program's run-time. Either way,
be clear what you're reporting!

Your program will also likely involve sending data to the IPU device
and reading back data from the device to the host, and this
data transfer is limited by PCIe speeds. Whether
you include these timings is up to you, as long as you're consistent
with what you're doing with comparison architectures like GPUs. 
Note that data transfers can be interleaved with computation on the IPUs, so
use the detailed execution traces in the PopVision Graph Analyser to 
see how they affect prograsm performance.

IPU programs can also access special buffers in host memory
 that isn't allocated to the (host) OS kernel - these are 
 called `RemoteBuffer`s. How you account for the timing
 of `RemoteBuffer` reads and writes is also worth being specific 
 about, since reads and writes are likely to be scheduled while non-
 dependent computation is happening in parallel on the IPU.
 
Invoking program runs on the IPU (e.g `engine.run(0)`) means
communicating the with IPU device, and this also takes a small amount of time. 
 
Lastly, we recommend being able to initialse your `Engine` with "debug" 
options (so that you get profiling output that you can inspect in the
PopVision Graph Analyser tools), and "release" modes (minimal options). Instrumenting
your code with profiling makes it run a bit slower, so report on "release" build
timings.
 
## A simple approach
A simple starting point for timings might be to
split your program flow into several 
`Program`s that are loaded into the `Engine`.
Only time the execution of the 'main' part of the
program:

```C++
    ...
    engine.run(programIds["loadInitialData"]);
    engine.run(programIds["initialiseSeeds"]);
    auto tic = now();
    engine.run(programIds["mainProgram"]);
    auto toc = now();
    engine.run(programIds["copyBackData"]);
    ...
```

(And you probably want to do this multiple times and average.)

After your program has taken shape, you can refine the exactness of your timing as described next.

## Using counters on the IPU
For the most precise timings, there are performance counters on the IPU. Note that each tile
has its own _independent_ hardware clock cycle counter, so you should always perform your cycle counts
on the same tile(s).

The API provides both `CycleCount`s (timing the execution of
a given program `Sequence` on a tile), and `CycleStamp`s
(recording a snapshot of the hardware counter on a tile).
These cycle counts are 64-bit `unsigned long long`s, which
come as a size-2 `Tensor` of 32-bit ints containing the lower and
upper 32-bits respectively.

Note that as of API v2.0, you should specify what type of
Sync happens before the measurement is taken:
* `INTERNAL` syncs wait for all the other tiles on the same IPU
  to reach the start of this Sync program in the BSP flow
* `EXTERNAL` syncs wait for all the IPUs in this device (e.g. an 4-IPU M2000) to 
  reach the sync point
* `GLOBAL` syncs wait for all the IPUs globally to reach the sync point

Of course the trick is that these are cycle counts and you want
_time_. You can calculate the time by using the 
clock frequency from `device->getTarget().getTileClockFrequency()`;
_*UNFORTUNATELY THIS API COULD BE LYING TO YOU (at least pre v2.0, and you should check what the current status is)*_. 
The IPU can run at lower than its max clock frequency to save power, 
and `getTileClockFrequency()` might be returning 1.6GHz when it's
actually running at 1.3GHz. Whether or not that's a
problem for you depends on your needs (arguably you could
just clock up the IPU to its max frequency, so why 
report that the IPU is slower than it could be?) 

To be sure, you can check the IPU's actual frequency on the IPU server from the command line using:

```bash
gc-inventory
```

which gives output like:
```
...
Device:
  id: 15
  target: PCIe
  Driver version: 1.0.45
  Firmware Major Version: 1
  Firmware Minor Version: 3
  Firmware Patch Version: 31
  IPU: 1
  IPU version: ipu1
  PCI Id: 0000:bb:00.0
  clock: 1300MHz     <------- THIS IS THE ONE YOU WANT
  link correctable error count: 0
  link speed: 8 GT/s
  link width: 8
  numa node: 1
  parity initialised: 0
  physical slot: PCIe Slot 9
  remote buffers: 1
  serial number: 0024.0004.919312
  sysfs file id: 15
  type: C2
```

Let's put this together in a program fragment:
```C++
    ...
    Sequence fragmentToTime;
    // Wrap the sequence with a cycle count on tile 0, performing only INTERNAL sync
    auto timing = poplar::cycleCount(graph, fragmentToTime, 0, SyncType::INTERNAL, "timer");

    // Create a "host read" (like a DataStream) to retrieve the timing tensor
    graph.createHostRead("readTimer", timing, true);
      
    unsigned long ipuTimer;
    double clockCycles = 0.;
    const auto NUM_RUNS = 5;
    for (auto run = 0; run < NUM_RUNS; run++) {
        engine.run(programIds["theMainProgramIncludingTheWrappedTimer"];
        engine.readTensor("readTimer", &ipuTimer);
        clockCycles += ipuTimer;
    }
    double clockFreq = device->getTarget().getTileClockFrequency();
    std::cout << "IPU reports " << std::fixed << clockFreq * 1e-6 << "MHz clock frequency" << std::endl;
    std::cout << "Average IPU timing for program is: " << std::fixed << std::setprecision(5) << std::setw(12)
              << clockCycles / (NUM_RUNS * 1.0) / clockFreq << "s" << std::endl;

```


## Further reading 
* https://docs.graphcore.ai/projects/poplar-api/en/latest/poplar_api.html#namespacepoplar_1abc340aac4af97e88855d17c4529294b9