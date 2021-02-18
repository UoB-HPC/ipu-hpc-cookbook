# ipu-hpc-cookbook
We've had an increasing number of questions about how we've used the
Graphcore Intelligence Processing Unit (IPU) for our HPC work. The IPU is a new platform, and
most of the help 
and documentation for it is aimed at its core application domain: Machine Learning. 
So for other workflows, such as n-body simulation or structured/unstructued
grid codes, it can be difficult to figure out how to achieve your programming aims.
We hope this repository can help you get started!

Please feel free to contribute by submitting pull requests or raising issues.

Please note that this repository will be aimed at low-level (i.e. Poplar) C++
code for the IPU. If you're looking for help with your Tensorflow or PyTorch project, it
won't be very helpful to you.


## Basics

### The Poplar SDK
We assume that you've downloaded and installed the Graphcore SDK from https://www.graphcore.ai/developer,
and have read Graphcore's own excellent documentation before starting here. Especially make sure that
you've read the _Poplar and Poplibs User Guide_ (https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/),
and followed the basic Poplar tutorials at https://github.com/graphcore/examples/tree/master/tutorials.

We also recommend bookmarking the API documentation (https://docs.graphcore.ai/projects/poplar-api/en/latest/): you're
going to be looking at it often!

### If you haven't got access to an IPU
Note that you can get started even if you haven't got access to an actual
IPU device. The Poplar SDK includes a very useful IPU emulator that
lets you run your programs on a CPU.

At the moment, the Poplar SDK is only officially available for specific CentOS and Ubuntu versions. 
If you haven't got access to a real IPU system, we recommend setting up a virtual machine or docker container  for your development efforts that you can
SSH into. This allows you to set up a remote development workflow similar to what you might use
for a real IPU system.

### A productive development workflow
You'll most likely be using an IPU on a remote system. We have found huge benefits in setting up a remote development 
workflow through a modern IDE like CLion or VSCode, and setting yourself up to benefit from
fast code editing, easy compilation, debugging and running of your IPU programs.

[Setting up a productive workflow](productive-dev-workflow/README.md) shows you our setup for getting 
remote IPU development working through CLion. CLion requires you to structure your projects using the CMake build system. 

If CMake isn't your thing, we should also say that we've used VSCode with great success, and you should use
whatever your like.

However you set yourself up, make sure that you can achieve rapid feedback cycles that allow you to quickly try out
ideas.

### Skeleton program
The [Skeleton program](skeleton-program/README.md) is a useful, bare-bones program that includes code to
connect to an IPU Device, build a compute graph,
incorporate some custom code, run this code on the IPU,
and send and receive some data. It does nothing special, but it's a good starting point for your project.

### IPUModel (emulator)
TODO


### Testing
Writing tests to validate the correctness of programs is a vital part of any serious development effort, and
provides an satisfying safety net to check for regressions. In the
[Testing](testing) tutorial we show how you can set up quick unit tests for your codelets, and run 
integration tests on both the IPUModel and on a real device.

### Timing program execution

## Writing faster codelets
TODO

### Inspecting compiler output
TODO

### Encouraging auto-vectorisation
TODO

### Manual vectorisation
TODO

### Alignment
TODO

### Including inline assembly
TODO

### Writing an Assembly vertex
TODO

### Preventing data rearrangements
TODO

### Representing complex local data structures
TODO


## Scheduling workers
The IPU has 6 hardware worker threads that run in a time-sliced fashion. By scheduling multiple vertexes on the same tile
in the same compute set, we can use task-based parallelism to hide instruction and memory latency and increase throughput
approximately 6x. The recipes in this section show you how you to use multiple workers on each tile.

### When data naturally fits in tensors
TODO
### Sharing data structures between workers on a tile using Supervisor Vertexes
TODO

## When data is too big for the IPU: Using off-chip memory ("RemoteBuffers")
In the [Using RemoteBuffers](using-remote-buffers/README.md) demo, we show how to use `RemoteBuffers` to enable IPUs to access dedicated off-chip RAM (which is also not
managed by the host program's OS). This allows us tackle problems requiring many GiB of memory. Poplar requires
us to manage transfers from external RAM to the chip's SRAM manually - you can think of it as manual cache management
using pre-compiled data movement. It means structuring programs a little differently, but with the help of the compiler,
we can schedule co-operating IPUs to alternate between processing and transfer phases.

## Pattern: structured grids
TODO

### Halo exchange
TODO

## Pattern: unstructured grids
TODO

### Neighbour lists

## Recipe: appending values to a global distributed array
TODO

## Recipe: efficient streaming of data from the host
This example shows how you can use callbacks to efficiently stream data from the host to the device
TODO

## Pattern: Pipelined wavefront execution
TODO
And example of combining data streaming with spatio-temporal tiling to parallelise an operation using
pipelined wavefront execution.

## Recipe: Sending variable amounts of data to neighbouring tiles
TODO