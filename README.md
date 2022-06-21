# An HPC Cookbook for the Graphcore IPU
We've had an increasing number of questions about how we've used the
Graphcore Intelligence Processing Unit (IPU) for our HPC work. The IPU is a new platform, and
most of the help 
and documentation for it is aimed at its core application domain: Machine Learning. 
So for other workflows, such as n-body simulation or structured/unstructured
grid codes, it can be difficult to figure out how to achieve your programming aims.
We hope this repository can help you get started!

Please feel free to contribute by submitting pull requests or raising issues.

Please note that this repository will be aimed at low-level (i.e. Poplar) C++
code for the IPU. If you're looking for help with your Tensorflow or PyTorch project, it
won't be very helpful to you.


- [Table of Contents](#an-hpc-cookbook-for-the-graphcore-ipu)
  * [Basics](#basics)
    + [The Poplar SDK](#the-poplar-sdk)
    + [If you haven't got access to an IPU](#if-you-haven-t-got-access-to-an-ipu)
    + [A productive development workflow](#a-productive-development-workflow)
    + [Skeleton program](#skeleton-program)
    + [IPUModel (emulator)](#ipumodel--emulator-)
    + [Testing](#testing)
    + [Timing program execution](#timing-program-execution)
  * [Writing faster codelets](#writing-faster-codelets)
    + [Inspecting compiler output](#inspecting-compiler-output)
    + [Encouraging auto-vectorisation](#encouraging-auto-vectorisation)
    + [Manual vectorisation](#manual-vectorisation)
    + [Alignment](#alignment)
    + [Preventing data rearrangements](#preventing-data-rearrangements)
    + [Representing complex local data structures](#representing-complex-local-data-structures)
  * [Assembly vertexes](#assembly-vertexes)
    + [Including inline assembly](#including-inline-assembly)
  * [Scheduling workers](#scheduling-workers)
    + [When data naturally fits in distinct tensors](#when-data-naturally-fits-in-distinct-tensors)
    + [Sharing data structures between workers on a tile using MultiVertexes](#sharing-data-structures-between-workers-on-a-tile-using-multi-vertexes)
  * [When data is too big for the IPU: Using off-chip memory ("RemoteBuffers")](#when-data-is-too-big-for-the-ipu--using-off-chip-memory---remotebuffers--)
  * [Pattern: structured grids](#pattern--structured-grids)
    + [Halo exchange](#halo-exchange)
  * [Pattern: unstructured grids](#pattern--unstructured-grids)
    + [Neighbour lists](#neighbour-lists)
  * [General Recipes](#general-recipes)
    + [Evaluating memory bandwidth](#evaluating-memory-bandwidth)
    + [Appending values to a global distributed array](#appending-values-to-a-global-distributed-array)
    + [Efficient streaming of data from the host](#efficient-streaming-of-data-from-the-host)
    + [Pipelined wavefront execution](#pipelined-wavefront-execution)
    + [Sending variable amounts of data to neighbouring tiles](#sending-variable-amounts-of-data-to-neighbouring-tiles)
- [References](#references)
- [Acknowledgements](#acknowledgements)


## Basics

### The Poplar SDK
We assume that you've downloaded and installed the Graphcore SDK from https://www.graphcore.ai/developer,
and have read Graphcore's own excellent documentation before starting here. Especially make sure that
you've read the _Poplar and Poplibs User Guide_ (https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/),
and followed the basic Poplar tutorials at https://github.com/graphcore/tutorials.

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

[Setting up a productive workflow](productive-dev-workflow/) shows you our setup for getting 
remote IPU development working through CLion. CLion requires you to structure your projects using the CMake build system. 
We also show how to set up the Poplar PopVision analysis tools to drill down into the
performance of your IPU code.

If CMake isn't your thing, we should also say that we've used VSCode with great success, and you should use
whatever your like.

However you set yourself up, make sure that you can achieve rapid feedback cycles that allow you to quickly try out
ideas (maybe by writing a unit test that you can refer to later!).

### Skeleton program
The [Skeleton program](skeleton-program/) is a useful, bare-bones program that includes code to
connect to an IPU Device, build a compute graph,
incorporate some custom code, run this code on the IPU,
and send and receive some data. It does nothing special, but it's a good starting point for your project.

### IPUModel (emulator)
The [Targeting the IPU Model](ipu-model/) recipe shows you how to target the IPUModel instead of 
a real IPU device.


### Testing
Writing tests to validate the correctness of programs is a vital part of any serious development effort, and
provides an satisfying safety net to check for regressions. In the
[Writing Unit Tests](writing-unit-tests/) recipe we show how you can set up quick unit tests for your codelets, and run 
integration tests on both the IPUModel and on a real device.

### Timing program execution
There are some extra considerations when timing your IPU program's execution. Be sure you
know what you're measuring, and that comparisons to other platforms are valid. These concerns
are discussed in the [Timing Program Execution](timing-program-execution/) recipe.

## Writing faster codelets
If you've already managed to decompose your problem so that it can be
expressed as multiple independent workers that mostly target local memory 
and minimise communication between tiles, your program is probably already flying.
Even with naively written codelets, the massive parallelism and low memory access latency of the IPU is what makes
programs run fast compared to execution on a CPU, or even compared to a GPU when memory accesses are irregular 
or your workers are doing different things at different times.

But you can squeeze out much more performance: perhaps another 2-3x speedup from optimising
codelets, and up to another 6x by making sure you have utilised all the workers
available on cores. In this section, we explain how to go about those
optimisations.

As usual, don't optimise prematurely, or blindly. Working, robust code is 
much better than fast, wrong code. And you can waste months applying optimisations
willy-nilly: for example, trying to get better compute performance (operations/s) when your
problem is memory bandwidth-bound is pointless, and you should focus on better memory
instruction utilisation.

If you haven't heard of [Roofline Modelling](#references), it's a great tool
to help guide your optimisations.

### Inspecting compiler output
To better understand what `popc` is doing to the code you write, 
we show you how to inspect the compiler output and understand which
instructions are generated in the [Inspecting Compiler Output](inspecting-compiler-output/) recipe.

### Encouraging auto-vectorisation
Writing loops naively using Poplar's Vector abstractions can make it difficult for the LLVM-based `popc` compiler to
apply good optimisations and vectorise your code. In the [Encouraging Auto-vectorisation](encouraging-auto-vectorisation/) recipe, we 
should you how to get some better performance automatically.

### Manual vectorisation
The [Manual Vectorisation](manual-vectorisation/) recipe shows you how to use
the intrinsics and vectorised data types like `float2` and `half4` that can easily
boost a unvectorised program's execution by up to 4x.

### Alignment
The [Alignment](alignment/) recipe shows you how to align vectors of
data so that auto-vectorisation works better, and how to use directives
which tell `popc` that memory is in different banks, allowing it to
generate more efficient code.

### Preventing data rearrangements
Attaching slices of tensors that include remotely-stored elements can be an elegant way
to specify communication, but can also cause the compiler to introduce data rearrangements
that undo your careful optimisation. The [Preventing Data Rearrangements](preventing-data-rearrangements/)
recipe shows various approaches to avoiding this problem.  

### Representing complex local data structures
Poplar's codelets take scalar or (up to 2-dimensional) vector inputs,
but sometimes your data is more complex (e.g. a tree). Codelets also lack support
for dynamic (heap) memory, making familiar data structures hard to implement. In [Representing complex
local data structures](complex-local-data-structures) we look at some approaches to this problem.

## Assembly vertexes
For the ultimate control, you might want to write some parts of your
IPU in assembly. This also allows you to access hardware instructions that
might not be available to you in the Poplar C++ APIs (yet?) It's also a fun
way to indulge your curiosity about the IPU.

This seems very exciting, but first, a word of caution. None of what you write
will be portable to other platforms, and you'll need to go to extra lengths to test your code 
and make sure it's correct. Is this really the right thing for you to be doing for your application?

There are lots of assembly vertex examples in the open poplibs SDK code, and
there's the excellent [Vertex assembly programming guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/asm_vertices.html) from Graphcore.
Note that to make any real headway without guessing instruction formats, you'll need access to the IPU's ISA Specification
(<cite>[Graphcore Tile Worker ISA](#references)</cite>), which as far as we know,
isn't generally publicly available, so you need to contact Graphcore for a copy.

### Including inline assembly
The [Using inline assembly](./using-inline-assembly/) recipe shows you how you can write IPU assembly inline in a C++
vertex using the extended assembly syntax supported by LLVM. This is useful if you
want to access some hardware instructions directly, but keep most of your code in
C++. We also show you how to see the list of compiler instrinsics, which may
already cover your use case.

## Scheduling workers
The IPU has 6 hardware worker threads that run in a time-sliced fashion. By scheduling multiple vertexes on the same tile
in the same compute set, we can use task-based parallelism to hide instruction and memory latency and increase throughput
approximately 6x. The recipes in this section show you how you to use multiple workers on each tile.

### When data naturally fits in distinct tensors
The [Scheduling multiple workers per tile](scheduling-multiple-workers-per-tile/) recipe shows
you how to schedule multiple workers per tile to keep all 6 hardware worker threads busy. This
idiomatic way of scheduling is perfect when each worker can "own" its own data and
communicates with other workers using the normal mechanisms (`Copy` or wiring up overlapping tensor slices).

### Sharing data structures between workers on a tile using Multi-vertexes
Sometimes complex (e.g. graph) data is already in a tile's memory, but we want to elegantly partition the data between the 6 workers in one
compute set to "process it 6x as fast". For this, we can use a Multi-Vertex, which instantiates workers with different workerIds, but which reference a common data 
data structure. This form of scheduling is demonstrated in
[Scheduling multiple workers that share data](scheduling-multiple-workers-that-share-data/).

## When data is too big for the IPU: Using off-chip memory ("RemoteBuffers")
In the [Using RemoteBuffers](using-remote-buffers/) demo, we show how to use `RemoteBuffers` (also called "Streaming Memory" in Graphcore materials) to enable IPUs to access dedicated off-chip RAM (which is also not
managed by the host program's OS). This allows us tackle problems requiring many GiB of memory. Poplar requires
us to manage transfers from external RAM to the chip's SRAM manually - you can think of it as manual cache management
using pre-compiled data movement. It means structuring programs a little differently, but with the help of the compiler,
we can schedule co-operating IPUs to alternate between processing and transfer phases.

## Pattern: structured grids
The common parallel programming pattern of "structured grids" is 
used when data can be expressed on a computational grid where each cell knows
where its top, bottom, left, right etc. neighbour is, and can access these via memory offset calculations.
It's commonly used for stencil processing in partial differential equations, or in applications such 
as the Lattice Boltzmann Method. Each worker can be assigned  "block" of data that it operates on,
an only needs to communicate with its neighbours for data on the "borders".

### Halo exchange
When border data needs to be communicated between workers, the common method is
to use ghost cells that contain a copy of another workers' cells in a border region, and
synchronise these using a pattern known as 'halo exchange'. We show how to implement
this in the [Structured Halo Exchange](structured-halo-exchange/) recipe.


## Pattern: unstructured grids
In contrast with structured grids, _unstructured_ grids use a more complex
data structure (such as graph or sparse matrix) to describe the arbitrary connections between nodes
and the cartesian concepts such as "my left neighbour" are replaced with edge lists describing
connections between nodes.

### Neighbour lists
We demonstrate how a simple unstructured grid code can be implemented on the IPU in
[Unstructured Neighbour Lists](unstructured-neighbour-lists/).


## General Recipes

### Evaluating memory bandwidth
We used the [BabelStream](https://github.com/UoB-HPC/BabelStream) benchmark, with our implementation for Poplar [here](https://github.com/thorbenlouw/BabelStream/blob/master/PoplarKernels.cpp).
Understanding the achievable fraction of STREAM that can be obtained is
much more useful than blindly following the IPU data sheets, which claim 47TB/s memory bandwidth for the
tile SRAM memories. For a STREAM-type kernel, our achievable results are closer to 8TB/s for optimised, vectorised
C++ vertices on the Mk1 IPU.

### Appending values to a global distributed array
Sometimes you need to collect values to an array over the course of many iterations, and want to hold them in the
IPU memory rather than writing them back each iteration. We demonstrate an approach to this in
[Appending values to a global array](appending-vals-to-a-global-array/).

### Efficient streaming of data from the host
[Efficiently streaming data from the host](efficient-data-streaming/) shows how you can use callbacks to efficiently stream data from the host to the device

### Pipelined wavefront execution
An example of combining data streaming with spatio-temporal tiling to parallelise an operation using
pipelined wavefront execution is in [Pipelined Wavefront Execution](pipelined-wavefront/).

### Sending variable amounts of data to neighbouring tiles
In some simulations (e.g. particle simulations), the communication patterns are dynamic and data-driven. This seems at odds
with Poplar's compiled communication approach, but we show one way to work around this in the 
[Data-dependent Communication](data-dependent-communication) recipe.

# References
[1] Williams, Samuel, Andrew Waterman, and David Patterson. <cite>"Roofline: an insightful visual performance model for multicore architectures."</cite> Communications of the ACM 52.4 (2009): 65-76.

[2] Graphcore, Tile Worker ISA, Release 1.1.3


# Acknowledgements
Some of the examples use Jarryd Beck's [cxxopts](https://github.com/jarro2783/cxxopts) library for options parsing.

