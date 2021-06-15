# RemoteBuffers

The IPU has an extraordinary amount of the chip's transistors devoted to on-chip SRAM memory: much more than any
other chip of its size. For example, each of the 1216 cores on a MK-1 IPU has a local 256KiB on-chip memory. 
We can think of the performance of this memory as equivalent to caches in a CPU, or 
shared memory on a GPU. So it's blazingly fast, and we want to be using it as much as possible. 

But while the IPU's SRAM memories are large, they are still tiny relative to the needs of real-world 
HPC problems. We could use `DataStream`s to swap chunks of memory from host RAM to the IPU for processing, but
IPU systems offer an even better approach: `RemoteBuffer`s. You might see this referred to as _Streaming Memory_ in
some Graphcore guides.

On the MK-1 based IPU server, you can configure some of the system RAM to be unavailable to the OS kernel, and instead
have it mapped directly for use by the attached IPU devices. On the MK-2 based M2000 devices, there is additional
RAM on the device (that is not even part of the host system). Poplar can use these RAM areas (`RemoteBuffer`s) directly for storing
off-chip data, without going through the OS.

Note that a MK-1 IPU server needs to be configured to map this memory to the IPUs rather than for the kernel, 
and you should have different NUMA-affinity RAM regions 
made available to the different NUMA-affinity IPUs. The server documentation describes how to do this.

Essentially, we will be performing _manual_ cache management by explicitly bringing data from `RemoteBuffer`s onto the IPU
for processing. This data movement is compiled upfront, allowing the graph compiler to make some optimisations of the
copy operations.

## Ping Pong
In this recipe, we will be "processing" a large chunk of data (>6GiB) that is too large to fit in the IPU memories. We will be
using 2 IPUs in a "Ping Pong" configuration so that one IPU is processing while another is streaming data. This avoids
contention over the PCIe links and allows us to overlap communication and computation.

![A screenshot from PopVision Graph Analyser showing the execution trace of the Ping Pong program on 2 IPUs][remote-buffer-ping-pong]

[remote-buffer-ping-pong]: ./remote-buffer-ping-pong.png "Execution trace from the Ping Pong 2-IPU program"

In this case, our data will be a whole bunch of `int`s that we just want to increment in parallel.
Since this program is about showing the remote buffers, we don't actually do anything clever during the processing: in fact, we
just launch one worker per core. An optimised version would be using 6 workers for each of the 1216 x 2 cores, and processing
64-bit register-wide `int2`s to maximise the memory bandwidth. (i.e. we could be processing almost 30,000 `int`s in parallel
on 2 IPUs).  

We'll transfer the data to the IPU in chunks of 28,000,000 `int`s (around 106MiB). Our remote buffer will hold 32 'rows'
of this data, and we'll have 2 remote buffers. This is around 6.6GiB of data, so it's quite representative of larger
arrays. The size of `RemoteBuffer`s is limited by the global size each IPU can address (256GiB?) and the configuration of your system
(how much RAM you've set aside for remote buffers, how much you wanted to spend on Streaming Memory for your M2000 etc).
Transfers are currently limited to 128MiB, after which we must resort to doing transfers in multiple chunks.

Only one IPU can access each `RemoteBuffer`.

## Setting up `RemoteBuffer`s in the graph

The lines for adding remote buffers in the graph are very like the options for DataStreams: we specify a handle,
the type of the data, the size of a 'row' of data (1 transfer) and the number of rows in the data. In our case,
each remote buffer will 32("`NumDataRepeats`")rows of 28,000,000 (the const `NumElemsToTransfer` in the code) `int`s.
```C++
auto remoteBuffer0 = graph.addRemoteBuffer("remoteBuffer0", INT, NumElemsToTransfer, NumDataRepeats);
```

When we read from and write to the data buffer, we'll need to specify which 'row' we're writing to. That row needs to
be held in a `Tensor`. So we additionally create a scalar variable `remoteBuffer0Index` to hold this.

```C++
    auto remoteBuffer0Index = graph.addVariable(INT, {}, "offset0");
    graph.setTileMapping(remoteBuffer0Index, 0);
    graph.setInitialValue(remoteBuffer0Index, 0);
```

We can modify this tensor by passing it into custom `Vertex`es, or by using the higher-level libraries. In our example,
we increment the index using an operation

```C++
const auto increment = [&](Tensor &t) -> Sequence {
        Sequence s;
        popops::addInPlace(graph, t, One, s, "t++");
        return s;
};
...
increment(remoteBuffer1Index),

```

## Loading data from system RAM into `RemoteBuffer`s
Before we begin transferring data from the `RemoteBuffer` to the IPU, we need to initialise the values in the `RemoteBuffer`.
We can do this using the `copyToRemoteBuffer` method in the `Engine` class, which lets us 
transfer data from some system buffer into the remote buffer. Note that we write "rows" of data into the `RemoteBuffer`
rather than doing it all at once.

We start off initialising the values in each of our 'rows' to the value of the row (so all values in row 13 are `13` to begin with).

```C++ auto dataInKernelMemory = new int[NumElemsToTransfer];
    auto fillBufferWith = [&dataInKernelMemory](const int val) -> auto {
        for (int i = 0; i < (int) NumElemsToTransfer; i++) {
            dataInKernelMemory[i] = val;
        }
    };
    for (auto i = 0; i < NumDataRepeats; i++) {
        fillBufferWith(i);
        engine.copyToRemoteBuffer(dataInKernelMemory, remoteBuffer0, i);
    }
```

Similarly, when the computation is done, we need to retrieve the values from the `RemoteBuffer` and process them 
in system memory (summarise them, write the result to a file, etc.)

In our demo program, we just check the results have the expected value (row+1):
```
for (auto i = 0; i < NumDataRepeats; i++) {
        engine.copyFromRemoteBuffer(remoteBuffer0, dataInKernelMemory, i);
        ipu::assertThat("chunk " + std::to_string(i) + " remoteBuffer 0 did not have the expected value everywhere",
                        everyValueInChunkIs(i + 1));
}
```

## Transferring data between `RemoteBuffer`s and on-chip `Tensor`s in a graph program
In the graph program, we specify transfers from and to `RemoteBuffer`s using the familiar `Copy` program primitive,
but we also include a parameter for the `Tensor` representing the row offset in the `RemoteBuffer` that we want to write to
or read from.

```C++
    const auto copyFromRbToIpu0 = Copy(remoteBuffer0, data0, remoteBuffer0Index);
    const auto copyFromIpu0ToRb = Copy(data0, remoteBuffer0, remoteBuffer0Index);
```

Once the data is in an on-device `Tensor` (in this case, `data0`), we can process it as normal using custom vertexes or
Poplibs functions. Our dummy program just includes an expensive Custom codelet that increments the values using lots of
operations, allowing us to get a nice picture in the Execution trace from PopVision Graph Analyser.
