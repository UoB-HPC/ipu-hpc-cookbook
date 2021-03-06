# Scheduling multiple workers per tile

The IPU has 6 hardware worker threads, which can be used to hide instruction and memory 
latency. This happens because most instructions complete within 6 cycles, and
optimally loading a tile with 6 workers means that, in effect, an instruction can complete
every cycle.

Scheduling multiple workers is easy: just map more vertex instances to
the same tile.

```C++
auto computeSet = graph.addComputeSet("aComputeSet");
// Scheduling 1 worker on tile 0
auto v = graph.addVertex(cs, "SomeCustomVertex", {
                {"input", tensors["input"]}
        });
graph.setTileMapping(v, 0);
auto program = Execute(computeSet);


// Scheduling 6 workers on tile 1
auto computeSet2 = graph.addComputeSet("anotherComputeSet");
for (auto tile = 0; tile < 6; tile++) {
    auto v = graph.addVertex(cs, "SomeCustomVertex", {
                    {"input", tensors["input"][tile]}
            });
    graph.setTileMapping(v, 1);
}
auto program = Execute(computeSet);
```

In this case, the input tensor will be processed about 6x as fast
with the second approach.

Note that in the above example we assumed the input tensor had some 
dimension that made sense to split along. In general, you can
partition the input tensor using `slice`. Aim for roughly equal 
slice sizes per worker, otherwise there will be load imbalance, and
workers that finish faster will just be waiting for the overloaded workers
to finish before they can all sync at the end of the BSP step.

Of course, you can add instances of different vertexes to the same compute set,
so different workers could be running different code. In this case 
checking for tile imbalance is particularly important.

There's not much problem with *oversubscribing* vertex instances to tiles either - in that
case they are just scheduled in a time-sliced fashion. But it helps if the 
number of workers is divisible by 6.

The Graph Vision tools can help you identify tile imbalance.

## Idiomatic Poplar - don't worry to much about the difference between tiles and workers
As you can see from the code above, you can use the same vertex codelets to run
with multiple workers - they're just wired with smaller slices of the same data.
There is some additional memory usage penalty to pay, though.
If you need to set up halo regions for inputs, then the compiler will
generate `Copy`s for the intra-tile memory copies between buffers for "neighbouring workers"  in the same 
tile memory (as opposed to message buffers from other tiles for halo regions from
actual neighbouring tiles).


## When to use this pattern
If that data you're processing fits in a single tile's memory, you should always aim to schedule 6 workers per tile for maximum utilisation,
especially if you've already spread other computation over other tiles. Communication between workers
on the same tile is cheaper than on remote tiles or remote IPUs (it's just a 
`memcpy`).

Scheduling multiple vertexes in this way is best used when data for each thread
is independent and can be sliced with no overhead. 

Note that, if your work is just to process some shared data structure 6 times as fast, i.e. with each
thread doing some part of the structure, or if workers could
determine for themselves based on their `workerId` what part of the shared structure is
theirs to process (a bit like in an MPI program where knowing your rank tells you
what you should be doing), then rather consider using a `MultiVertex` as
described in [Scheduling multiple workers that share data](../scheduling-multiple-workers-that-share-data).


## Notes
* Make code future-proof by using the `poplar::Target::getNumWorkerContexts()` function on the device API
rather than hard-coding for 6 workers.
* You can always write your program to start off with a version that allocates
one worker per tile, then, with very little code change, make it use all 6 workers
as an optimisation.