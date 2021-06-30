# Scheduling multiple workers that share common data
Sometimes you want to do something more sophisticated and want to control
how multiple threads access the same data, perhaps transforming a
 data structure in some way that could be done in parallel. Until recently, this was fiddly and required writing some
assembly to launch supervisor Vertexes, but since SDK v2.1 there are new `MultiVertex` worker threads
that make this task trivial.

See the docs at 
https://docs.graphcore.ai/projects/assembly-programming/en/latest/vertices_overview.html#multivertex-worker-threads
for details of `MultiVertexes`.

In this example, let's consider a data structure where we have an input array,
and another input containing the 'colour' of each item (let's pretend this
was some graph colouring/partitioning application).

We'll instantiate a `MultiVertex` where each thread only calculates 
the answer for items that match its colour, and writes back one value (the sum of
the items matching its colour) to the output array, indexed by its colour.

Note that this is a contrived example, and a terrible use of threads (you could achieve
the same with one thread in one pass), but it demonstrates the concept nicely.

```C++
class CalculateColourSum : public MultiVertex {
public:
  Input<Vector<float>> input;
  Input<Vector<unsigne int>> colours;
  Output<Vector<float>> colourSums;

  bool compute(unsigned workerId) {
    // We'll say that workerId is the colour
    const myColour = workerId;
    auto sum = 0.f;
    for (auto i = 0; i < input.size(); i++) {
        if (colours[i] == myColour) {
        sum += input[i];
    }       
    colourSums[myColour] = sum;
    return true;
  }
};
```

Scheduling a `MultiVertex` looks just like normal Vertexes:

```C++
auto computeSet = graph.addComputeSet("ComputeColourSumVtx");
// Scheduling multiple workers because it's a MultiVertex!
auto v = graph.addVertex(cs, "ComputeColourSum", {
                {"input", tensors["input"]}
        });
graph.setTileMapping(v, 0);
auto program = Execute(computeSet);
```
You don't get to choose how many threads you want to use -- 6 are scheduled. (Or more 
 robustly,
`poplar::Target::getNumWorkerContexts()`)

# IMPORTANT
With `MultiVertexes` you're responsible for maintaining thread safety
of accesses to Inputs and Outputs, so you need extra care writing these.

Make sure to read the notes on thread saftey and atomic store granularity in
https://docs.graphcore.ai/projects/assembly-programming/en/latest/vertices_overview.html#thread-safety!

If you're not careful, you could be overwriting another thread's output values
because your target output isn't aligned to the boundary of an atomic write region!


# Further Reading
* https://docs.graphcore.ai/projects/assembly-programming/en/latest/vertices_overview.html#multivertex-worker-threads
