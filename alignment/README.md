

# Data Alignment

When data is well aligned in tile memory, `popc` can compile code to use efficient vectorised SIMD instructions
that can process, say, 4 16-bit floats in a single instruction -- a 4x speedup!
You can specify minimum alignment for vectors in vertexes using the `MinAlign` parameter in
the template. In this case, you should also specify pointer compression for more efficient use of tile memory.

e.g.

```C++
...
  Input<Vector<char, VectorLayout::SCALED_PTR32, 4>> a;
  Input<Vector<char, VectorLayout::ONE_PTR, 8>> b;
  const unsigned int b_size;
```
When you use the `VectorLayout::ONE_PTR` scheme, the `size()` method isn't
available on vectors, so you need to add another field where you set the
size during wiring.

Our default is to use  ` Input<Vector<WHATEVERTYPE, VectorLayout::ONE_PTR, 8>> ;` and
provide the size during wiring:
```C++
  auto v = graph.addVertex(computeSet, "TheVertex",
                                 {
                                         {"a", tensors["a"]},
                                         {"b", tensors["b"]}
                                 });
  graph.setInitialValue(v["b_size"], tensors["b"].shape()[0])
```
then optimise for compression scheme and perhaps smaller MinAligns later.

## Telling `popc` about `__restrict__`ed vectors
Use the `poplar::constraint` directive during Vertex class definition:
```C++
class [[poplar::constraint("elem(*in) != elem(*out)")]] MyVertex
    : public Vertex {
  Input<Vector<FLOAT, poplar::VectorLayout::ONE_PTR, 8>> in;
  Output<Vector<FLOAT, poplar::VectorLayout::ONE_PTR, 8>> out;
...
}
```

## Memory banks
The IPU has an interleaved memory bank (Region 1) and a non-interleaved bank (Region 0).
Two 64-bit aligned addresses in the interleaved bank can be accessed simultaneously, allowing
`popc` to use special instructions that allow for 128-bit loads, or a simultaneous load and store 
in one cycle (e.g. `ld2xst64pace`) can be used when the source and destination are in different banks.

You can hint to popc that data should be placed in different banks with:

```C++
class [[poplar::constraint("region(*in) != region(*out)")]] MyVertex
    : public Vertex {
  Input<Vector<FLOAT, poplar::VectorLayout::ONE_PTR, 8>> in;
  Output<Vector<FLOAT, poplar::VectorLayout::ONE_PTR, 8>> out;
...
}

```

 
## Optimised memory layout for vectors
Memory layout for vectors, vectors of vectors and jagged vectors is discussed in
https://docs.graphcore.ai/projects/assembly-programming/en/latest/vertex_vectors.html#memory-layout-for-vectors.

(NOTE: @thorbenlouw I'm still not sure how to wire up jagged vectors, it would be great to provide an example of this).

## Further reading

* https://docs.graphcore.ai/projects/assembly-programming/en/latest/asm_vertices.html?highlight=memory%20bank#memory-architecture
* https://docs.graphcore.ai/projects/poplar-api/en/latest/poplar_api.html?highlight=poplar%3A%3Aconstraint#_CPPv4N6poplar29memory_elem_constraints_errorE
* https://docs.graphcore.ai/projects/assembly-programming/en/latest/vertices_overview.html?highlight=constraint#specifying-memory-constraints