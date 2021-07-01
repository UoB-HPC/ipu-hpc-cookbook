# Complex local data structures

## What is possible in vertexes?

The documentation at the moment is a little lacking, but if we write a program to test:

```C++
using MyOwnStructType = struct {float a, int b};

class WhatIsAllowedVertex : public Vertex {
public:

    //Inputs and outputs:

    Input<bool> aSimpleType; // OK!
    Input<MyOwnStructType> aComplexType; // NO.
    // Error: Field 'WhatIsAllowedVertex.aComplexType' has unsupported field type 'MyOwnStructType'
    Input<Vector<char>> aList; // OK
    Vector<Input<Vector<bool>>> aListOfLists; //OK
    Vector<Vector<Input<Vector<float>>>> moreThan2D; // NO.

    // fields:

    unsigned int someSimpleInitialValue; // OK
    unsigned int someArray[100]; // NO.
    // Error: Field 'WhatIsAllowedVertex.someArray' has unsupported field type 'unsigned int [100]'
    MyOwnStructType something; // NO.
    // Error: Field 'WhatIsAllowedVertex.something' has unsupported field type 'MyOwnStructType'
    char *buffer1; // NO
    // Error: Field 'WhatIsAllowedVertex.buffer1' has unsupported field type 'char *'
    static float *buffer; // OK, but also pointless without malloc/new

    bool compute() {
        // buffer = malloc(sizeof(MyOwnStructType)); // NO. can't find malloc stdlib

        buffer = new float[10000]; // NO.
        // terminate called after throwing an instance of 'poplar::link_error'
        // what(): undefined symbol _Znwm on tile 0

        auto thisIsNice = reinterpret_cast<::MyOwnStructType *>(&aList[0]);
        // Clunky but the only option!

        return true;
    }
};
```

## Our approach: `reinterpret_cast`ing `Input<Vector<char>>`
Maybe we have a 'complex' data structure something like this:
```C++

using TileData = struct {
    int numParticles;
    int nextToShed;
    int nextIndexToConsider;
    Bounds local;
    Bounds global;
    Particle particles[MaxNumParticles];
    int numProcessors;
    int myRank;
    int particlesShedThisIter;
    int particlesAcceptedThisIter;
    int offeredToMeThisIter;
};
```


```C++
inline auto asTileData(void *ref) -> TileData *const {
    return reinterpret_cast<TileData *const>(ref);
}
```


Each vertex starts off with:

```C++
class SomeVertex : public Vertex {
public:
    Input <Vector<char, VectorLayout::ONE_PTR>> data; // Just use chars and cast later
    ...
    bool compute() {
        auto tileData = asTileData(&data[0]);       
        // and use normally
        tileData->particlesShedThisIter = 0;                                      
```


* We can fill the structure host-side
* The vertex is just wired up with `char` and the buffer is pre-allocated to
  be a chunk of tile memory (static size)
  
## Dynamic memory
Tile executables don't support dynamic memory (i.e. there's no heap), and 
you can't use `malloc` or `new`. This makes implementing some structures awkward:
you essentially have to implement these using offsets into a `char` vector or,
possibly implementing a custom allocator (like https://stackoverflow.com/questions/771458/improvements-for-this-c-stack-allocator),
although we haven't done this yet.

Our strategy in some complex cases has been to grab an area of memory using an
"InOut" Vector of chars, aligned to an 8-byte boundary, and then cast that as 
above to whatever complex data structure we want to use. We grab a big chunk
and then are responsible for making sure that our structure doesn't grow bigger
than the chunk size, so have to manually keep track of its size. This approach
works fairly well when the structure can be changed in a thread-safe way and
can be combined with MultiVertexes (see [Scheduling Multiple Workers that share data](../scheduling-multiple-workers-that-share-data)),
such as for OctTrees z-ordering maps in particle simulations.

