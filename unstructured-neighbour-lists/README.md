# Unstructured neighbour lists
There's probably loads of potential for optimising the implementation we discuss here. 

## Example

In this example, we'll use the simple graph (in the sense of 'unstructured mesh', not 'Poplar compute graph') shown below, and show how to solve a problem that
is partitioned over 3 "workers" (which we can say correspond to tiles for now), and
which involved updating both node and edge values based on neighbour node and edge values.

![The simple graph and its decomposition][UnstructuredGraph]

[UnstructuredGraph]: ./UnstructuredGraphForIpuCookbook.png "Unstructured graph partitioned over 3 workers"

The graph is partitioned so that nodes 1 and 2 are assigned to worker 1,
nodes 3 and 4 to worker 2, and nodes 5 and 6 to worker 3. Similarly, the 
edges {(1,2), (2,3), (2,4)} are "owned" by worker 1, edges {(3,4) and (4,6)}
are "owned" by worker 2 and edges {(1,5), (3,6), (5,6)} are owned by worker 3. 

Note that, in contrast to the structured grid example, nodes have variable numbers of neighbours, and neighbours are in no 
particular order.

In our problem, nodes have values that are a vector of 3 `floats`, which edge
weights are scalar `float`s. 

At each iteration we update the node value such that
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{n}_i^{t%2b1} = 0.8\mathbf{n}_i^t %2b 0.2(\sum_{j \in D}\mathbf{n}_j^te_{ij}^t)">
and the edge weights so that <img src="https://render.githubusercontent.com/render/math?math=e_{ij}^{t%2b 1} = \frac{(\mathbf{n}_i^t)^T (\mathbf{n}_j^t)}{6}">

In other words, nodes are updated according the value of nodes they are connected to, weighted with the edge
strength, and edges are redefined as a scaled dot product of the incident nodes. This is an arbitrary example: we just wanted enough complexity that we have to show
managing node and edge lists, where some nodes and edges are "owned" by other workers.

At each iteration, a worker only updates the nodes and edges it "owns", even though
it uses information from neighbouring workers' edges and nodes. Neighbouring 
node and edge values are sent from their owners at the start of every iteration.

## Partitioning graphs
You can use existing software such as Scotch or Metis to partition graphs onto workers (tiles, or threads on a tile).
 If you use `popsparse`, you probably have these libraries available already.
 Then we exploit the compute graph in Poplar allows us to capture the communication that will occur between iterations
an _compile_ it upfront, allowing the Graph compiler to optimise it. 

In the most extreme case you could partition down to 1 graph node per worker, but this would
be very suboptimal in terms of memory use and communication.

## What vertexes look like

Since we can't represent complex data structures like maps
easily in a codelet, we will need to keep lists of pointers to neighbours and edges
for each node as follows:

* (Input) `localNodes`: a 2D tensor of my node values (locally renumbered): 3 floats for each node
* (Input) `foreignNodes` a 2D tensor of node values I use but don't own (locally renumbered): 3 floats for each node
* (Input) `localEdges`: a 1D tensor of edge values I own 
* (Input) `foreignEdges`: a 1D tensor of edge values I use but don't own
* (Input) `connectivityIndex`: 2 ints for each local node, a specifying the offset index in `connectivityMap` and its number of edges
* (Input) `connectivityMap`: A list of unsigned ints, being tuples that represent: the node array index (and local indicator) and edge array index (and local indicator).
 The local indicator is the highest bit of the unsigned int, indicating whether this is a reference into
 the local or foreign list.
* (Output) `newLocalNodes` The updated value of locally owned nodes
* (Output) `newLocalEdgees` The updated value of locally owned edges
The global numbering is taken care of by the compute graph structure, which will specify
the `Copy`s of foreign edges and nodes into these local lists.

We will calculate the static `connectivityIndex` and `connectivityMap` for each
vertex when we build the graph.

## The compute graph
Our iterations proceed in two steps:

* Foreign list updates
* Vertex calculation

## Notes:
* Use partitioning software like ParMetis to partition graph between workers
* Just schedule 6 workers per tile (can hierarchically partition)