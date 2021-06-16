# Unstructured neighbour lists

## Example

In this example, we'll use the graph shown below.

The graph is partitioned so that nodes 1 and 2 are assigned to worker 1,
nodes 3 and 4 to worker 2, and nodes 5 and 6 to worker 3.



At each iteration we update the node value ( a vector of dim 3) such that
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{n}_i^{t%2b1} = 0.8\mathbf{n}_i^t %2b 0.2(\sum_{j \in D}\mathbf{n}_j^te_{ij}^t)">
and the edge weights so that <img src="https://render.githubusercontent.com/render/math?math=e_{ij}^{t%2b 1} = \frac{(\mathbf{n}_i^t)^T (\mathbf{n}_j^t)}{6}">

In other words, nodes are updated according the value of nodes they are connected to, weighted with the edge
strength, and edges are redefined as a scaled dot product of the incident nodes. Basically
this is an arbitrary example: we just wanted enough complexity that we have to show
managing node and edge lists, where some nodes and edges are "owned" by other workers.

At each iteration, a worker only updates the nodes and edges it "owns", even though
it uses information from neighbouring workers' edges and nodes. Neighbouring 
node and edge values are sent from their owners at the start of every iteration.

The situation isn't helped by the fact that we can't represent complex data structures
easily in a codelet. We will need to keep lists of pointers to neighbours and edges
for each node as follows:

```
myNodes: a 2D tensor of my node values (locally renumbered)
foreignNodes: a 2D tensor of node values I use but don't own (locally renumbered)
myEdges: a 1D tensor of edge values I own 
foreignEdges: a 1D tensor of edge values I use but don't own
connectivityIndex: for each node, a structure specifying how the offset index in connectivityMap and how many edges it has
connectivityMap: A structure of node array index (and indicator whether I own or not) and edge array index (and whether I own it or not)
```


## Notes:
* Use partitioning software like ParMetis to partition graph between workers
* Just schedule 6 workers per tile (can hierarchically partition)