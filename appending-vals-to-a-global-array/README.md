# Appending values to a global array
Sometimes, during a simple experiment, you want to collect values and append them to a list as the iterations proceed.
If memory is not a issue for your experiment, it is viable to keep this array on the IPU rather than orchestrating the transfer of
values back over the PCIe link at each iteration. You can also take a hybrid approach to filling the 
array, then writing back values in chunks. In this recipe, we show you how to do this.

The main ideas are:
    
1. The empty array is pre-allocated as a `Tensor` and partitioned equally over tile memories
2. An `index` scalar `Tensor` is defined. 
3. As the loop proceeds, the index is incremented
4. At every iteration, all workers run their `AppendValue` codelet, and 
   check whether they "own" the slice of data the current `index` refers to.
   Only the "owning" worker writes the value.
5. After the operation completes, the whole `Tensor` is read back to the host 


In this example, at every timestep we perform a dummy operation to
"update the latest result" and then  use the method drscribed above to broadcast the
 latest result to every tile, and append it to the array.
 

In the execution trace below, note the `ExchangePre` phase that the compiler has introduced.
You can see that the `latestResult` variable is broadcast from tile 0 to all the other tiles
before the vertex is executed.
![A screenshot from PopVision Graph Analyser showing the execution trace of the Append program][append-pic]

[append-pic]: ./append-to-global-array.png "Zoomed in Execution trace from the Append program"



## Notes:
As an alternative, the `index` can be a 1-D Tensor with the value stored and incremented
in a distributed fashion at each tile. This uses the same amount of memory as before (since the
broadcast scalar will also take up space on each tile), but means no communication iof the index is required.

