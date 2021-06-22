# Structured halo exchange

For a fuller discussion of the brief notes we present here, please see Chapters 3 and 5
of Thorben Louw's MSc thesis [here](./StructuredGridHaloExchange.pdf).

![Halo exchange][halo-exchange]

[halo-exchange]: ./structured-halo-exchange.png "Halo exchange"




# Double-buffered implicit halo exchange: the most idiomatic/elegant
* Use `slice` and `concat` to wire up duplicated/overlapping regions of one Tensor to inputs of codelets to instruct
compiler to copy data for halos
* Since we can't read from and write to the same tensor region
from different vertex instances, this needs a double-buffered approach
in which we read from A and write to B in one iteration, and
then read from B and write to A in the next. This means doubling the 
memory requirement though.

This option also runs fastest!

# Double-buffered explicit halo exchange
* Create a distributed tensor that has extra 'padding' between the data
of partitions to store the halos
* Manually copy the borders of halos
* Group copies by direction (e.g. all "east" copies) for better performance
* Vertex code is much more complex

# Explicit halo exchange with copies grouped into directional 'waves'
* The order in which the explicit copies are performed has a massive performance
impact. 
* We can use existing "2-wave" optimisations to reduce the number of copies required


The difference between these  approaches is shown in [this example](src/HaloRegionApproaches.cpp),
with its [codelets](src/codelets/HaloRegionApproachesCodelets.cpp).

# In-place halo exchange: best memory use
* See [the example](src/HaloExchangeWithExtraBuffers.cpp) with its [codelets](src/codelets/HaloExchangeCodelets.cpp)
