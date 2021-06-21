# Structured halo exchange

![Halo exchange][halo-exchange]

[halo-exchange]: ./structured-halo-exchange.png "Halo exchange"



# Implicit halo exchange: the most idiomatic/elegant
* Wire up duplicated/overlapping regions to inputs if codelets to instruct
compiler to copy data for halos
* Since we can't read from and write to the same tensor region
from different vertex instances, this needs a double-buffered approach
in which we read from A and write to B in one iteration, and
then read from B and write to A in the next. This means doubling the 
memory requirement!

# Explicit halo exchange: better control and performance
* Create a distributed tensor that has extra 'padding' between the data
of partitions to store the halos
* Manually copy the borders of halos
* Group copies by direction (e.g. all "east" copies) for better performance



