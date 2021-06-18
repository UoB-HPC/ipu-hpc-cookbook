# Data-dependent communication
Poplar has pre-compiled (static) communication. Every program knows
exactly how many bytes are sent from which tiles to which other tiles and 
and what time. This makes situations where communication is dynamic more
awkward to implement.

An example is in particle simulations, where the data I send to my 'neighbour' tiles
is dependent on the movement of the particles, and at each timestep I 
have variable amounts of data to send (or perhaps evenn nothing).

The below simulation is actually from a run on 16 tiles of an IPU, so it can be done 😊

![A screenshot from PopVision Graph Analyser showing the execution trace of the Append program][particle-sim]

[particle-sim]: ./particles.gif "Data-dependent communication on the IPU"


## The pattern

