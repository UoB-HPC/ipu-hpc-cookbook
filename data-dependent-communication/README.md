# Data-dependent communication
Poplar has pre-compiled (static) communication, meaning that every program knows
*at graph compile time* exactly how many bytes are sent from which tiles to which other tiles and 
and what time. This makes situations where communication is dynamic more
awkward to implement.

An example is in particle simulations, where the data I send to my 'neighbour' tiles
is dependent on the movement of the particles, and at each timestep I 
have variable amounts of data to send (or perhaps evenn nothing).

The below simulation is actually from a run on 16 tiles of an IPU, so it can be done 😊

![A screenshot from PopVision Graph Analyser showing the execution trace of the Append program][particle-sim]

[particle-sim]: ./particles.gif "Data-dependent communication on the IPU"


## The pattern
We demonstrate this pattern using the code example in this folder:
a simple "particle simulation" in which particles with different
velocities and trajecctories bounce around between tile memories. 
Each tile is responsible for a certain region of cartesian space,
and when a particle's trajectory takes it out of this region,
it is transferred to the appropriate tile. We don't know upfront
how much communication needs to happen, so our main program is structured like:
```C++
...
// Vertexes for particle exchange, update etc. (see full code)
Sequence exchangeParticles = ...;
Sequence findAlienParticle = ...;
Sequence updateParticlePositions ...;


// Find out whether any tile has more particles to shed
Sequence reduceHasParticlesToShed;
auto stillParticlesToShed = popops::logicalNot(
        graph,
        popops::allTrue(graph,
                       popops::logicalNot(graph, hasParticlesToShed,
                                          reduceHasParticlesToShed),
                       reduceHasParticlesToShed),
        reduceHasParticlesToShed);

// Repeatedly shed particles until nobody has more to shed
Sequence loopUntilAllParticlesExchanged = RepeatWhileTrue(
        reduceHasParticlesToShed,
        stillParticlesToShed,
        exchangeParticles
);

// Main program single timestep
Sequence timestepProgram = Sequence{
        findAlienParticle,
        loopUntilAllParticlesExchanged,
        updateParticlePositions
};
...
```

In other words, at each time step, we allow each tile to 'offer up' a
particle for redistribution to its neighbours, and neighbours check
whether the particle is for them and add it to their list. Each
tile indicates in the `hasParticlesToShed` tensor whether it has
anything left to offer. Each loop iteration begins with a reduction
of this `hasParticlesToShed` array, so that we determine whether anyone
has anything left to offer.

## Further work
* The batch size of number of particles to exchange each iteration
is a tweakable parameter and plays off the amount of global reductions
vs communication time.

* Approaches when the particles do not fit in the aggregate IPU SRAM
* Repartitioning space when one tile has too many particles

* See the attached [Slide deck](nbody-sim-on-ipu.pdf) for some more background