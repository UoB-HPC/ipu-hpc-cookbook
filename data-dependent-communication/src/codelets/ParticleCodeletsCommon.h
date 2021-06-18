#ifndef IPUSOMETHING_PARTICLECODELETESCOMMON_H
#define IPUSOMETHING_PARTICLECODELETESCOMMON_H


const auto PI = 3.141592653589793f;
const auto PARTICLE_MAX_FLOAT = 3.40282347E+38f;

constexpr auto MaxNumParticles = 1300; // Per core
constexpr auto MaxNumParticlesToShed = MaxNumParticles;

using Vector2D = struct {float x, y;};

using ParticleForForceConsideration = struct {
    Vector2D position;
};

const auto SLIM_PARTICLE_DIM = sizeof(ParticleForForceConsideration) / sizeof(float);   //  Num 32-byte words that make up a particle


using Particle = struct {
//    uint32_t id;
    Vector2D position;
    Vector2D velocity, force;
};

const auto PARTICLE_DIM = sizeof(Particle) / sizeof(float);   //  Num 32-byte words that make up a particle


using Bounds = struct {
    Vector2D min, max;
};

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


#endif //IPUSOMETHING_PARTICLECODELETESCOMMON_H
