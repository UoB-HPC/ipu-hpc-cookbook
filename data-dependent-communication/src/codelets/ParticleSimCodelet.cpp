#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <print.h>
#include <math.h>
#include <ipudef.h>
#include <stdint.h>
#include "ParticleCodeletsCommon.h"
#include <assert.h>


using namespace poplar;


inline auto asTileData(void *ref) -> TileData *const {
    return reinterpret_cast<TileData *const>(ref);
}


inline auto findNextToShed(TileData &tileData) -> bool {
    auto found = false;
    auto nextToShed = tileData.nextToShed - 1;

    auto particles = tileData.particles;
    while (nextToShed >= 0) {
        auto x = particles[nextToShed].position.x;
        auto y = particles[nextToShed].position.y;
        if (x < tileData.local.min.x ||
            x >= tileData.local.max.x ||
            y < tileData.local.min.y ||
            y >= tileData.local.max.y) {
            found = true;
            break;
        }
        nextToShed--;
    }
    tileData.nextToShed = nextToShed;
    return found;
}

class FindFirstAlienParticle : public Vertex {

public:
    InOut <Vector<char, VectorLayout::ONE_PTR>> data;
    Output<bool> hasParticlesToShed;

    bool compute() {
        auto tileData = asTileData(&data[0]);
        tileData->particlesShedThisIter = 0;
        tileData->particlesAcceptedThisIter = 0;
        tileData->offeredToMeThisIter = 0;
        tileData->nextToShed = tileData->numParticles;
        *hasParticlesToShed = findNextToShed(*tileData);
        return true;
    }
};


class FindNextAlienParticle : public Vertex {

public:
    InOut <Vector<char, VectorLayout::ONE_PTR>> data;
    Output<bool> hasParticlesToShed;

    bool compute() {
        auto result = false;
        auto tileData = asTileData(&data[0]);
        if (tileData->nextToShed >= 0) {
            result = findNextToShed(*tileData);
        }
        *hasParticlesToShed = result;
        return true;
    }
};


class OfferNextAlienParticle : public Vertex {
public:
    Output <Vector<float>> particleToShed;
    InOut <Vector<char, VectorLayout::ONE_PTR>> data;

    bool compute() {
        auto tileData = asTileData(&data[0]);
        auto nextToShed = tileData->nextToShed;
        if (nextToShed >= 0) {
            auto outParticle = reinterpret_cast<Particle *>(&particleToShed[0]);
            auto particles = tileData->particles;
            *outParticle = particles[tileData->nextToShed];
            // Let's move the very last item in the list here and reduce the end of the list - an O(1)
            // delete that means no defrag needed
            if (nextToShed != tileData->numParticles - 1) {
                particles[tileData->nextToShed] = particles[tileData->numParticles - 1];
            }
            tileData->numParticles--;
            tileData->particlesShedThisIter++;
        }

        return true;
    }
};

auto normTheta(float theta) -> float { // Move into range 0..2PI
    while (theta < 0) {
        theta = theta + 2 * PI;
    }
    while (theta >= 2 * PI) {
        theta = theta - 2 * PI;
    }
    return theta;
}


// We could parallelise this with workers
class CalculateNextPositions : public Vertex {
public:
    InOut <Vector<char, VectorLayout::ONE_PTR>> data;

    bool compute() {
        auto tileData = asTileData(&data[0]);

        auto signum = [](float f) -> int {
            if (f == 0.0f) return 0;
            if (f < 0) return -1;
            return 1;
        };

        const auto particles = tileData->particles;
        for (int i = 0; i < tileData->numParticles; i++) {
            // How far to move
            const auto p = particles[i].position;
            const auto v = particles[i].velocity;
            auto dx = v.x / 10;
            auto dy = v.y / 10;


            if (p.x + dx < tileData->global.min.x) { // hit the left and reflect
                dx = tileData->global.min.x - (p.x + dx);
                particles[i].velocity.x = -particles[i].velocity.x;
            }
            if (p.y + dy < tileData->global.min.y) { // hit the bottom and reflect
                dy = tileData->global.min.y - (p.y + dy);
                particles[i].velocity.y = -particles[i].velocity.y;
            }
            if (p.x + dx >= tileData->global.max.x) { // hit the right and reflect
                dx = tileData->global.max.x - ((p.x + dx) - tileData->global.max.x) - p.x;
                particles[i].velocity.x = -particles[i].velocity.x;
            }
            if (p.y + dy >= tileData->global.max.y) { // hit the top and reflect
                dy = tileData->global.max.y - ((p.y + dy) - tileData->global.max.y) - p.y;
                particles[i].velocity.y = -particles[i].velocity.y;
            }

            particles[i].position.x = p.x + dx;
            particles[i].position.y = p.y + dy;


        }

        return true;
    }
};

// We could parallelise this with workers
class AcceptAlienParticles : public Vertex {
public:
    Input <Vector<float>> potentialNewParticles;
    Input <Vector<bool>> isOfferingParticle;
    InOut <Vector<char, VectorLayout::ONE_PTR>> data;
    int numNeighbours;

    bool compute() {

        auto tileData = asTileData(&data[0]);
        const auto particles = tileData->particles;
        auto maybeParticles = reinterpret_cast<Particle *>(const_cast<float*>(&potentialNewParticles[0]));
        const auto extra_x = 0;//(tileData->local.max.x - tileData->local.min.x)*.05;
        const auto extra_y = 0;//(tileData->local.max.y - tileData->local.min.y)*.05;

        auto inBounds = [&tileData, extra_x, extra_y](const Particle &p) -> bool {
            return (p.position.x >= (tileData->local.min.x - extra_x)) &&
                   (p.position.x < (tileData->local.max.x + extra_x)) &&
                   (p.position.y >= (tileData->local.min.y - extra_y)) &&
                   (p.position.y < (tileData->local.max.y + extra_y));
        };
        for (int i = 0; i < numNeighbours; i++) {

            if (tileData->numParticles > MaxNumParticles) {
                // Oh no, should probably error or something?! For now we just lose the particle
                return true;
            }
            if (isOfferingParticle[i]) tileData->offeredToMeThisIter++;
            if (isOfferingParticle[i] && inBounds(maybeParticles[i])) {
                particles[tileData->numParticles] = maybeParticles[i];
                tileData->numParticles++;
                tileData->particlesAcceptedThisIter++;
            }
        }

        return true;
    }
};

