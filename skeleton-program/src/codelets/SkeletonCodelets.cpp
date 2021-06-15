#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <print.h>
#include <math.h>
#include <ipudef.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using namespace poplar;

class SkeletonVertex : public Vertex {
public:
    InOut <Vector<float>> data;
    int howMuchToAdd;

    auto compute() -> bool {
        for (auto i = 0; i < data.size(); i++) {
            data[i] += howMuchToAdd;
        }
        return true;
    }
};