// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <ipudef.h>
#include <climits>
#include <print.h>
#include <math.h>

using namespace poplar;

class PiVertex : public MultiVertex {

public:
    Output<Vector<unsigned int>> hits;
    int iterations;

    auto compute(unsigned i) -> bool {
        int count = 0;
        for (auto i = 0; i < iterations; i++) {
            auto x = (float)__builtin_ipu_urand32() / UINT_MAX;
            auto y = (float)__builtin_ipu_urand32() / UINT_MAX;

            auto val = x * x + y * y;
            count +=  val < 1.f;
        }

        hits[i] = count;
        return true;
    }
};

