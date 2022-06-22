// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <ipudef.h>
#include <climits>
#include <print.h>
#include <math.h>

using namespace poplar;

struct ContinousVertex : public MultiVertex {
    Output<Vector<float>> out;
    int iterations;
    Input<Vector<float>> c;
    Input<Vector<float>> w;

    auto compute(unsigned t) -> bool {
        float result = 0.f;

        if (c.size() != w.size())
            return false;
        for (auto i = 0; i < iterations; i++) {
            float sum = 0.f;
            for (auto j = 0; j < c.size(); j++) {
                auto v = (float)__builtin_ipu_urand32() / UINT_MAX;
                sum += -c[j] * (v - w[j]);
            }

            result +=  __builtin_expf(sum);
        }

        out[t] = result;
        return true;
    }
};

struct NAGVertex : public MultiVertex {
    Output<Vector<float>> out;
    int iterations;

    auto compute(unsigned t) -> bool {
        float result = 0.f;

        for (auto i = 0; i < iterations; i++) {
            auto v1 = (float)__builtin_ipu_urand32() / UINT_MAX;
            auto v2 = (float)__builtin_ipu_urand32() / UINT_MAX;
            auto v3 = (float)__builtin_ipu_urand32() / UINT_MAX;
            auto v4 = (float)__builtin_ipu_urand32() / UINT_MAX;

            result +=  (4 * v1 * (v3 * v3) * expf(2*v1*v3)) / ( 1 + v2 + v4) * (1 + v2 + v4);
        }

        out[t] = result;
        return true;
    }
};

struct CornerPeakVertex : public MultiVertex {
    Output<Vector<float>> out;
    int iterations;
    int dimensions;
    Input<Vector<float>> c;

    auto compute(unsigned t) -> bool {
        float result = 0.f;

        for (auto i = 0; i < iterations; i++) {
            float sum = 1.f;
            for (auto j = 0; j < dimensions; j++) {
                auto v = (float)__builtin_ipu_urand32() / UINT_MAX;
                sum += c[j] * v;
            }
            result += __builtin_powf(sum, -(dimensions + 1.));
            //result += __builtin_expf(float(-dimensions + 1.) + __builtin_logf(sum));
        }
        out[t] = result;
        return true;
    }
};

struct ProductPeakVertex : public MultiVertex {
    Output<Vector<float>> out;
    int iterations;
    int dimensions;
    Input<Vector<float>> c;
    Input<Vector<float>> w;

    auto compute(unsigned t) -> bool {
        float result = 0.f;
        if (c.size() != w.size())
            return false;

        for (auto i = 0; i < iterations; i++) {
            float prod = 1.f;
            for (auto j = 0; j < dimensions; j++) {
                auto v = (float)__builtin_ipu_urand32() / UINT_MAX;
                auto a = v - w[j];
                auto c_2 = __builtin_sqrtf(c[j]);

                prod *= 1.f/(a * a + c_2);
            }
            result += prod;
        }

        out[t] = result;;
        return true;
    }
};


