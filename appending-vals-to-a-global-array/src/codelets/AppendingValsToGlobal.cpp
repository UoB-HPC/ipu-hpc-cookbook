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

class AppendValToGlobalArray : public Vertex {
public:
    Input<float> currentResult;
    Output <Vector<float>> results;
    int index;
    int myStartIndex;

    auto compute() -> bool {
        const bool onOrAfterStartOfMyRange = (index >= myStartIndex);
        const bool beforeEndOfMyRange = (index < myStartIndex + results.size());
        if (onOrAfterStartOfMyRange && beforeEndOfMyRange) {
            results[index - myStartIndex] = *currentResult;
        }
        index++;
        return true;
    }
};


/* A dummy placeholder for calculating the "next result" */
class CalculateNextResult: public Vertex {
public:
    Output<float> result;
    auto compute() -> bool {
        *result = *result * 1.001f;
        return true;
    }
};