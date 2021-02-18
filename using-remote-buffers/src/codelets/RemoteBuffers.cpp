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
class ProcessData : public Vertex {

public:
    InOut <Vector<int>> data;

    // Just increment each value in the vector using a lot of operations, it's just an example of doing
    // something with the data
    bool compute() {
        for (auto j = 0; j < 100; j++) {
            for (auto i = 0; i < data.size(); i++) {
                data[i]++;
            }
        }
        for (auto j = 0; j < 99; j++) {
            for (auto i = 0; i < data.size(); i++) {
                data[i]--;
            }
        }
        return true;
    }
};