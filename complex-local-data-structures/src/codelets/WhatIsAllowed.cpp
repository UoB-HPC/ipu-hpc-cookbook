#include <poplar/Vertex.hpp>
#include <cstddef>
#include <cstdlib>
#include <print.h>
#include <math.h>
#include <ipudef.h>
#include <stdint.h>
#include <assert.h>
#include <cmath>

using MyOwnStructType = struct {float a, int b};

class WhatIsAllowedVertex : public Vertex {
public:

    //Inputs and outputs:

    Input<bool> aSimpleType; // OK!
    Input<MyOwnStructType> aComplexType; // NO.
    // Error: Field 'WhatIsAllowedVertex.aComplexType' has unsupported field type 'MyOwnStructType'
    Input<Vector<char>> aList; // OK
    Vector<Input<Vector<bool>>> aListOfLists; //OK
    Vector<Vector<Input<Vector<float>>>> moreThan2D; // NO.

    // fields:

    unsigned int someSimpleInitialValue; // OK
    unsigned int someArray[100]; // NO.
    // Error: Field 'WhatIsAllowedVertex.someArray' has unsupported field type 'unsigned int [100]'
    MyOwnStructType something; // NO.
    // Error: Field 'WhatIsAllowedVertex.something' has unsupported field type 'MyOwnStructType'
    char *buffer1; // NO
    // Error: Field 'WhatIsAllowedVertex.buffer1' has unsupported field type 'char *'
    static float *buffer; // OK, but also pointless without malloc/new

    bool compute() {
        // buffer = malloc(sizeof(MyOwnStructType)); // NO. can't find malloc stdlib

        buffer = new float[10000]; // NO.
        // terminate called after throwing an instance of 'poplar::link_error'
        // what(): undefined symbol _Znwm on tile 0

        auto thisIsNice = reinterpret_cast<::MyOwnStructType *>(&aList[0]);
        // Clunky but the only option!

        return true;
    }
};