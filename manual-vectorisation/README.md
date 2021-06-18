# Manual SIMD vectorisation
You can use compiler instrinsics and the SIMD vector data types to
write manually vectorised  code when shaving off cycles or
maximally using memory bandwidth matters. But first always check
how well your code is automatically vectorising before you do this -- it
makes your code less portable and buggier.

Remember the IPU has hardware SIMD instructions for 64-bit wide operations.
So if you're using 16-bit `half` types, there are instructions which can
operate on 4 values in one instruction, potentially speeding up code 4x.

In addition, specifying 64-bit loads and stores results in more data loaded
per instruction, achieving much better memory bandwidth.

In codelets, include the file `<ipudef.h>`. It's worth inspecting this file
at `poplar_sdk/poplar-*/lib/graphcore/include/ipudef.h`. It includes
definitions for the vectorised types  e.g.


```C++
typedef int int4 __attribute__((vector_size(sizeof(int) * 4)));
```

It is also worth inspecting the math functions available in `lib/graphcore/include/__vector_math.h`
such as 
```C++
half4 half4_fma(half4 x, half4 y, half4 z);
...
float2 float2_log10(float2 x);
```

## Writing codelets with explicit vectorisation

Compare the following two codelets:
```C++
class WithoutSimd : public Vertex {
public:
    InOut <Vector<half, VectorLayout::ONE_PTR, 8>> data;
    half  howMuchToAdd;
    int size;


    auto compute() -> bool {
        for (auto i = 0; i < size; i++) {
                data[i] += howMuchToAdd;
        }
        return true;
    }
};

class WithSimd : public Vertex {
public:
    InOut <Vector<half, VectorLayout::ONE_PTR, 8>> data;
    half  howMuchToAdd;
    int size;

    auto compute() -> bool {
        auto dataAsHalf4 = reinterpret_cast<half4 *>(&data);
        for (auto i = 0; i < size / 4; i++) {
          half4 toAdd = {howMuchToAdd, howMuchToAdd, howMuchToAdd, howMuchToAdd};
          dataAsHalf4[i] += toAdd;
        }
    return true;    
    }
};
```

Without any optimisation applied (`-O0`) we see 
that the `WithoutSimd` uses 32-bit vectorised `v162add` instructions, and 32-bit loads
and stores (in conjunction with `ldb16`) for the float16 parts.

In contrast, `WithSimd` uses the 64-bit vectorised instructions and 64-bit loads and stores
such as

```asm
        ld64 $a2:3, $m1, $m15, $m2
        f16v4add $a0:1, $a2:3, $a0:1
        st64 $a0:1, $m1, $m15, $m2
```

## Should I bother, or autovectorise?

And, note that with the SDK 2.0, auto-vectorisation has improved dramatically, and
with `-O3`, the compiler produces similarly optimised

```asm
        ld64 $a2:3, $m4, $m15, 0
        f16v4add $a2:3, $a0:1, $a2:3
        st64step $a2:3, $m15, $m4+=, 1
```

So maybe only bother with manual vectorisation if the compiler doesn't do it for you,
and remember that the tools just get better and better, so as with writing assembly
vertexes, this is for squeezing the last drop of performance out of codelets!

## REMEMBER:
* Of course you still have to compensate for cases when arrays are not divisible by the
vector width! You should write tests for all your lovely new corner cases.

