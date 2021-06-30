# Encouraging SIMD auto-vectorisation

`popc` is LLVM-based and so is capable of automatically applying powerful optimisations
for your codelets, especially if you follow some basic guidelines.

## 1. Specify "-O3" as  compiler option when compiling codelets
e.g.
```C++
    graph.addCodelets({"codelets/TheCodeletFilename.cpp"}, "-O3 -I codelets");
```

## 2. Add data alignments to Vectors, use VectorLayout::ONE_PTR for simple vectors
`popc` is continually improving and now automatically vectorises 
loops over Vectors without having to `reinterpret_cast` them. Still,
alignments help:


Compare the output of `-O3` compilation of:

```C++
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

```
which produces
```asm
        ld32 $a2, $m4, $m15, 0
        ld32 $a3, $m4, $m15, 1
        f32v2add $a2:3, $a2:3, $a0:1
        st32 $a3, $m4, $m15, 1
        st32step $a2, $m15, $m4+=, 2
        brnzdec $m3, .LBB0_4
```
(extract)


vs.

```C++

class SkeletonVertex2 : public Vertex {
public:
    InOut <Vector<float, VectorLayout::ONE_PTR, 8>> data;
    int size;
    float  howMuchToAdd;

    auto compute() -> bool {
        for (auto i = 0; i < size; i++) {
            data[i] += howMuchToAdd;
        }
        return true;
    }
};

```
which produces
```asm
...
        ld64 $a2:3, $m5, $m15, 0
        f32v2add $a2:3, $a0:1, $a2:3
        st64step $a2:3, $m15, $m5+=, 1
        brnzdec $m4, .LBB2_7
...
```
(extract)

The second case has been optimised to use 64-bit loads and stores (`st64` vs `st32`),
which doubles the memory bandwidth achieved. Both examples now use width-2 SIMD float vector intructions (`f32v2add`)
vs the unvectorised (`f32add`) form.


## See also:
Proper data alignment and specifying `restrict`ions on src and dest arrays, as well
as adding constraints for memory bank placement can all affect good vectorisation. See
the writeup in [Data alignment](../alignment).