# Inspecting compiler output


## POPC (Codelet compilation)

One of the easiest ways to code the `popc` compiler is generating for your codelet is to look at the IPU assembly output.
You can do this by compiling a codelet file from the command line, as follows:
s:

Given a codelet file `SomeCodelet.cpp`, you can write the assembly to file with
```bash
popc -O2 SomeCodelet.cpp -S -target=ipu1 -o SomeCodelet.S 
```

(check `popc --list-targets` for the full list of targets, currently `ipu1`, `ipu2` or `cpu`)

Now check the output in `SomeCodelet.S`, which will be something like

```asm
	.text
	.allow_optimizations
	.file	"codelet-6yqjd2z0.cpp"
	.section	.text.__runCodelet_SomeVertex,"ax",@progbits
	.globl	__runCodelet_SomeVertex         # -- Begin function __runCodelet_SomeVertex
	.p2align	2
	.type	__runCodelet_SomeVertex,@function
__runCodelet_SomeVertex:                # @__runCodelet_SomeVertex
.Lfunc_begin0:
# %bb.0:
	ld32 $m0, $m13, $m15, 1
	brz $m0, .LBB0_3
# %bb.1:
	ld32 $m0, $m13, $m15, 0
	mov	$m1, $m15
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	ld32 $m2, $m0, $m15, 0
	add $m2, $m2, 1
	st32step $m2, $m15, $m0+=, 1
	add $m1, $m1, 1
	ld32 $m2, $m13, $m15, 1
	cmpult $m2, $m1, $m2
	brnz $m2, .LBB0_2
.LBB0_3:
	exitz $m15
.Lfunc_end0:
	.size	__runCodelet_SomeVertex, .Lfunc_end0-__runCodelet_SomeVertex
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_SomeVertex
	.long	.Lfunc_begin0
	.byte	0
	.section	.text.__runCodelet_SomeVertex,"ax",@progbits
                                        # -- End function
	.ident	"clang version 11.0.0 (git@phabricator.sourcevertex.net:diffusion/LLVMPROJECT/llvm-project.git 75a6ae69cf51c0c33aa5d3311d1aecd27418292c)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
```

Note that this creates the full vertex code, ready for a tile to run, including
specifying entry points and offsets of the data and code segments. The area of interest
is the entry point `__runCodelet_SomeVertex`, up to the line `exitz $m15`.

Naturally you will have to have read the assembly guide to understand this output! But even without
understanding the tile ISA fully, this is a useful way to understanding things like
*  whether vectorised instructions are being generated, which you can tell from
the use of `...v2` or `..v4` instructions like `f16v4add` or `f32v2add`
* whether the loads and stores are 32-bit or 64-bit,  as shown by the `...32` in the instruction name,
                                                                                               like `st32` or `ld64`
* whether instruction bundles are being generated (where the FP and int/memory pipelines each process an instruction in parallel), which you
  can tell from the presence of pairs of instructions in `{}`s, like 
  ```
        {
                ldb16step $a1, $m15, $m8+=, 1
                or $a0, $a0, 1006632960
        }
  ```                                                                 
* whether user functions are being inlined, etc.


Important
* Remember to use the same compiler flags you pass in your main program
* Make your release builds use more optimisation (e.g. `-O3`) and check what the compiler is doing for both debug and release builds.
* You could also look at the LLVM IR using the `popc  --emit-llvm` option

## Callbacks and logging during compilation from Poplar code (graph compilation)

During graph compilation, you can pass a number options to the graph compiler. 
See the [Further reading](#references) for more details on the options. 

For example
```C++
const auto POPLAR_ENGINE_OPTIONS_DEBUG = OptionFlags{
        {"target.saveArchive",                "archive.a"},
        {"debug.instrument",                  "true"},
        {"debug.instrumentCompute",           "true"},
        {"debug.loweredVarDumpFile",          "vars.capnp"},
        {"debug.instrumentControlFlow",       "true"},
        {"debug.computeInstrumentationLevel", "tile"},
        {"debug.outputAllSymbols",            "true"},
        {"autoReport.all",                    "true"},
        {"autoReport.outputSerializedGraph",  "true"},
        {"debug.retainDebugInformation",      "true"},
//        {"debug.runtimeVerify","true"},
//        {"debug.trace", "true"},
//        {"debug.traceFile", "debug.trace.out"},
//        {"debug.verify", "true"}
};
    const auto POPLAR_ENGINE_OPTIONS_NODEBUG = OptionFlags{};

   ...
   ProgressFunc progressFunc = {
                [](int a, int b) -> void { std::cout << "  Step " << a << " of " << b << std::endl; }};
        exe = {poplar::compileGraph(graph, programs,
                                    captureProfile ? POPLAR_ENGINE_OPTIONS_DEBUG : POPLAR_ENGINE_OPTIONS_NODEBUG,
                                    progressFunc)};
   ...
```

Note that output from the 'progress function' here is largely useless. We don't know what the
different steps of the compilation are, and we don't have anything useful to say about them!
However the other options set the levels of debug instrumentation, and specify capturing files which
can be inspected using the performance tools.

During graph compilation, you can get more useful information from the compiler by setting the log level for the compiler appropriately
```
POPLAR_LOG_LEVEL=INFO POPLAR_ENGINE_LOG_LEVEL=TRACE POPLAR_LOG_DEST=compile.log ./your_executable
```

Now in compile.log you can see detail such as
```
09:38:52.878 15288 PO:ENGINE   [I]     Starting phase: elimUnusedComputeSets
09:38:52.878 15288 PO:ENGINE   [I]     Ending phase: elimUnusedComputeSets (duration 0 ms; diff RSS: 0 MB; total RSS: 2423.74 MB)
09:38:52.878 15288 PO:ENGINE   [I]     Starting phase: smallSubGraphOptimisation
09:38:52.878 15288 PO:ENGINE   [I] Sub-graph optimisation begins
09:38:52.878 15288 PO:ENGINE   [I]       Starting phase: findRootNodeCandidates
09:38:52.889 15288 PO:ENGINE   [I]       Ending phase: findRootNodeCandidates (duration 10 ms; diff RSS: 0 MB; total RSS: 2423.74 MB)
09:38:52.889 15288 PO:ENGINE   [D] Found 1 root node candidates
09:38:52.889 15288 PO:ENGINE   [I]       Starting phase: Create adjacencies
09:38:52.923 15288 PO:ENGINE   [I]       Ending phase: Create adjacencies (duration 30 ms; diff RSS: 16.4961 MB; total RSS: 2440.24 MB)
09:38:52.923 15288 PO:ENGINE   [I]       Starting phase: tryToReplicateSubGraphs
09:38:52.923 15288 PO:ENGINE   [T] BFS begins for programId0#0 (2)
09:38:52.923 15288 PO:ENGINE   [T] BFS ends at depth 0
09:38:52.923 15288 PO:ENGINE   [D] Replicating switch on var programId0#0 (2
09:38:52.927 15288 PO:ENGINE   [D] Aborting optimisation because programId0#0 is written by a StreamCopy program
09:38:52.928 15288 PO:ENGINE   [T] Candidate invalid!
09:38:52.928 15288 PO:ENGINE   [I]       Ending phase: tryToReplicateSubGraphs (duration 4 ms; diff RSS: 0 MB; total RSS: 2440.24 MB)
09:38:52.928 15288 PO:ENGINE   [D] Phase tryToReplicateSubGraphs counters (max ms/total ms/mean ms/stddev/count):
09:38:52.928 15288 PO:ENGINE   [D]  - addNewEdges: 0, 0, 4.1e-05, 0.0, 1
09:38:52.928 15288 PO:ENGINE   [D]  - copyLateInitFields: 0, 0, 0.000503, 0.0, 1
09:38:52.928 15288 PO:ENGINE   [D]  - replicateVarsCopiesAndVertices: 0, 0, 0.00306, 0.0, 1
09:38:52.928 15288 PO:ENGINE   [D]  - replicateFieldInitializers: 0, 0, 6.6e-05, 0.0, 1
09:38:52.928 15288 PO:ENGINE   [D]  - GraphReplicator::search: 0, 0, 0.004344, 0.0, 1
09:38:52.928 15288 PO:ENGINE   [D]  - analyseProgs: 3, 3, 3.117397, 0.0, 1
09:38:52.928 15288 PO:ENGINE   [D] Scalar optimisation successfully replicated 0 variables and 0 vertices
09:38:52.932 15288 PO:ENGINE   [I]     Ending phase: smallSubGraphOptimisation (duration 53 ms; diff RSS: 16.4961 MB; total RSS: 2440.24 MB)
09:38:52.933 15288 PO:ENGINE   [I]     Starting phase: replicateConstants
09:38:52.939 15288 PO:ENGINE   [D] Replicating single region constants across 9728 tiles
09:38:52.940 15288 PO:ENGINE   [I]     Ending phase: replicateConstants (duration 7 ms; diff RSS: 1.03125 MB; total RSS: 2441.27 MB)
09:38:52.940 15288 PO:ENGINE   [I]     Starting phase: lowerControlFlow
```

## References:
* Check out the docs for setting the environment variables at 
https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/env-vars.html


* https://docs.graphcore.ai/projects/poplar-api/en/latest/poplar_api.html#poplar-engine-hpp

* https://docs.graphcore.ai/projects/poplar-api/en/latest/using_libs.html?highlight=ENGINE_OPTIONS#environment-variables
