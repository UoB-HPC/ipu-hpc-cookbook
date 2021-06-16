# Writing unit tests

There are some great examples of unit tests in Graphcore's open-sourced Poplibs code, for example at
https://github.com/graphcore/poplibs/tree/sdk-release-2.0/tests . These use Boost's unit testing
framework. Some of these are written to run both the IPUModel or a real IPU device.

We note that we've also used the Google Test framework to similar effect in our work.