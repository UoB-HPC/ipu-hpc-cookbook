// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "pi_options.hpp"

#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <cstdint>
#ifdef USE_DRNG
#include "immintrin.h"
#endif
/*
static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t next(uint64_t *s) {
	const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result;
}
*/
int main(int argc, char *argv[]) {

    pi_options options = parse_options(argc, argv, "Monte Carlo PI - CPU");
    uint64_t  iterations = options.iterations;
    auto precision = options.precision;
    unsigned long long count = 0;
    std::random_device dev;

    iterations -= iterations % omp_get_num_threads();

    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel reduction(+:count)
    {
        // do per-thread initialization
        //thread_local std::mt19937_64 mt_64(dev() + omp_get_thread_num());
        thread_local std::mt19937 mt(dev() + omp_get_thread_num());
        //thread_local std::uniform_int_distribution<uint64_t> dist;
        //thread_local uint64_t s[4]= {dist(mt), dist(mt), dist(mt), dist(mt)};
        uint64_t done = iterations / omp_get_num_threads();

        // then do this thread's portion of the computation:
        for (uint64_t i = 0;  i < done;  ++i) {
            std::uniform_real_distribution<float> dist(0.f, 1.f);
            auto x = dist(mt);
            auto y = dist(mt);
#ifdef USE_DRNG
            unsigned xi, yi;

            _rdrand32_step(&xi);
            _rdrand32_step(&yi);

            auto x = (float)xi/std::numeric_limits<unsigned>::max();
            auto y = (float)yi/std::numeric_limits<unsigned>::max();
#endif
/*
            auto x = (float)next(s)/std::numeric_limits<unsigned>::max();
            auto y = (float)next(s)/std::numeric_limits<unsigned>::max();
*/
            float val = x * x + y * y;
            count +=  val < 1.0;
        }
    }
    auto stop = std::chrono::steady_clock::now(); 

    std::cout << "tests = " << iterations << " took " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl; 
    std::cout << "pi = " << std::setprecision(precision) << (4. * count / iterations) << std::endl;

    return EXIT_SUCCESS;
}
