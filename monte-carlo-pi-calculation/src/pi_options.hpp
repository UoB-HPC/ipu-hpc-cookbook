// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <boost/program_options.hpp>
#include <iostream>

static const int NUM_DATA_ITEMS = 30000000;
static const auto CHUNK_SIZE = 10000000ul;
struct pi_options {
    unsigned long iterations;
    unsigned long chunk_size;
    unsigned int num_ipus;
    int precision;
};

static inline pi_options parse_options(int argc, char* argv[], const char *desc)
{
    using boost::program_options::options_description;
    using boost::program_options::value;
    using boost::program_options::variables_map;
    
    options_description options(desc);
    options.add_options()
        ("help", "help message")
        ("iterations", value<unsigned long>(), "number of iterations")
        ("chunk_size", value<unsigned long>(), "size of compute chunk")
        ("num_ipus", value<unsigned int>(), "number of IPUs")
        ("precision", value<int>(), "pi print precision");
    variables_map vm;

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), vm);

    if (vm.count("help")) {
        std::cout << options << std::endl;
        exit(EXIT_SUCCESS);
    }

    pi_options ret = {NUM_DATA_ITEMS, CHUNK_SIZE, 1, 10};

    if (vm.count("iterations"))
        ret.iterations = vm["iterations"].as<unsigned long>();

    if (vm.count("chunk_size"))
        ret.chunk_size = vm["chunk_size"].as<unsigned long>();

    if (vm.count("num_ipus"))
        ret.num_ipus = vm["num_ipus"].as<unsigned int>();

    if (vm.count("precistion"))
        ret.precision = vm["precision"].as<int>();

    return ret;
}

