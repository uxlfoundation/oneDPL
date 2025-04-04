// -*- C++ -*-
//===-- philox_uniform_real_distr_dp_test.pass.cpp --------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of philox_engine statistics with uniform_real_distribution

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS

#include "common_for_distributions.hpp"

/* ------   Tested the statistics of different engines   ------ */
// n = 2
using philox2x32 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 32, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox2x64 =
    oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 64, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;

template<typename RealType>
using Distr = oneapi::dpl::uniform_real_distribution<RealType>;

#endif // TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    constexpr int nsamples = 100;
    int err = 0;

    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device()))
    {
        // testing sycl::vec<double, 1> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,1>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<Distr<sycl::vec<double, 1>>, std::uint_fast32_t, philox2x32>(queue, nsamples);
#    if TEST_LONG_RUN
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, nsamples);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, std::uint_fast32_t, philox2x32>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, std::uint_fast32_t, philox2x32>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 100, 2);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 1> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,1>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<Distr<sycl::vec<double, 1>>, std::uint_fast64_t, philox2x64>(queue, nsamples);
#    if TEST_LONG_RUN
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, nsamples);
        err += tests_set<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, nsamples);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, std::uint_fast64_t, philox2x64>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, std::uint_fast64_t, philox2x64>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 100, 2);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 8> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,8>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<Distr<sycl::vec<double, 8>>, std::uint_fast32_t, philox2x32>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint_fast32_t, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint_fast32_t, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint_fast32_t, philox2x32>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 9);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 8> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,8>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<Distr<sycl::vec<double, 8>>, std::uint_fast64_t, philox2x64>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint_fast64_t, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint_fast64_t, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint_fast64_t, philox2x64>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 9);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 16> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,16>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<Distr<sycl::vec<double, 16>>, std::uint_fast32_t, philox2x32>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint_fast32_t, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint_fast32_t, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint_fast32_t, philox2x32>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 17);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 16> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,16>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<Distr<sycl::vec<double, 16>>, std::uint_fast64_t, philox2x64>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160);
        err += tests_set<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint_fast64_t, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint_fast64_t, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint_fast64_t, philox2x64>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 17);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
