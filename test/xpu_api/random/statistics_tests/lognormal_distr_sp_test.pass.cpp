// -*- C++ -*-
//===-- lognormal_distr_sp_test.cpp ---------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//
//
// Abstract:
//
// Test of lognormal_distribution - check statistical properties of the distribution

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_distributions.hpp"

template<typename RealType>
using Distr = oneapi::dpl::lognormal_distribution<RealType>;

#endif // TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    constexpr int nsamples = 100;
    int err = 0;

    // testing sycl::vec<float, 1> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<Distr<sycl::vec<float, 1>>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, std::uint32_t>(queue, 100, 2);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 2);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 2);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 2);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 2);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 2);
    err += tests_set_portion<Distr<sycl::vec<float, 1>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 2);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 2> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,2>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<Distr<sycl::vec<float, 2>>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
    err += tests_set<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, std::uint32_t>(queue, 100, 3);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 3);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 3);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 3);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 3);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 3);
    err += tests_set_portion<Distr<sycl::vec<float, 2>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 3);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 3> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,3>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<Distr<sycl::vec<float, 3>>, std::uint32_t>(queue, 99);
#if TEST_LONG_RUN
    err += tests_set<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 16>>(queue, 99);
    err += tests_set<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 8>>(queue, 99);
    err += tests_set<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 4>>(queue, 99);
    err += tests_set<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 3>>(queue, 99);
    err += tests_set<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 2>>(queue, 99);
    err += tests_set<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 1>>(queue, 99);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, std::uint32_t>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, std::uint32_t>(queue, 99, 4);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 16>>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 8>>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 4>>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 3>>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 2>>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 1>>(queue, 99, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 16>>(queue, 99, 4);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 8>>(queue, 99, 4);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 4>>(queue, 99, 4);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 3>>(queue, 99, 4);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 2>>(queue, 99, 4);
    err += tests_set_portion<Distr<sycl::vec<float, 3>>, sycl::vec<std::uint32_t, 1>>(queue, 99, 4);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 4> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,4>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<Distr<sycl::vec<float, 4>>, std::uint32_t>(queue, 100);
#if TEST_LONG_RUN
    err += tests_set<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 16>>(queue, 100);
    err += tests_set<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 8>>(queue, 100);
    err += tests_set<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 4>>(queue, 100);
    err += tests_set<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 3>>(queue, 100);
    err += tests_set<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 2>>(queue, 100);
    err += tests_set<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 1>>(queue, 100);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, std::uint32_t>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, std::uint32_t>(queue, 100, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 4>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 5);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<Distr<sycl::vec<float, 8>>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, std::uint32_t>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, std::uint32_t>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, std::uint32_t>(queue, 160, 9);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 5);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 9);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 9);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 9);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 9);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 9);
    err += tests_set_portion<Distr<sycl::vec<float, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 9);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<Distr<sycl::vec<float, 16>>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
    err += tests_set<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 160);
    err += tests_set<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 160);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, std::uint32_t>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, std::uint32_t>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, std::uint32_t>(queue, 160, 17);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 140, 7);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 17);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 17);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 17);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 17);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 17);
    err += tests_set_portion<Distr<sycl::vec<float, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 17);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
