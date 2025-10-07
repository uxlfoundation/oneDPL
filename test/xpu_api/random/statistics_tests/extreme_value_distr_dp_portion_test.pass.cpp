// -*- C++ -*-
//===-- extreme_value_distr_dp_portion_test.cpp ---------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of extreme_value_distribution - check statistical properties of the distribution

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_distributions.hpp"

template<typename RealType>
using Distr = oneapi::dpl::extreme_value_distribution<RealType>;

#endif // TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    int err = 0;

    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device())) {
#if TEST_LONG_RUN
        // testing sycl::vec<double, 1> and std::uint32_t ... sycl::vec<std::uint32_t, 3>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,1>, std::uint32_t ... sycl::vec<std::uint32_t, 3> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set_portion<Distr<sycl::vec<double, 1>>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, std::uint32_t>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 2);
        err += tests_set_portion<Distr<sycl::vec<double, 1>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 2);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

#if TEST_LONG_RUN
        // testing sycl::vec<double, 2> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,2>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set_portion<Distr<sycl::vec<double, 2>>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, std::uint32_t>(queue, 100, 3);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 3);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 3);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 3);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 3);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 3);
        err += tests_set_portion<Distr<sycl::vec<double, 2>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 3);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

#if TEST_LONG_RUN
        // testing sycl::vec<double, 3> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,3>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set_portion<Distr<sycl::vec<double, 3>>, std::uint32_t>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, std::uint32_t>(queue, 99, 4);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 16>>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 8>>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 4>>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 3>>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 2>>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 1>>(queue, 99, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 16>>(queue, 99, 4);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 8>>(queue, 99, 4);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 4>>(queue, 99, 4);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 3>>(queue, 99, 4);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 2>>(queue, 99, 4);
        err += tests_set_portion<Distr<sycl::vec<double, 3>>, sycl::vec<std::uint32_t, 1>>(queue, 99, 4);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

#if TEST_LONG_RUN
        // testing sycl::vec<double, 4> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,4>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set_portion<Distr<sycl::vec<double, 4>>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, std::uint32_t>(queue, 100, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 16>>(queue, 100, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 8>>(queue, 100, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 4>>(queue, 100, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 3>>(queue, 100, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 2>>(queue, 100, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 4>>, sycl::vec<std::uint32_t, 1>>(queue, 100, 5);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

#if TEST_LONG_RUN
        // testing sycl::vec<double, 8> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,8>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint32_t>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint32_t>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, std::uint32_t>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 5);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 9);
        err += tests_set_portion<Distr<sycl::vec<double, 8>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 9);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

#if TEST_LONG_RUN
        // testing sycl::vec<double, 16> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,16>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint32_t>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint32_t>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, std::uint32_t>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 140, 7);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 16>>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 8>>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 4>>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 3>>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 2>>(queue, 160, 17);
        err += tests_set_portion<Distr<sycl::vec<double, 16>>, sycl::vec<std::uint32_t, 1>>(queue, 160, 17);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
