// -*- C++ -*-
//===-- bernoulli_distr_device_test.pass.cpp ---------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Device copyable tests for bernoulli distribution

#include "support/utils.h"
#include <iostream>

#if TEST_UNNAMED_LAMBDAS
#include "common_for_device_tests.h"
#endif // TEST_UNNAMED_LAMBDAS

constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483563u;

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device())) {
        int err = 0;

        // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>> oneapi::dpl::linear_congruential_engine
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<sycl::vec<bool, 1>> linear_congruential_engine" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>> oneapi::dpl::linear_congruential_engine
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<sycl::vec<bool, 2>> linear_congruential_engine" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>> oneapi::dpl::linear_congruential_engine
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<sycl::vec<bool, 3>> linear_congruential_engine" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>> oneapi::dpl::linear_congruential_engine
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<sycl::vec<bool, 4>> linear_congruential_engine" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>> oneapi::dpl::linear_congruential_engine
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<sycl::vec<bool, 8>> linear_congruential_engine" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>> oneapi::dpl::linear_congruential_engine
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<sycl::vec<bool, 16>> linear_congruential_engine" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
        err += device_copyable_test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
