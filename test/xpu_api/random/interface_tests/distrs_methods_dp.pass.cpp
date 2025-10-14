// -*- C++ -*-
//===-- distrs_methods_dp.pass.cpp -------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Testing of different distributions' methods with integer and double
// whose implementation uses DP inside

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_distrs_methods.hpp"
#endif // TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();
    std::int32_t err = 0;

    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device())) {

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "uniform_int_distribution<std::int32_t>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::uniform_int_distribution<std::int32_t>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 16>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 8>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 4>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 3>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 2>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::int32_t, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "uniform_int_distribution<std::uint32_t>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::uniform_int_distribution<std::uint32_t>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 16>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 8>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 4>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 3>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 2>>>(queue);
        err += test_vec<oneapi::dpl::uniform_int_distribution<sycl::vec<std::uint32_t, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "uniform_real_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::uniform_real_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "normal_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::normal_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "exponential_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::exponential_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "bernoulli_distribution<bool>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::bernoulli_distribution<bool>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>>(queue);
        err += test_vec<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>>(queue);
        err += test_vec<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>>(queue);
        err += test_vec<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>>(queue);
        err += test_vec<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>>(queue);
        err += test_vec<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "geometric_distribution<std::int32_t>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::geometric_distribution<std::int32_t>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::int32_t, 16>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::int32_t, 8>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::int32_t, 4>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::int32_t, 3>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::int32_t, 2>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::int32_t, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "geometric_distribution<std::uint32_t>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::geometric_distribution<std::uint32_t>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::uint32_t, 16>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::uint32_t, 8>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::uint32_t, 4>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::uint32_t, 3>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::uint32_t, 2>>>(queue);
        err += test_vec<oneapi::dpl::geometric_distribution<sycl::vec<std::uint32_t, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "weibull_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::weibull_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "lognormal_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::lognormal_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "cauchy_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::cauchy_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "extreme_value_distribution<double>" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        err += test<oneapi::dpl::extreme_value_distribution<double>>(queue);
#if TEST_LONG_RUN
        err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>>(queue);
        err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>>(queue);
        err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>>(queue);
        err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>>(queue);
        err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>>(queue);
        err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>>(queue);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}