// -*- C++ -*-
//===-- distrs_methods_sp.pass.cpp -------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Testing of different distributions' methods with single precision

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

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::uniform_real_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::normal_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::normal_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::exponential_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::exponential_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::weibull_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::weibull_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "lognormal_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::lognormal_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::lognormal_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::cauchy_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<float>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += test<oneapi::dpl::extreme_value_distribution<float>>(queue);
#if TEST_LONG_RUN
    err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<float, 16>>>(queue);
    err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<float, 8>>>(queue);
    err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<float, 4>>>(queue);
    err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<float, 3>>>(queue);
    err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<float, 2>>>(queue);
    err += test_vec<oneapi::dpl::extreme_value_distribution<sycl::vec<float, 1>>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
