// -*- C++ -*-
//===-- philox_uniform_real_distribution_test.pass.cpp --------------------===//
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
#include <iostream>

#if TEST_UNNAMED_LAMBDAS
#include <vector>
#include <random>
#include <oneapi/dpl/random>
#include "statistics_common.h"

/* ------   Tested the statistics of different engines   ------ */
// n = 2
using philox2x32 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 32, 2, 10, 0xd256d193, 0x0>;
using philox2x64 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 64, 2, 10, 0xD2B74407B1CE6E93, 0x0>;

// bitsize(result_type) != word_size, test only scalar output
using philox2x32_w5 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 5, 2, 10, 0xd256d193, 0x0>;
using philox2x32_w15 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 15, 2, 10, 0xd256d193, 0x0>;
using philox2x32_w18 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 18, 2, 10, 0xd256d193, 0x0>;
using philox2x32_w30 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 30, 2, 10, 0xd256d193, 0x0>;

using philox2x64_w5 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 5, 2, 10, 0xD2B74407B1CE6E93, 0x0>;
using philox2x64_w15 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 15, 2, 10, 0xD2B74407B1CE6E93, 0x0>;
using philox2x64_w18 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 18, 2, 10, 0xD2B74407B1CE6E93, 0x0>;
using philox2x64_w25 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 25, 2, 10, 0xD2B74407B1CE6E93, 0x0>;
using philox2x64_w49 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 49, 2, 10, 0xD2B74407B1CE6E93, 0x0>;

using philox4x32_w5 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 5, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x32_w15 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 15, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x32_w18 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 18, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x32_w30 = oneapi::dpl::experimental::philox_engine<std::uint_fast32_t, 30, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;

using philox4x64_w5 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 5, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w15 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 15, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w18 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 18, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w25 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 25, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;
using philox4x64_w49 = oneapi::dpl::experimental::philox_engine<std::uint_fast64_t, 49, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15, 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

template <typename RealType>
std::int32_t
statistics_check(int nsamples, RealType left, RealType right, const std::vector<RealType>& samples)
{
    // theoretical moments
    double tM = (right + left) / 2.0;
    double tD = ((right - left) * (right - left)) / 12.0;
    double tQ = ((right - left) * (right - left) * (right - left) * (right - left)) / 80.0;

    return compare_moments(nsamples, samples, tM, tD, tQ);
}

template <typename RealType, typename UIntType, typename Engine>
int
test(sycl::queue& queue, oneapi::dpl::internal::element_type_t<RealType> left,
     oneapi::dpl::internal::element_type_t<RealType> right, int nsamples)
{

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * num_elems;
                Engine engine;
                engine.discard(offset);
                oneapi::dpl::uniform_real_distribution<RealType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc);
            });
        });
        queue.wait();
    }

    // statistics check
    int err = statistics_check(nsamples, left, right, samples);

    if (err)
    {
        std::cout << "\tFailed" << std::endl;
    }
    else
    {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template <typename RealType, typename UIntType, typename Engine>
int
test_portion(sycl::queue& queue, oneapi::dpl::internal::element_type_t<RealType> left,
             oneapi::dpl::internal::element_type_t<RealType> right, int nsamples, unsigned int part)
{

    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<RealType>> samples(nsamples);
    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<RealType>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<RealType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;

    // generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<RealType>> buffer(samples.data(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * n_elems;
                Engine engine;
                engine.discard(offset);
                oneapi::dpl::uniform_real_distribution<RealType> distr(left, right);

                sycl::vec<oneapi::dpl::internal::element_type_t<RealType>, num_elems> res = distr(engine, part);
                for (int i = 0; i < n_elems; ++i)
                    acc[offset + i] = res[i];
            });
        });
        queue.wait();
    }

    // statistics check
    int err = statistics_check(nsamples, left, right, samples);

    if (err)
    {
        std::cout << "\tFailed" << std::endl;
    }
    else
    {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template <typename RealType, typename UIntType, typename Engine>
int
tests_set(sycl::queue& queue, int nsamples)
{

    constexpr int nparams = 2;
    oneapi::dpl::internal::element_type_t<RealType> left_array[nparams] = {0.0, -10.0};
    oneapi::dpl::internal::element_type_t<RealType> right_array[nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i]
                  << ", nsamples  = " << nsamples;
        if (test<RealType, UIntType, Engine>(queue, left_array[i], right_array[i], nsamples))
        {
            return 1;
        }
    }
    return 0;
}

template <typename RealType, typename UIntType, typename Engine>
int
tests_set_portion(sycl::queue& queue, int nsamples, unsigned int part)
{

    constexpr int nparams = 2;
    oneapi::dpl::internal::element_type_t<RealType> left_array[nparams] = {0.0, -10.0};
    oneapi::dpl::internal::element_type_t<RealType> right_array[nparams] = {1.0, 10.0};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "uniform_real_distribution test<type>, left = " << left_array[i] << ", right = " << right_array[i]
                  << ", nsamples = " << nsamples << ", part = " << part;
        if (test_portion<RealType, UIntType, Engine>(queue, left_array[i], right_array[i], nsamples, part))
        {
            return 1;
        }
    }
    return 0;
}

#endif // TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    constexpr int nsamples = 100;
    int err = 0;

    // testing sycl::vec<float, 1> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32>(queue, nsamples);
#    if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, nsamples);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 100, 2);
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 1> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64>(queue, nsamples);
#    if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, nsamples);
    err += tests_set<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, nsamples);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<float, 1>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 100, 2);
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32>(queue, 160);
#    if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::int_fast32_t, 16>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 9);
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64>(queue, 160);
#    if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 5);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 9);
    err += tests_set_portion<sycl::vec<float, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 9);
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32>(queue, 160);
#    if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 17);
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64>(queue, 160);
#    if TEST_LONG_RUN
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160);
    err += tests_set<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 1);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 140, 7);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 17);
    err += tests_set_portion<sycl::vec<float, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 17);
#    endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device()))
    {
        // testing sycl::vec<double, 1> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,1>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 1>, std::uint_fast32_t, philox2x32>(queue, nsamples);
#    if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, nsamples);
        err += tests_set_portion<sycl::vec<double, 1>, std::uint_fast32_t, philox2x32>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, std::uint_fast32_t, philox2x32>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 100, 2);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 1> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,1>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 1>, std::uint_fast64_t, philox2x64>(queue, nsamples);
#    if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, nsamples);
        err += tests_set<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, nsamples);
        err += tests_set_portion<sycl::vec<double, 1>, std::uint_fast64_t, philox2x64>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, std::uint_fast64_t, philox2x64>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<double, 1>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 100, 2);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 8> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,8>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 8>, std::uint_fast32_t, philox2x32>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint_fast32_t, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint_fast32_t, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint_fast32_t, philox2x32>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 9);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 8> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,8>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 8>, std::uint_fast64_t, philox2x64>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint_fast64_t, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint_fast64_t, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, std::uint_fast64_t, philox2x64>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<double, 8>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 9);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 16> and std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,16>, std::uint_fast32_t ... sycl::vec<std::uint_fast32_t, 16>, philox2x32 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 16>, std::uint_fast32_t, philox2x32>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint_fast32_t, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint_fast32_t, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint_fast32_t, philox2x32>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 16>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 8>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 4>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 3>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 2>, philox2x32>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast32_t, 1>, philox2x32>(queue, 160, 17);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<double, 16> and std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<double,16>, std::uint_fast64_t ... sycl::vec<std::uint_fast64_t, 16>, philox2x64 type"
                  << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<double, 16>, std::uint_fast64_t, philox2x64>(queue, 160);
#    if TEST_LONG_RUN
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160);
        err += tests_set<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint_fast64_t, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint_fast64_t, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, std::uint_fast64_t, philox2x64>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 16>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 8>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 4>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 3>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 2>, philox2x64>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<double, 16>, sycl::vec<std::uint_fast64_t, 1>, philox2x64>(queue, 160, 17);
#    endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

    // testing sycl::vec<float, 1> and std::uint_fast32_t philox2x32_w*/philox4x32_w*
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint_fast32_t, philox2x32_w*/philox4x32_w* type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox2x32_w30>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox4x32_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox4x32_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox4x32_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast32_t, philox4x32_w30>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 1> and std::uint_fast64_t philox2x64_w*/philox4x64_w*
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,1>, std::uint_fast64_t, philox2x64_w*/philox4x64_w* type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64_w25>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox2x64_w49>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox4x64_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox4x64_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox4x64_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox4x64_w25>(queue, nsamples);
    err = tests_set<sycl::vec<float, 1>, std::uint_fast64_t, philox4x64_w49>(queue, nsamples);

    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint_fast32_t philox2x32_w*/philox4x32_w*
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint_fast32_t, philox2x32_w*/philox4x32_w* type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox2x32_w30>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox4x32_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox4x32_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox4x32_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast32_t, philox4x32_w30>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 8> and std::uint_fast64_t philox2x64_w*/philox4x64_w*
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,8>, std::uint_fast64_t, philox2x64_w*/philox4x64_w* type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64_w25>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox2x64_w49>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox4x64_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox4x64_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox4x64_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox4x64_w25>(queue, nsamples);
    err = tests_set<sycl::vec<float, 8>, std::uint_fast64_t, philox4x64_w49>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float, 16> and std::uint_fast32_t philox2x32_w*/philox4x32_w*
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint_fast32_t, philox2x32_w*/philox4x32_w* type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox2x32_w30>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox4x32_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox4x32_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox4x32_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast32_t, philox4x32_w30>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<float,16> and std::uint_fast64_t philox2x64_w*/philox4x64_w*
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "sycl::vec<float,16>, std::uint_fast64_t, philox2x64_w*/philox4x64_w* type" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64_w25>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox2x64_w49>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox4x64_w5>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox4x64_w15>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox4x64_w18>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox4x64_w25>(queue, nsamples);
    err = tests_set<sycl::vec<float, 16>, std::uint_fast64_t, philox4x64_w49>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
