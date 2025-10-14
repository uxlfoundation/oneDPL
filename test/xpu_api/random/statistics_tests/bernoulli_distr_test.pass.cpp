// -*- C++ -*-
//===-- bernoulli_distr_test.cpp ------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of bernoulli_distribution - check statistical properties of the distribution

#include "support/utils.h"
#include <iostream>

#if TEST_UNNAMED_LAMBDAS
#include <limits>
#include <oneapi/dpl/random>
#include <math.h>
#include "statistics_common.h"

// Engine parameters
constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483563u;
constexpr auto seed = 777;

int
statistics_check(int nsamples, double p, bool* samples)
{
    // theoretical moments
    double tM = p;
    double tD = p * (1 - p);
    double tQ = p;

    std::vector<bool> samples_vec (samples, samples + nsamples);
    return compare_moments(nsamples, samples_vec, tM, tD, tQ);
}

template <class BoolType, class UIntType>
int
test(sycl::queue& queue, double p, int nsamples)
{

    // memory allocation
    std::unique_ptr<bool[]> samples(new bool[nsamples]);

    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<BoolType>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<BoolType>::num_elems;

    // dpstd generation
    {
        sycl::buffer<bool, 1> buffer(samples.get(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::bernoulli_distribution<BoolType> distr(p);

                sycl::vec<bool, num_elems> res = distr(engine);
                res.store(idx.get_linear_id(), acc);
            });
        });
    }

    // statistics check
    int err = statistics_check(nsamples, p, samples.get());

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

template <class BoolType, class UIntType>
int
test_portion(sycl::queue& queue, double p, int nsamples, unsigned int part)
{

    // memory allocation
    std::unique_ptr<bool[]> samples(new bool[nsamples]);
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<BoolType>::num_elems == 0
                                           ? 1
                                           : oneapi::dpl::internal::type_traits_t<BoolType>::num_elems;
    int n_elems = (part >= num_elems) ? num_elems : part;


    // generation
    {
        sycl::buffer<bool, 1> buffer(samples.get(), nsamples);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * n_elems;
                oneapi::dpl::linear_congruential_engine<UIntType, a, c, m> engine(seed, offset);
                oneapi::dpl::bernoulli_distribution<BoolType> distr(p);

                sycl::vec<bool, num_elems> res = distr(engine, part);
                for (int i = 0; i < n_elems; ++i)
                    acc[idx.get_linear_id() * n_elems + i] = res[i];
            });
        });
        queue.wait_and_throw();
    }

    // statistics check
    int err = statistics_check(nsamples, p, samples.get());

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

template <class BoolType, class UIntType>
int
tests_set(sycl::queue& queue, int nsamples)
{
    constexpr int nparams = 2;

    double p_array[nparams] = {0.2, 0.9};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "bernoulli_distribution test<type>, p = " << p_array[i]
                  << ", nsamples  = " << nsamples;
        if (test<BoolType, UIntType>(queue, p_array[i], nsamples))
        {
            return 1;
        }
    }
    return 0;
}

template <class BoolType, class UIntType>
int
tests_set_portion(sycl::queue& queue, std::int32_t nsamples, unsigned int part)
{
    constexpr int nparams = 2;

    sycl::vec<double, 1> p_array[nparams] = {0.2, 0.9};

    // Test for all non-zero parameters
    for (int i = 0; i < nparams; ++i)
    {
        std::cout << "bernoulli_distribution test<type>, p = " << p_array[i] << ", nsamples = " << nsamples
                  << ", part = " << part;
        if (test_portion<BoolType, UIntType>(queue, p_array[i], nsamples, part))
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
    // Skip tests if DP is not supported
    if (TestUtils::has_type_support<double>(queue.get_device())) {
        constexpr int nsamples = 100;
        int err = 0;

        // testing sycl::vec<bool, 1> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<bool,1>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<bool, 1>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
        err += tests_set_portion<sycl::vec<bool, 1>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 1>, std::uint32_t>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 16>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 8>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 4>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 3>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 2>>(queue, 100, 2);
        err += tests_set_portion<sycl::vec<bool, 1>, sycl::vec<std::uint32_t, 1>>(queue, 100, 2);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<bool, 2> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<bool,2>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<bool, 2>, std::uint32_t>(queue, nsamples);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 16>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 8>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 4>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 3>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 2>>(queue, nsamples);
        err += tests_set<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 1>>(queue, nsamples);
        err += tests_set_portion<sycl::vec<bool, 2>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, std::uint32_t>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 16>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 8>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 4>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 3>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 2>>(queue, 100, 3);
        err += tests_set_portion<sycl::vec<bool, 2>, sycl::vec<std::uint32_t, 1>>(queue, 100, 3);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<bool, 3> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<bool,3>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<bool, 3>, std::uint32_t>(queue, 99);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99);
        err += tests_set<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99);
        err += tests_set<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99);
        err += tests_set<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99);
        err += tests_set<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99);
        err += tests_set<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99);
        err += tests_set_portion<sycl::vec<bool, 3>, std::uint32_t>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, std::uint32_t>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99, 1);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 16>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 8>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 4>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 3>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 2>>(queue, 99, 4);
        err += tests_set_portion<sycl::vec<bool, 3>, sycl::vec<std::uint32_t, 1>>(queue, 99, 4);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<bool, 4> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<bool,4>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<bool, 4>, std::uint32_t>(queue, 100);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100);
        err += tests_set<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100);
        err += tests_set<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100);
        err += tests_set<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100);
        err += tests_set<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100);
        err += tests_set<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100);
        err += tests_set_portion<sycl::vec<bool, 4>, std::uint32_t>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, std::uint32_t>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100, 1);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 16>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 8>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 4>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 3>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 2>>(queue, 100, 5);
        err += tests_set_portion<sycl::vec<bool, 4>, sycl::vec<std::uint32_t, 1>>(queue, 100, 5);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<bool, 8> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<bool,8>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<bool, 8>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160);
        err += tests_set<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160);
        err += tests_set<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160);
        err += tests_set<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160);
        err += tests_set<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160);
        err += tests_set<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160);
        err += tests_set_portion<sycl::vec<bool, 8>, std::uint32_t>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, std::uint32_t>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, std::uint32_t>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 5);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 16>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 8>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 4>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 3>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 2>>(queue, 160, 9);
        err += tests_set_portion<sycl::vec<bool, 8>, sycl::vec<std::uint32_t, 1>>(queue, 160, 9);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");

        // testing sycl::vec<bool, 16> and std::uint32_t ... sycl::vec<std::uint32_t, 16>
        std::cout << "---------------------------------------------------------------------" << std::endl;
        std::cout << "sycl::vec<bool,16>, std::uint32_t ... sycl::vec<std::uint32_t, 16> type" << std::endl;
        std::cout << "---------------------------------------------------------------------" << std::endl;
        err = tests_set<sycl::vec<bool, 16>, std::uint32_t>(queue, 160);
#if TEST_LONG_RUN
        err += tests_set<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160);
        err += tests_set<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160);
        err += tests_set<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160);
        err += tests_set<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160);
        err += tests_set<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160);
        err += tests_set<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160);
        err += tests_set_portion<sycl::vec<bool, 16>, std::uint32_t>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, std::uint32_t>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, std::uint32_t>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160, 1);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 16>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 8>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 4>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 3>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 2>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 1>>(queue, 140, 7);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 16>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 8>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 4>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 3>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 2>>(queue, 160, 17);
        err += tests_set_portion<sycl::vec<bool, 16>, sycl::vec<std::uint32_t, 1>>(queue, 160, 17);
#endif // TEST_LONG_RUN
        EXPECT_TRUE(!err, "Test FAILED");
    }

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
