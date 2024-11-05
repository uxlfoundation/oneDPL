// -*- C++ -*-
//===-- subtract_with_carry_std_template_test.cpp -------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of subtract_with_carry_engine - comparison with std::

#include "support/utils.h"
#include <iostream>

#if TEST_UNNAMED_LAMBDAS
#include <vector>
#include <random>
#include <oneapi/dpl/random>

template<class UIntType, size_t w,
                         size_t s,
                         size_t r>
int test(sycl::queue& queue, oneapi::dpl::internal::element_type_t<UIntType> seed, int nsamples) {
    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<UIntType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<UIntType>> dpstd_samples(nsamples);
    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<UIntType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<UIntType>::num_elems;

    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<UIntType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / num_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * num_elems;
                oneapi::dpl::subtract_with_carry_engine<UIntType, w, s, r> engine(seed, offset);

                sycl::vec<oneapi::dpl::internal::element_type_t<UIntType>, num_elems> res = engine();
                res.store(idx.get_linear_id(), dpstd_acc);
            });
        });
        queue.wait();
    }

    // std generation
    std::subtract_with_carry_engine<oneapi::dpl::internal::element_type_t<UIntType>, w, s, r> std_engine(seed);
    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = std_engine();

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if (std_samples[i] != dpstd_samples[i]) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else{
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class UIntType, size_t w,
                         size_t s,
                         size_t r>
int test_portion(sycl::queue& queue, oneapi::dpl::internal::element_type_t<UIntType> seed, int nsamples, unsigned int part) {
    // memory allocation
    std::vector<oneapi::dpl::internal::element_type_t<UIntType>> std_samples(nsamples);
    std::vector<oneapi::dpl::internal::element_type_t<UIntType>> dpstd_samples(nsamples);

    // Calculate n_wi
    constexpr unsigned int num_elems = oneapi::dpl::internal::type_traits_t<UIntType>::num_elems == 0 ? 1 : oneapi::dpl::internal::type_traits_t<UIntType>::num_elems;
    unsigned int n_elems = (part >= num_elems) ? num_elems : part;
    // dpstd generation
    {
        sycl::buffer<oneapi::dpl::internal::element_type_t<UIntType>, 1> dpstd_buffer(dpstd_samples.data(), nsamples);

        queue.submit([&](sycl::handler &cgh) {
            auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(nsamples / n_elems),
                    [=](sycl::item<1> idx) {

                unsigned long long offset = idx.get_linear_id() * n_elems;
                oneapi::dpl::subtract_with_carry_engine<UIntType, w, s, r> engine(seed, offset);

                auto res = engine(part);
                for(unsigned int i = 0; i < n_elems; ++i)
                    dpstd_acc[offset + i] = res[i];
            });
        });
        queue.wait();
    }

    // std generation
    std::subtract_with_carry_engine<oneapi::dpl::internal::element_type_t<UIntType>, w, s, r> std_engine(seed);
    for(int i = 0; i < nsamples; ++i)
        std_samples[i] = std_engine();

    // comparison
    int err = 0;
    for(int i = 0; i < nsamples; ++i) {
        if (std_samples[i] != dpstd_samples[i]) {
            std::cout << "\nError: std_sample[" << i << "] = " << std_samples[i] << ", dpstd_samples[" << i << "] = " << dpstd_samples[i];
            err++;
        }
    }

    if(err) {
        std::cout << "\tFailed" << std::endl;
    }
    else {
        std::cout << "\tPassed" << std::endl;
    }

    return err;
}

template<class Type>
int tests_set(sycl::queue& queue, int nsamples) {
    constexpr int nseeds = 2;
    std::int64_t seed_array [nseeds] = {0, 19780503u};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nseeds; ++i) {
        std::cout << "SWC test<Type, 24u, 10u, 24u>(" << seed_array[i] << ", "<< nsamples << ")";
        err = test<Type, 24u, 10u, 24u>(queue, seed_array[i], nsamples);
        if(err) {
            return 1;
        }
    }

    // Test for zero addition parameter
    for(int i = 0; i < nseeds; ++i) {
        std::cout << "SWC test<Type, 24u, 5u, 12u>(" << seed_array[i] << ", "<< nsamples << ")";
        err = test<Type, 24u, 5u, 12u>(queue, seed_array[i], nsamples);
        if(err) {
            return 1;
        }
    }

    return 0;
}

template<class Type>
int tests_set_portion(sycl::queue& queue, int nsamples, unsigned int part) {
    constexpr int nseeds = 1;
    std::int64_t seed_array [nseeds] = {19780503u};

    int err;
    // Test for all non-zero parameters
    for(int i = 0; i < nseeds; ++i) {
        std::cout << "SWC test_portion<Type, 24u, 10u, 24u>(" << seed_array[i] << ", "<< nsamples << ", " << part << ")";
        err = test_portion<Type, 24u, 10u, 24u>(queue, seed_array[i], nsamples, part);
        if(err) {
            return 1;
        }
    }

    return 0;
}

#endif // TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    constexpr int nsamples = 100;
    int err = 0;

    // testing std::uint32_t
    std::cout << "-----------------------------" << std::endl;
    std::cout << "std::uint32_t Type" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<std::uint32_t>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing std::uint64_t
    std::cout << "-----------------------------" << std::endl;
    std::cout << "std::uint64_t Type" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<std::uint64_t>(queue, nsamples);
    EXPECT_TRUE(!err, "Test FAILED");

#if TEST_LONG_RUN
    // testing sycl::vec<std::uint32_t, 1>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint32_t, 1>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint32_t, 1>>(queue, nsamples);
    err += tests_set_portion<sycl::vec<std::uint32_t, 1>>(queue, nsamples, 1);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint32_t, 2>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint32_t, 2>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint32_t, 2>>(queue, nsamples);
    err += tests_set_portion<sycl::vec<std::uint32_t, 2>>(queue, nsamples, 1);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint32_t, 3>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint32_t, 3>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint32_t, 3>>(queue, 99);
    err += tests_set_portion<sycl::vec<std::uint32_t, 3>>(queue, 100, 2);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint32_t, 4>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint32_t, 4>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint32_t, 4>>(queue, 100);
    err += tests_set_portion<sycl::vec<std::uint32_t, 4>>(queue, 99, 3);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint32_t, 8>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint32_t, 8>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint32_t, 8>>(queue, 400);
    err += tests_set_portion<sycl::vec<std::uint32_t, 8>>(queue, 100, 5);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint32_t, 16>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint32_t, 16>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint32_t, 16>>(queue, 400);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 99, 3);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 100, 4);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 60, 6);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 70, 7);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 80, 8);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 90, 9);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 100, 10);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 110, 11);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 120, 12);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 130, 13);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 140, 14);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 150, 15);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 160, 16);
    err += tests_set_portion<sycl::vec<std::uint32_t, 16>>(queue, 160, 17);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint64_t, 1>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint64_t, 1>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint64_t, 1>>(queue, nsamples);
    err += tests_set_portion<sycl::vec<std::uint64_t, 1>>(queue, nsamples, 1);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint64_t, 2>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint64_t, 2>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint64_t, 2>>(queue, nsamples);
    err += tests_set_portion<sycl::vec<std::uint64_t, 2>>(queue, nsamples, 1);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint64_t, 3>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint64_t, 3>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint64_t, 3>>(queue, 99);
    err += tests_set_portion<sycl::vec<std::uint64_t, 3>>(queue, 100, 2);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint64_t, 4>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint64_t, 4>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint64_t, 4>>(queue, 100);
    err += tests_set_portion<sycl::vec<std::uint64_t, 4>>(queue, 99, 3);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint64_t, 8>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint64_t, 8>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint64_t, 8>>(queue, 80);
    err += tests_set_portion<sycl::vec<std::uint64_t, 8>>(queue, 100, 5);
    EXPECT_TRUE(!err, "Test FAILED");

    // testing sycl::vec<std::uint64_t, 16>
    std::cout << "-----------------------------" << std::endl;
    std::cout << "sycl::vec<std::uint64_t, 16>" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    err += tests_set<sycl::vec<std::uint64_t, 16>>(queue, 400);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 100, 1);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 100, 2);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 99, 3);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 100, 4);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 100, 5);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 60, 6);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 70, 7);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 80, 8);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 90, 9);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 100, 10);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 110, 11);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 120, 12);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 130, 13);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 140, 14);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 150, 15);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 160, 16);
    err += tests_set_portion<sycl::vec<std::uint64_t, 16>>(queue, 160, 17);
    EXPECT_TRUE(!err, "Test FAILED");
#endif // TEST_LONG_RUN

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
