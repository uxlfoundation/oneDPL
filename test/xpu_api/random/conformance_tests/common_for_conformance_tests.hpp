// -*- C++ -*-
//===-- common_for_conformance_tests.hpp -----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Common functionality for conformance tests

#ifndef _DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON_HPP
#define _DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON_HPP

#include <vector>
#include <random>

#include "support/utils.h"

constexpr auto REF_SAMPLE_ID = 9999;

template<class Engine, int NGenSamples, int NElemsInResultType>
typename Engine::scalar_type test(sycl::queue& queue) {

    using result_type = typename Engine::scalar_type;

    // Memory allocation
    std::vector<result_type> dpstd_samples(NGenSamples);

    // Random number generation
    {
        sycl::buffer<result_type, 1> dpstd_buffer(dpstd_samples.data(), NGenSamples);

        try {
            queue.submit([&](sycl::handler &cgh) {
                auto dpstd_acc = dpstd_buffer.template get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for<>(sycl::range<1>(NGenSamples / NElemsInResultType),
                        [=](sycl::item<1> idx) {

                    unsigned long long offset = idx.get_linear_id() * NElemsInResultType;
                    Engine engine;
                    engine.discard(offset);

                    sycl::vec<result_type, NElemsInResultType> res = engine();
                    res.store(idx.get_linear_id(), dpstd_acc);
                });
            });
            queue.wait_and_throw();
        }
        catch(sycl::exception const& e) {
            std::cout << "\t\tSYCL exception during generation\n"
                      << e.what() << std::endl;
            return 0;
        }
    }

    return dpstd_samples[REF_SAMPLE_ID];
}

#endif // _DPSTD_RANDOM_CONFORMANCE_TESTS_COMMON_HPP
