// -*- C++ -*-
//===-- common_for_device_tests.h -----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Common functionality for device tests

#ifndef _ONEDPL_RANDOM_DEVICE_TESTS_COMMON_H
#define _ONEDPL_RANDOM_DEVICE_TESTS_COMMON_H

#include <oneapi/dpl/random>
#include <limits>
#include <iostream>
#include <iomanip>

#include "support/utils.h"

constexpr auto seed = 777;
constexpr int N = 96;

template <typename Fp>
int comparison(Fp* r0, Fp* r1, std::uint32_t length) {
    int numErrors = 0;
    for (size_t i = 0; i < length; ++i) {
        if constexpr (std::is_integral_v<Fp>) {
            if (((int)r0[i] - (int)r1[i]) > 1 || ((int)r1[i] - (int)r0[i]) > 1) {
                std::cout << "mismatch in " << i << " element: " << r0[i] << " " << r1[i] << std::endl;
                ++numErrors;
            }
        } else {
            auto diff = std::fabs(r0[i] - r1[i]);
            auto norm = std::fmax(fabs(r0[i]), fabs(r1[i]));
            if (diff > norm * 1000 * 16 * std::numeric_limits<Fp>::epsilon()) {
                std::cout <<  "mismatch in " << i << " element: "  << std::endl;
                std::cout << std::setprecision (15) << r0[i]  << std::endl;
                std::cout << std::setprecision (15) << r1[i] << std::endl;
                ++numErrors;
            }
        }
    }
    return numErrors;
}

template<class Distr, class Engine>
int device_copyable_test(sycl::queue& queue) {

    using result_type = typename Distr::result_type;
    using scalar_type = typename Distr::scalar_type;

    Engine engine(seed);
    Distr distr;

    for (int i = 0; i < 99; i++)
        distr(engine);

    // memory allocation
    scalar_type r_dev[N];
    scalar_type r_host[N];


    constexpr int num_elems = oneapi::dpl::internal::type_traits_t<result_type>::num_elems == 0
                                  ? 1
                                  : oneapi::dpl::internal::type_traits_t<result_type>::num_elems;
    // device generation
    {
        sycl::buffer<scalar_type, 1> buffer(r_dev, N);

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor acc(buffer, cgh, sycl::write_only);

            cgh.parallel_for(sycl::range<1>(N / num_elems), [=](sycl::item<1> idx) {
                unsigned long long offset = idx.get_linear_id() * num_elems;
                Engine device_engine(engine);
                Distr device_distr(distr);
                device_engine.discard(offset);
                result_type res = device_distr(device_engine);
                res.store(idx.get_linear_id(), acc);
            });
        });
    }

    // host generation
    for (int i = 0; i < N/num_elems; i++)
    {
        result_type res = distr(engine);
        for (int j = 0; j < num_elems; j++)
            r_host[i*num_elems + j] = res[j];
    }

    // compare
    return comparison(r_dev, r_host, N);
}

#endif // _ONEDPL_RANDOM_DEVICE_TESTS_COMMON_H
