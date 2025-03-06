// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include <oneapi/dpl/execution>

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <cstdint>
#if _ONEDPL_DEBUG_SYCL
#include <iostream>
#endif

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    const int n = 7, n_res = 4;
    int a[n] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    int b[n] = {9, 8, 7, 6, 5, 4, 3}; // input values

    sycl::buffer<int> A(a, sycl::range<1>(n));
    sycl::buffer<int> B(b, sycl::range<1>(n));
    sycl::buffer<int> C(n);           // output keys
    sycl::buffer<int> D(n);           // output values

    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;

    auto exec = TestUtils::default_dpcpp_policy;

    [[maybe_unused]] auto res = reduce_by_segment(exec, views::all_read(A), views::all_read(B),
                                                  views::all_write(C), views::all_write(D));

    int key_exp[n_res] = {1, 3, 2, 1};    // expected keys
    int value_exp[n_res] = {9, 21, 9, 3}; // expected values

#if _ONEDPL_DEBUG_SYCL
    ::std::cout << "keys: ";
    for(auto v: views::host_all(C) | views::take(res))
        ::std::cout << v << " ";
    ::std::cout << ::std::endl;

    ::std::cout << "values: ";
    for(auto v: views::host_all(D) | views::take(res))
        ::std::cout << v << " ";
    ::std::cout << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

    //check result
    EXPECT_EQ_N(key_exp, views::host_all(C).begin(), n_res, "wrong keys from reduce_by_segment");
    EXPECT_EQ_N(value_exp, views::host_all(D).begin(), n_res, "wrong values from reduce_by_segment");

// Check if a kernel name can be omitted when a compiler supports implicit names
#if __SYCL_UNNAMED_LAMBDA__
    sycl::buffer<::std::uint64_t> E(n);
    reduce_by_segment(exec, views::all_read(A), views::all_read(B), views::all_write(C), views::all_write(E));
#endif

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
