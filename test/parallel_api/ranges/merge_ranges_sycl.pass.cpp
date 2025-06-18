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

#include <iostream>

#if _ENABLE_RANGES_TESTING
class merge_2;
#endif

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    using T = int;

    const int in_n = 10;
    const int out_n = 2 * in_n;

    T in1[in_n] = { 0,  1, 2, 3, 4, 5, 6, 6, 6, 6};
    T in2[in_n] = {-2, -1, 2, 3, 3, 5, 6, 7, 8, 9};
    T out1[out_n] = {};
    T out2[out_n] = {};

    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<T> A(in1, sycl::range<1>(in_n));
        sycl::buffer<T> B(in2, sycl::range<1>(in_n));
        sycl::buffer<T> D(out1, sycl::range<1>(out_n));
        sycl::buffer<T> E(out2, sycl::range<1>(out_n));

        auto exec = TestUtils::get_dpcpp_test_policy();

        merge(exec, all_view(A), all_view(B), all_view<T, sycl::access::mode::write>(D));
        merge(TestUtils::make_device_policy<merge_2>(exec), A, B, E, ::std::less<T>()); //check passing sycl buffers directly
    }

    //check result
    bool res1 = ::std::is_sorted(out1, out1 + out_n, ::std::less<T>());
    res1 &= ::std::includes(out1, out1 + out_n, in1, in1 + in_n, ::std::less<T>());
    res1 &= ::std::includes(out1, out1 + out_n, in2, in2 + in_n, ::std::less<T>());
    EXPECT_TRUE(res1, "wrong effect from 'merge' with sycl ranges");

    bool res2 = ::std::is_sorted(out2, out2 + out_n, ::std::less<T>());
    res2 &= ::std::includes(out2, out2 + out_n, in1, in1 + in_n, ::std::less<T>());
    res2 &= ::std::includes(out2, out2 + out_n, in2, in2 + in_n, ::std::less<T>());
    EXPECT_TRUE(res2, "wrong effect from 'merge' with sycl ranges with predicate");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
