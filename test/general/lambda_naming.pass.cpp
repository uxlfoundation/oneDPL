// -*- C++ -*-
//===-- lambda_naming.pass.cpp --------------------------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

using namespace TestUtils;

// This is the simple test for compilation only, to check if lambda naming works correctly
int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    const int n = 1000;
    sycl::buffer<int> buf{ sycl::range<1>(n) };
    sycl::buffer<int> out_buf{ sycl::range<1>(n) };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end = buf_begin + n;

    const auto policy = TestUtils::default_dpcpp_policy;
    auto buf_begin_discard_write = oneapi::dpl::begin(buf, sycl::write_only, sycl::property::no_init{});

    ::std::fill(policy, buf_begin_discard_write, buf_begin_discard_write + n, 1);
#if __SYCL_UNNAMED_LAMBDA__
    ::std::sort(policy, buf_begin, buf_end);
    ::std::for_each(policy, buf_begin, buf_end, [](int& x) { x += 41; });

#if !ONEDPL_FPGA_DEVICE
    sycl::buffer<float> out_buf_2{ sycl::range<1>(n) };
    auto buf_out_begin_2 = oneapi::dpl::begin(out_buf_2);
    ::std::copy(policy, buf_begin, buf_end, buf_out_begin_2);
    ::std::copy(policy, buf_out_begin_2, buf_out_begin_2 + n, buf_begin);
    ::std::inplace_merge(policy, buf_begin, buf_begin + n / 2, buf_end);
    auto red_val = ::std::reduce(policy, buf_begin, buf_end, 1);
    EXPECT_TRUE(red_val == 42001, "wrong return value from reduce");
    auto buf_out_begin = oneapi::dpl::begin(out_buf);
    ::std::inclusive_scan(policy, buf_begin, buf_end, buf_out_begin);
    bool is_equal = ::std::equal(policy, buf_begin, buf_end, buf_out_begin);
    EXPECT_TRUE(!is_equal, "wrong return value from equal");
    auto does_1_exist = ::std::find(policy, buf_begin, buf_end, 1);
    EXPECT_TRUE(does_1_exist - buf_begin == 1000, "wrong return value from find");
#endif // !ONEDPL_FPGA_DEVICE

#else
    // ::std::for_each(policy, buf_begin, buf_end, [](int& x) { x++; }); // It's not allowed. Policy with different name is needed
    ::std::for_each(TestUtils::make_device_policy<class ForEach>(policy), buf_begin, buf_end, [](int& x) { x++; });
    auto red_val = ::std::reduce(policy, buf_begin, buf_end, 1);
    EXPECT_TRUE(red_val == 2001, "wrong return value from reduce");
#endif // __SYCL_UNNAMED_LAMBDA__


    {
        const auto policy1 = TestUtils::default_dpcpp_policy;

        const auto red_val1 = std::reduce(policy1, buf_begin, buf_end, 1);
        EXPECT_TRUE(red_val1 == 42001, "wrong return value from reduce #1");
        const auto red_val11 = std::reduce(policy1, buf_begin, buf_end, 1);
        EXPECT_TRUE(red_val11 == 42001, "wrong return value from reduce #1");
        const auto red_val2 = std::reduce(std::move(policy1), buf_begin, buf_end, 1);
        EXPECT_TRUE(red_val2 == 42001, "wrong return value from reduce #2");
    }

#endif // TEST_DPCPP_BACKEND_PRESENT

    return done(TEST_DPCPP_BACKEND_PRESENT);
}
