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

#if TEST_DPCPP_BACKEND_PRESENT

struct ForEach;

template <typename T>
struct PostIncrementOp
{
    void operator()(T& x) const
    {
        x++;
    }
};

template <typename T>
struct Add41Op
{
    void operator()(T& x) const
    {
        x += 41;
    }
};

template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int n = 1000;
    sycl::buffer<int> buf{sycl::range<1>(n)};
    sycl::buffer<int> out_buf{sycl::range<1>(n)};
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end = buf_begin + n;

    auto buf_begin_discard_write = oneapi::dpl::begin(buf, sycl::write_only, sycl::property::no_init{});

    std::fill(CLONE_TEST_POLICY(exec), buf_begin_discard_write, buf_begin_discard_write + n, 1);

#if __SYCL_UNNAMED_LAMBDA__

    std::sort(CLONE_TEST_POLICY(exec), buf_begin, buf_end);
    std::for_each(CLONE_TEST_POLICY(exec), buf_begin, buf_end, Add41Op<int>());

#if !ONEDPL_FPGA_DEVICE
    sycl::buffer<float> out_buf_2{sycl::range<1>(n)};
    auto buf_out_begin_2 = oneapi::dpl::begin(out_buf_2);
    std::copy(CLONE_TEST_POLICY(exec), buf_begin, buf_end, buf_out_begin_2);
    std::copy(CLONE_TEST_POLICY(exec), buf_out_begin_2, buf_out_begin_2 + n, buf_begin);
    std::inplace_merge(CLONE_TEST_POLICY(exec), buf_begin, buf_begin + n / 2, buf_end);
    auto red_val = std::reduce(CLONE_TEST_POLICY(exec), buf_begin, buf_end, 1);
    EXPECT_TRUE(red_val == 42001, "wrong return value from reduce");

    auto buf_out_begin = oneapi::dpl::begin(out_buf);
    std::inclusive_scan(CLONE_TEST_POLICY(exec), buf_begin, buf_end, buf_out_begin);
    bool is_equal = std::equal(CLONE_TEST_POLICY(exec), buf_begin, buf_end, buf_out_begin);
    EXPECT_TRUE(!is_equal, "wrong return value from equal");

    auto does_1_exist = std::find(CLONE_TEST_POLICY(exec), buf_begin, buf_end, 1);
    EXPECT_TRUE(does_1_exist - buf_begin == 1000, "wrong return value from find");
#endif // !ONEDPL_FPGA_DEVICE

#else

    // std::for_each(exec, buf_begin, buf_end, [](int& x) { x++; }); // It's not allowed. Policy with different name is needed
    std::for_each(CLONE_TEST_POLICY_NAME(exec, ForEach), buf_begin, buf_end, PostIncrementOp<int>());
    auto red_val = std::reduce(CLONE_TEST_POLICY(exec), buf_begin, buf_end, 1);
    EXPECT_TRUE(red_val == 2001, "wrong return value from reduce");

#endif // __SYCL_UNNAMED_LAMBDA__
}
#endif // TEST_DPCPP_BACKEND_PRESENT

// This is the simple test for compilation only, to check if lambda naming works correctly
int main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
