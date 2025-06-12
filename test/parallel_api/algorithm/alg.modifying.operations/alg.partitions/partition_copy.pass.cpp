// -*- C++ -*-
//===-- partition_copy.pass.cpp -------------------------------------------===//
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

// Tests for stable_partition and partition_copy
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <cstdlib>
#include <iterator>

using namespace TestUtils;

template <typename T>
struct test_partition_copy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2,
              typename UnaryOp>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator true_first,
               OutputIterator, OutputIterator2 false_first, OutputIterator2, UnaryOp unary_op)
    {

        auto actual_ret = ::std::partition_copy(std::forward<Policy>(exec), first, last, true_first, false_first, unary_op);

        EXPECT_TRUE(::std::distance(true_first, actual_ret.first) == ::std::count_if(first, last, unary_op),
                    "partition_copy has wrong effect from true sequence");
        EXPECT_TRUE(::std::distance(false_first, actual_ret.second) ==
                        ::std::count_if(first, last, oneapi::dpl::__internal::__not_pred<UnaryOp>(unary_op)),
                    "partition_copy has wrong effect from false sequence");
    }

//dummy specialization in case of broken configurations
#if _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN
template <typename InputIterator, typename OutputIterator, typename OutputIterator2,
          typename UnaryOp>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, InputIterator /* first */, InputIterator /* last */, OutputIterator /* true_first */,
               OutputIterator /* true_last */, OutputIterator2 /* false_first */, OutputIterator2 /* false_last */, /* UnaryOp unary_op */)
    {
    }

    template <typename InputIterator, typename OutputIterator, typename OutputIterator2,
              typename UnaryOp>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, InputIterator /* first */, InputIterator /* last */, OutputIterator /* true_first */,
               OutputIterator /* true_last */, OutputIterator2 /* false_first */, OutputIterator2 /* false_last */, /* UnaryOp unary_op */)
    {
    }
#endif
};

template <typename T, typename UnaryPred>
void
test(UnaryPred pred)
{

    const ::std::size_t max_size = 100000;
    Sequence<T> in(max_size, [](::std::size_t v) -> T { return T(v); });
    Sequence<T> actual_true(max_size);
    Sequence<T> actual_false(max_size);
    for (::std::size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : ::std::size_t(3.1415 * n))
    {

        // for non-const input iterators
        invoke_on_all_policies<0>()(test_partition_copy<T>(), in.begin(), in.begin() + n, actual_true.begin(),
                                    actual_true.begin() + n, actual_false.begin(), actual_false.begin() + n, pred);

        // for const input iterators
        invoke_on_all_policies<1>()(test_partition_copy<T>(), in.cbegin(), in.cbegin() + n, actual_true.begin(),
                                    actual_true.begin() + n, actual_false.begin(), actual_false.begin() + n, pred);
    }
}

struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        partition_copy(std::forward<Policy>(exec), input_iter, input_iter, out_iter, out_iter, non_const(TestUtils::IsEven<float64_t>{}));
    }
};

int
main()
{
    test<std::int16_t>(TestUtils::IsEven<std::int32_t>{});
    test<std::int32_t>([](const std::int32_t) { return true; });

#if !ONEDPL_FPGA_DEVICE
    test<float64_t>([](const float64_t value) { return value > 2 << 6; });
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test<Wrapper<float64_t>>([](const Wrapper<float64_t>& value) -> bool { return value.get_my_field() != nullptr; });
#endif

    test_algo_basic_double<std::int32_t>(run_for_rnd_bi<test_non_const>());

    return done();
}
