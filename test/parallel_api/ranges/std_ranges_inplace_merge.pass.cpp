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

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
// Adaptors for the test harness to pass a middle iterator.
struct inplace_merge_dpl_fn
{
    template <typename Policy, std::ranges::random_access_range _R, typename _Comp = std::ranges::less,
              typename _Proj = std::identity>
    std::ranges::borrowed_iterator_t<_R>
    operator()(Policy&& exec, _R&& r, _Comp comp = {}, _Proj proj = {}) const
    {
        auto middle = std::ranges::begin(r) + std::ranges::size(r) / 2;
        return oneapi::dpl::ranges::inplace_merge(std::forward<Policy>(exec), std::forward<_R>(r), middle, comp, proj);
    }
} inplace_merge_dpl;

struct inplace_merge_checker_fn
{
    template <std::ranges::random_access_range _R, typename _Comp = std::ranges::less, typename _Proj = std::identity>
    std::ranges::borrowed_iterator_t<_R>
    operator()(_R&& r, _Comp comp = {}, _Proj proj = {}) const
    {
        auto middle = std::ranges::begin(r) + std::ranges::size(r) / 2;
        return std::ranges::inplace_merge(std::forward<_R>(r), middle, comp, proj);
    }
} inplace_merge_checker;

struct stable_data_gen_fn
{
    test_std_ranges::P2
    operator()(int i) const
    {
        return test_std_ranges::P2(/*x = key*/ 0, /*y = payload*/ i);
    }
};
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // The default data generator (std::identity) produces an ascending sequence, so both halves
    // [begin, middle) and [middle, end) are sorted with respect to std::ranges::less.

    test_range_algo<0>{big_sz}(inplace_merge_dpl, inplace_merge_checker);
    test_range_algo<1>{}(inplace_merge_dpl, inplace_merge_checker, std::ranges::less{});

    test_range_algo<2>{}(inplace_merge_dpl, inplace_merge_checker, std::ranges::less{}, proj);

    test_range_algo<3, P2>{}(inplace_merge_dpl, inplace_merge_checker, std::ranges::less{}, &P2::x);
    test_range_algo<4, P2>{}(inplace_merge_dpl, inplace_merge_checker, std::ranges::less{}, &P2::proj);

    // Stability check: equal projected keys with distinct payloads must preserve relative order.
    test_range_algo<5, P2, data_in, stable_data_gen_fn>{}(inplace_merge_dpl, inplace_merge_checker,
                                                          std::ranges::less{}, &P2::x);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
