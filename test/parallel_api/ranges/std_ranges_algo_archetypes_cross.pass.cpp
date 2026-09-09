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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include "std_ranges_archetypes.h"
#include "std_ranges_algo_archetypes_test.h"
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // The lhs_archetype/rhs_archetype family: the two-range algorithms which compare the elements of two
    // ranges of unrelated types through a user callable only.
    // Covers equal, mismatch, search, find_end, find_first_of and includes, first with const callables
    // and then with callables taking their arguments by non-const reference.

    // Two ranges of unrelated element types, compared only through the user predicate.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch");

    // The two ranges hold the very same sequence, so the second one occurs in the first one exactly
    // once, at its very beginning.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "search");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end");

    // KSATODO: std::indirectly_comparable<_It1, _It2, _Pred> only requires the predicate to be
    // invocable as __pred(*__it1, *__it2), never the other way round. The SIMD brick swaps the two
    // arguments, so the vectorized host policies unseq and par_unseq do not compile:
    //  - unseq_backend_simd.h:827 - __simd_find_first_of builds __u_pred as
    //    __pred(__val, *__first) with __val taken from the second range and *__first from the first
    //    one; the branch is a plain if, so it is instantiated whatever the sizes of the ranges are.
    // Fixing this means keeping the argument order of the two ranges in both branches.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HOST>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); }, "find_first_of");

    // includes needs a comparator accepting the two element types in all four combinations, see
    // cross_comp. Both ranges hold the very same ascending sequence, so the second one is included in
    // the first one.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch, non-const callable");

    // The two ranges hold the very same sequence, so the second one occurs in the first one exactly
    // once, at its very beginning.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, non-const callable");

    // KSATODO: std::indirectly_comparable<_It1, _It2, _Pred> only requires the predicate to be
    // invocable as __pred(*__it1, *__it2), never the other way round. The SIMD brick swaps the two
    // arguments, so the vectorized host policies unseq and par_unseq do not compile:
    //  - unseq_backend_simd.h:827 - __simd_find_first_of builds __u_pred as
    //    __pred(__val, *__first) with __val taken from the second range and *__first from the first
    //    one; the branch is a plain if, so it is instantiated whatever the sizes of the ranges are.
    // Fixing this means keeping the argument order of the two ranges in both branches.
    //
    // KSATODO: the device path of find_first_of copies the element of the first range into a const
    // local, which std::indirectly_comparable neither asks for nor allows to require, so the call does
    // not compile:
    //  - unseq_backend_sycl.h:632,636 - first_match_pred::operator() writes
    //    const auto __elem = __acc[__shifted_idx]; and passes __elem to the predicate. A forwarding
    //    reference instead of the const copy fixes both the const-ness and the extra copy.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); },
        "find_first_of, non-const callable");

    // includes needs a comparator accepting the two element types in all four combinations, see
    // cross_comp_mut. Both ranges hold the very same ascending sequence, so the second one is included
    // in the first one.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, non-const comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
