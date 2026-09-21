// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    // Covers equal, mismatch, search, find_end, find_first_of, includes, contains_subrange, starts_with,
    // ends_with and lexicographical_compare, first with const callables, then with callables taking
    // their arguments by non-const reference, and finally without a callable at all, i.e. with the
    // default std::ranges::equal_to and std::ranges::less.

    // Two ranges of unrelated element types, compared only through the user predicate.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 0>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 1>(
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
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 2>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "search");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 3>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end");

    // KSATODO: the vectorized brick of find_first_of runs the shorter of the two sequences in the
    // outer loop, and in the branch which takes the first sequence for the shorter one
    // (unseq_backend_simd.h:808) it passes the element of the second sequence as the first argument
    // of the predicate. std::indirect_binary_predicate asks the predicate for that one order of the
    // two element types which the call spells out, so cross_pred does not accept the swapped pair.
    // Fixing this means wrapping the predicate of that branch in __reorder_pred, as the brick of
    // __simd_first does. The device brick of find_first_of has a gap of its own, see the non-const
    // predicate call below; the two share one macro, so this call is off for every policy.
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 4>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); }, "find_first_of");
#endif

    // includes needs a comparator accepting the two element types in all four combinations, see
    // cross_comp. Both ranges hold the very same ascending sequence, so the second one is included in
    // the first one.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 5>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes");

    // contains_subrange, starts_with and ends_with are constrained by std::indirectly_comparable just
    // like search, so they see the user predicate only. The two ranges hold the very same sequence, so
    // the second one is a subrange of the first one and is both its prefix and its suffix.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 6>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::contains_subrange(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "contains_subrange");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 7>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::starts_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "starts_with");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 8>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::ends_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "ends_with");

    // lexicographical_compare orders the two ranges, so it needs a comparator accepting the two element
    // types in all four combinations, see cross_comp. The two ranges are equal, so neither is less than
    // the other.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 9>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::lexicographical_compare(std::forward<decltype(policy)>(policy), view1, view2,
                                                        cross_comp{});
        },
        [](auto&&, auto&&, bool res) { return !res; }, "lexicographical_compare");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 10>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 11>(
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
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 12>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 13>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, non-const callable");

    // KSATODO: the host side hits the very same argument swap of unseq_backend_simd.h:808 as the
    // const callable above.
    //
    // KSATODO: the device brick of find_first_of, first_match_pred at unseq_backend_sycl.h:632,
    // reads the element of the first sequence into a const copy and hands that to the predicate,
    // while std::indirect_binary_predicate only asks it to accept iter_reference_t, i.e. a non-const
    // lvalue. Taking that element by reference, as the loop below already reads the second sequence,
    // is enough to fix it; the const copy additionally asks the element for a copy constructor which
    // the algorithm does not need.
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 14>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); },
        "find_first_of, non-const callable");
#endif

    // includes needs a comparator accepting the two element types in all four combinations, see
    // cross_comp_mut. Both ranges hold the very same ascending sequence, so the second one is included
    // in the first one.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 15>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, non-const comparator");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 16>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::contains_subrange(std::forward<decltype(policy)>(policy), view1, view2,
                                                    cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "contains_subrange, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 17>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::starts_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "starts_with, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 18>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::ends_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "ends_with, non-const callable");

    // KSATODO: the device pattern of lexicographical_compare, the transform functor at
    // hetero/utils_hetero.h:138,139, binds the two elements to an auto const& local and compares
    // those, so a comparator taking its arguments by non-const reference does not compile, although
    // std::indirect_strict_weak_order only asks it to accept iter_reference_t. Both reads come from a
    // read-only accessor, so dropping the const of the two locals, i.e. holding them by decltype(auto),
    // is enough. The host side is conforming here: its vectorized brick does swap the two elements, but
    // std::strict_weak_order asks the comparator for both orders anyway, so cross_comp_mut accepts the
    // swapped pair. That swap is a defect only once the call has projections of its own, where it hands
    // the first projection an element of the second sequence; see
    // _TEST_CPP20_RANGES_BROKEN_REQUIRES_LEXICOGRAPHICAL_COMPARE_HOST and
    // _TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_LEXICOGRAPHICAL_COMPARE_PROJ1_HOST in
    // std_ranges_lexicographical_compare.pass.cpp.
    {
        auto call = [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::lexicographical_compare(std::forward<decltype(policy)>(policy), view1, view2,
                                                        cross_comp_mut{});
        };
        auto check = [](auto&&, auto&&, bool res) { return !res; };

        run_algo2_host_policies<lhs_archetype, rhs_archetype>(call, check,
                                                                "lexicographical_compare, non-const comparator");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_LEXICOGRAPHICAL_COMPARE_HETERO
        run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 19>(
            call, check, "lexicographical_compare, non-const comparator");
#endif
    }

    //----------------------------------------------------------------------------------------------
    // The same algorithms called without a callable at all, i.e. with the default
    // std::ranges::equal_to and std::ranges::less. There is no user callable left to relate the two
    // element types, so both ranges hold the very same archetype and the comparison comes from it:
    // equality_archetype has operator== only, ordered_archetype is std::totally_ordered.
    //----------------------------------------------------------------------------------------------
    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 20>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 21>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                    res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch, default predicate");

    // The two ranges hold the very same sequence, so the second one occurs in the first one exactly
    // once, at its very beginning.
    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 22>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 23>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 24>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); },
        "find_first_of, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 25>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::contains_subrange(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "contains_subrange, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 26>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::starts_with(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "starts_with, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 27>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::ends_with(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "ends_with, default predicate");

    // Both ranges hold the very same ascending sequence, so the second one is included in the first one
    // and neither of them is lexicographically less than the other.
    run_algo2_all_policies<ordered_archetype, ordered_archetype, ordered_archetype_dc, ordered_archetype_dc, 28>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, default comparator");

    run_algo2_all_policies<ordered_archetype, ordered_archetype, ordered_archetype_dc, ordered_archetype_dc, 29>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::lexicographical_compare(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return !res; }, "lexicographical_compare, default comparator");

    // The two-range read pattern over plain_archetype_view, i.e. over ranges without the members
    // std::ranges::view_interface provides; see the plain range section of the read test for what this
    // proves. equal is the representative shape here: two ranges walked in lockstep, with the number of
    // elements coming from the sized sentinel of each of them and not from a size() member.
    run_algo2_plain_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 30>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, plain ranges");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
