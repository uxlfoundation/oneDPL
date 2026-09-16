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

    // The permutable archetype family, i.e. the algorithms constrained by std::permutable or by
    // std::sortable: reverse, rotate, shift_left, shift_right, remove_if, unique, partition,
    // stable_partition, sort, stable_sort, is_sorted, and, with a non-const comparator only,
    // partial_sort, nth_element and inplace_merge. The first section calls them with const callables,
    // the second one with callables taking their arguments by reference, and the third one without a
    // callable at all, i.e. with the default std::ranges::less and std::ranges::equal_to.

    //----------------------------------------------------------------------------------------------
    // Const callables, and the algorithms which take no user callable at all.
    //----------------------------------------------------------------------------------------------
    // permutable_archetype is movable but not copyable, so it is not device copyable either: the
    // host policies are the only ones its constraints allow.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) { return dpl_ranges::reverse(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto) {
            const auto n = std::ranges::size(view);
            return std::ranges::begin(view)[0].val == (int)n - 1 && std::ranges::begin(view)[n - 1].val == 0;
        },
        "reverse");

    // rotate, shift_left and shift_right take no user callable at all: std::permutable, i.e. moving and
    // swapping through the iterator, is everything they are allowed to ask for. The range is 0, 1, 2,
    // ... so after rotating it by ten the tenth element is at the front and the old first one is ten
    // positions from the end.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::rotate(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 10 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 10].val == 0;
        },
        "rotate");

    // Shifting left by ten moves the eleventh element to the front.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::shift_left(std::forward<decltype(policy)>(policy), view, 10);
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[0].val == 10; }, "shift_left");

    // Shifting right by ten moves the first element ten positions to the right.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::shift_right(std::forward<decltype(policy)>(policy), view, 10);
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 0; }, "shift_right");

    // The storage is filled with 0, 1, 2, ... so every third element is removed. The returned
    // subrange is the tail holding the removed elements.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            const auto n = std::ranges::size(view);
            return std::ranges::size(res) == (n + 2) / 3;
        },
        "remove_if");

    // All the elements are unique, so nothing is dropped.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique");

    // partition returns the tail of the elements which do not satisfy the predicate.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition");

    // stable_partition is constrained exactly like partition, i.e. by std::permutable, so it may only
    // move and swap the elements through the iterator; the returned subrange is again the tail of the
    // elements which do not satisfy the predicate.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_partition(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "stable_partition");

    // The storage of the harness is filled in ascending order, so sorting it keeps it as it is: what
    // these two cases check is that the call compiles and leaves the range intact, not the ordering.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 10>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&&, auto res) { return res; }, "is_sorted");

    // The three algorithms whose non-const comparator counterparts below are broken: with a comparator
    // accepting a const lvalue they are expected to compile everywhere, which is what pins the defect to
    // the const-ness of the argument and not to the algorithm itself.
    // The range is ascending already, so the first ten elements are 0 ... 9 afterwards.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 11>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
        },
        "partial_sort");

    // partial_sort_copy writes the sorted prefix into a range of its own, so its input element only has
    // to be assignable into the output one and comparable with it, see family 14. The output range is
    // deliberately shorter than the input, so ten elements are copied and sorted, and the comparator
    // orders descending, so those are the ten largest values of the ascending input.
    //
    // KSATODO: the parallel host pattern of partial_sort_copy does not compile, and unlike the three
    // algorithms above the reason has nothing to do with the const-ness of the comparator argument:
    //  - algorithm_impl.h:2707 - the branch taken when the output range is shorter than the input one
    //    sorts in a temporary buffer of the output value type and fills it with
    //    ::new (__k) _T2(*__it), i.e. it constructs the output element from the input one, while
    //    std::indirectly_copyable only ever grants the assignment *__out = *__in. Assigning into the
    //    already initialized buffer element is not an option either, because std::sortable does not
    //    ask the output element for default construction; copying the input into the output range and
    //    sorting it in place, as the other branch at algorithm_impl.h:2686 does, needs no construction
    //    at all.
    // The serial pattern (algorithm_ranges_impl.h:617) forwards to std::ranges::partial_sort_copy and
    // is conforming by construction, so seq and unseq would compile; the gap macro covers the host side
    // as a whole and switches them off as well.
    {
        auto call = [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 10);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::partial_sort_copy(std::forward<decltype(policy)>(policy), view, out_view,
                                                     psort_copy_comp{});
            return std::ranges::begin(out_view)[0].val == (int)archetype_test_size - 1 &&
                   std::ranges::begin(out_view)[9].val == (int)archetype_test_size - 10 &&
                   res.out == std::ranges::end(out_view);
        };
        auto check = [](auto&&, auto res) { return res; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_COPY_HOST
        run_algo_host_policies<psort_copy_in_archetype>(call, check, "partial_sort_copy");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<psort_copy_in_archetype_dc, 12>(call, check, "partial_sort_copy");
#endif
    }

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 13>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                           permutable_comp{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; }, "nth_element");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "inplace_merge");

    //----------------------------------------------------------------------------------------------
    // Callables taking their arguments by non-const reference. The element of a permutable range is
    // mutable by definition, so its predicate and its comparator only ever see a non-const lvalue and
    // are not required to accept a const one.
    //----------------------------------------------------------------------------------------------
    // Every third element is removed; the returned subrange is the tail holding the removed elements.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == (std::ranges::size(view) + 2) / 3; },
        "remove_if, non-const callable");

    // All the elements are unique, so nothing is dropped.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 16>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv_mut{});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 0; }, "unique, non-const callable");

    // partition returns the tail of the elements which do not satisfy the predicate.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 17>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition, non-const callable");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 18>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "stable_partition, non-const callable");

    // The very same comparator as the one the sorting algorithms below are called with.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 19>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted of a permutable range, non-const comparator");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 20>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, non-const comparator");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 21>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, non-const comparator");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 22>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp_mut{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9; },
        "partial_sort, non-const comparator");

    // partial_sort_copy projects its input range and its output range with two distinct callables, so
    // both of them are passed here by non-const reference on top of the comparator, see family 14. The
    // gap of the const case above is independent of the callables and breaks the very same policies
    // here; the parallel host pattern additionally inherits the const comparator argument of the
    // parallel merge sort, see the note above partial_sort.
    //
    // KSATODO: the parallel host and the device patterns drop _Proj1 altogether
    // (algorithm_ranges_impl.h:608 and hetero/algorithm_ranges_impl_hetero.h:1701 build
    // __binary_op<_Comp, _Proj2, _Proj2>), so when the output range is shorter than the input one they
    // select the elements to keep by projecting copies of the input elements with the projection of the
    // output range, and never call the comparator with the mixed argument pair the requires-clause asks
    // for. That is a wrong result and not a compilation failure, so it stays invisible here: both
    // projections below return the element itself. The serial pattern forwards the two projections to
    // std::ranges::partial_sort_copy and is correct.
    {
        auto call = [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 10);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::partial_sort_copy(std::forward<decltype(policy)>(policy), view, out_view,
                                                     psort_copy_comp_mut{}, psort_copy_in_proj_mut{},
                                                     psort_copy_out_proj_mut{});
            return std::ranges::begin(out_view)[0].val == (int)archetype_test_size - 1 &&
                   std::ranges::begin(out_view)[9].val == (int)archetype_test_size - 10 &&
                   res.out == std::ranges::end(out_view);
        };
        auto check = [](auto&&, auto res) { return res; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_COPY_HOST
        run_algo_host_policies<psort_copy_in_archetype>(call, check, "partial_sort_copy, non-const callables");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<psort_copy_in_archetype_dc, 23>(call, check, "partial_sort_copy, non-const callables");
#endif
    }

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 24>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                           permutable_comp_mut{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; }, "nth_element, non-const comparator");

    // KSATODO: inplace_merge does not compile with any host policy, for two independent reasons:
    //  - algorithm_ranges_impl.h:848 - the serial path returns __end(__r), i.e. the sentinel of the
    //    range, while the declared return type is std::ranges::borrowed_iterator_t<_R>. For a range
    //    which is not a common_range the two types differ, so seq already fails to compile. This one
    //    is independent of the comparator and hits any user range with a distinct sentinel type;
    //  - the const lvalue of the merge split point is handed to the comparator, which std::sortable
    //    never asks for: std::inplace_merge compares against its const value parameter for unseq, and
    //    parallel_backend_tbb.h:1240,1245 does the same through std::upper_bound / std::lower_bound
    //    for par and par_unseq.
    // Both halves of the ascending range are sorted, so merging them keeps it as it is.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 25>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2,
                                             permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "inplace_merge, non-const comparator");

    //----------------------------------------------------------------------------------------------
    // The same algorithms called without a comparator at all, i.e. with the default
    // std::ranges::less: std::sortable then asks the element type itself for std::totally_ordered,
    // which is what permutable_ordered_archetype provides and nothing else. These calls are the only
    // ones which instantiate the default comparator path of the implementation, e.g.
    // __is_comp_ascending on the device side.
    //
    // The parallel host merge sort hands a const lvalue to the comparator, which is what breaks the
    // non-const comparator cases above; std::ranges::less accepts one, so par and par_unseq compile
    // here and all four host policies run.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 26>(
        [](auto&& policy, auto&& view) { return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, default comparator");

    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 27>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, default comparator");

    // The range is ascending already, so the first ten elements are 0 ... 9 afterwards.
    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 28>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view,
                                            std::ranges::begin(view) + 10);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
        },
        "partial_sort, default comparator");

    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 29>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view,
                                           std::ranges::begin(view) + 10);
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; },
        "nth_element, default comparator");

    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 30>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "inplace_merge, default comparator");

    // unique defaults its equivalence relation to std::ranges::equal_to, which needs the equality of
    // the element type and no ordering at all. All the elements are unique, so nothing is dropped.
    run_algo_all_policies<permutable_equality_archetype, permutable_equality_archetype_dc, 31>(
        [](auto&& policy, auto&& view) { return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique, default predicate");

    //----------------------------------------------------------------------------------------------
    // The sorting algorithms with a projection which maps the element to an integer key, see family
    // 15. std::sortable then asks the element for nothing but moving and swapping, and these are the
    // only calls of the whole suite which reach the radix sort of the device backend: it is selected
    // by the projected key type and never by the element type, so an archetype element can only get
    // there through such a projection.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 32>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, std::ranges::less{},
                                    permutable_proj_key{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, projected key");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 33>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, std::ranges::less{},
                                           permutable_proj_key{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, projected key");

    // The permuting pattern over plain_archetype_view, i.e. over a range without the members
    // std::ranges::view_interface provides; see the plain range section of the read test for what this
    // proves. sort is the representative shape here, because it is the pattern which splits the range
    // into sub ranges of its own and therefore has the most reasons to ask the user range for its size.
    run_algo_plain_all_policies<permutable_archetype, permutable_archetype_dc, 34>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, plain range");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
