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
    // The serial pattern (algorithm_ranges_impl.h:521) forwards to std::ranges::partial_sort_copy and
    // is conforming by construction, so seq and unseq would compile; the gap macro covers the host side
    // as a whole and switches them off as well.
    //
    // KSATODO: the device path assigns the output element from a const lvalue of the input one, which
    // std::indirectly_copyable does not ask for, exactly like rotate_copy:
    //  - hetero/algorithm_impl_hetero.h:1512,1541,1556 - the three initial copies go through
    //    __pattern_hetero_walk2 with the input read through an access_mode::read accessor, so
    //    __brick_copy (hetero/algorithm_impl_hetero.h:397) assigns from a const _Tp&.
    // Requesting read_write access for the input of those walks is enough to fix it.
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
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_COPY_HETERO
        run_algo_hetero_policies<psort_copy_in_archetype_dc, 12>(call, check, "partial_sort_copy");
#endif
    }

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 13>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                           permutable_comp{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; }, "nth_element");

    // KSATODO: inplace_merge does not compile with any host policy even with a const comparator, because
    // of a defect which has nothing to do with the comparator at all:
    //  - algorithm_ranges_impl.h:848 - the serial path returns __end(__r), i.e. the sentinel of the
    //    range, while the declared return type is std::ranges::borrowed_iterator_t<_R>. The two types
    //    differ for every range which is not a common_range, so seq already fails to compile.
    // Both halves of the ascending range are sorted, so merging them keeps it as it is.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2,
                                             permutable_comp{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "inplace_merge");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 14>(call, check, "inplace_merge");
#endif
    }

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
    //
    // KSATODO: the device path of partition applies the predicate to a const lvalue, which
    // std::indirect_unary_predicate over a permutable iterator does not ask for, so it does not
    // compile:
    //  - unseq_backend_sycl.h:122 - walk_n::operator() is const and calls __f(__rngs[__idx]...) on
    //    the const range members of single_match_pred_by_idx.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "partition, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 17>(call, check, "partition, non-const callable");
#endif
    }

    // KSATODO: the device path of stable_partition applies the predicate to a const lvalue, which
    // std::indirect_unary_predicate over a permutable iterator does not ask for, so it does not
    // compile:
    //  - utils.h:187 - __unary_op::operator() is const and forwards what it is handed into std::invoke;
    //  - unseq_backend_sycl.h:555,557 - single_match_pred_by_idx passes the accessor on as const, so
    //    walk_n subscripts a const range;
    //  - parallel_backend_sycl_reduce_then_scan.h:473,475 - __gen_mask::operator() is const as well and
    //    subscripts the range it is handed, which reaches it as a const one.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "stable_partition, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_PARTITION_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 18>(call, check, "stable_partition, non-const callable");
#endif
    }

    // The very same comparator as the one the sorting algorithms below are called with.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 19>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted of a permutable range, non-const comparator");

    // KSATODO: std::sortable<_It, _Comp> only requires the comparator to be invocable with
    // iter_reference_t<_It>, which is a non-const lvalue for archetype_view, so a comparator taking
    // its arguments by non-const reference is enough. The parallel host merge sort compares against a
    // const lvalue instead, so par and par_unseq do not compile:
    //  - parallel_backend_tbb.h:1037 - std::lower_bound(..., _M_comp) passes the const lvalue _Val
    //    of the merge split point to the comparator;
    //  - utils.h:203 - __binary_op::operator() forwards that const lvalue into std::invoke.
    // seq and unseq keep the element non-const all the way down, but the gap macro covers the host side
    // as a whole, so they are switched off together with par and par_unseq.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "sort, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 20>(call, check, "sort, non-const comparator");
#endif
    }

    // KSATODO: stable_sort shares the merge sort of the parallel host policies with sort, so it is
    // broken for par and par_unseq in exactly the same way, see the note above.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "stable_sort, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 21>(call, check, "stable_sort, non-const comparator");
#endif
    }

    // KSATODO: partial_sort shares the parallel merge sort with sort, so the parallel host policies
    // hand a const lvalue to the comparator here as well and par / par_unseq do not compile:
    //  - parallel_backend_tbb.h:1023,1026,1034 - __merge_func::split_merging passes *(_M_x_beg + __ym)
    //    to std::upper_bound / std::lower_bound, which compares against their const lvalue parameter;
    //  - utils.h:203 - __binary_op::operator() forwards that const lvalue into std::invoke.
    // seq and unseq keep the element non-const all the way down, but the gap macro covers the host side
    // as a whole, so they are switched off together with par and par_unseq.
    // The range is ascending already, so the first ten elements are 0 ... 9 afterwards.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "partial_sort, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 22>(call, check, "partial_sort, non-const comparator");
#endif
    }

    // partial_sort_copy projects its input range and its output range with two distinct callables, so
    // both of them are passed here by non-const reference on top of the comparator, see family 14. The
    // two gaps of the const case above are independent of the callables and break the very same
    // policies here; the parallel host pattern additionally inherits the const comparator argument of
    // the parallel merge sort, see the note above partial_sort.
    //
    // KSATODO: the parallel host and the device patterns drop _Proj1 altogether
    // (algorithm_ranges_impl.h:503,513 and hetero/algorithm_ranges_impl_hetero.h:1528,1536 build
    // __binary_op<_Comp, _Proj2, _Proj2>), so they project the input range with the projection of the
    // output range and never call the comparator with the mixed argument pair the requires-clause asks
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
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_COPY_HETERO
        run_algo_hetero_policies<psort_copy_in_archetype_dc, 23>(call, check, "partial_sort_copy, non-const callables");
#endif
    }

    // KSATODO: std::sortable only requires the comparator to be invocable with the non-const
    // reference of the element, but the parallel path of nth_element compares against a const lvalue,
    // so par and par_unseq do not compile:
    //  - algorithm_impl.h:2841 - the partition predicate of the quickselect loop takes const _Tp& and
    //    passes it into std::invoke(__comp, __x, *__first).
    // Taking the element by reference in that lambda is enough to fix it. seq and unseq are fine, but the
    // gap macro covers the host side as a whole and switches them off as well.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                           permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_NTH_ELEMENT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "nth_element, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 24>(call, check, "nth_element, non-const comparator");
#endif
    }

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
    //
    // KSATODO: the device path of inplace_merge compares two const lvalues, which std::sortable does
    // not ask for, so the call does not compile:
    //  - parallel_backend_sycl_merge.h:128-133 - the lambda of __find_start_point captures __rng1 and
    //    __rng2 and subscripts them as const, and both results go into the comparator.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2,
                                             permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "inplace_merge, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 25>(call, check, "inplace_merge, non-const comparator");
#endif
    }

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

    // KSATODO: the host defect of inplace_merge is the return type of its serial path, see the note
    // above, so it stays broken whatever the comparator is. The device defect is the const-ness of the
    // comparator arguments only, which std::ranges::less accepts, so the device call is expected to
    // compile here.
    // Both halves of the ascending range are sorted, so merging them keeps it as it is.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2);
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HOST
        run_algo_host_policies<permutable_ordered_archetype>(call, check, "inplace_merge, default comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_ordered_archetype_dc, 30>(call, check, "inplace_merge, default comparator");
#endif
    }

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
    //----------------------------------------------------------------------------------------------
    // KSATODO: the radix sort these two calls select on the device takes the address of the element
    // with a plain operator&, which nothing in std::sortable asks the element type for:
    //  - parallel_backend_sycl_radix_sort_one_wg.h:123 - new (&__values[__i]) _ValueT(__src[__idx])
    //    in __block_load, which is also what makes the __block_load call at :200 fail to resolve;
    //  - parallel_backend_sycl_radix_sort_one_wg.h:311 - new (&__exchange_lacc[__r]) _ValT(...).
    // Spelling both of them std::addressof is enough; the placement new itself is legitimate here,
    // because the local storage of the kernel is raw memory. Note that the same two lines also copy
    // construct, respectively move construct, the element, which std::sortable does allow for the move
    // and does not for the copy at :123 - a device archetype has to be trivially copyable, so this test
    // cannot tell the two apart and the addressof is the only part it pins down.
    // The host policies have no radix sort at all and are expected to compile.
    // The range is ascending already, so sorting it keeps it as it is.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, std::ranges::less{},
                                    permutable_proj_key{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "sort, projected key");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_RADIX_SORT_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 32>(call, check, "sort, projected key");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, std::ranges::less{},
                                           permutable_proj_key{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "stable_sort, projected key");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_RADIX_SORT_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 33>(call, check, "stable_sort, projected key");
#endif
    }

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
