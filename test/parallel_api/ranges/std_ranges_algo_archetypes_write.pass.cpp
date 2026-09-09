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

    // The write archetype families: an element which is only assignable, from an unrelated value type
    // (fill, replace_if, replace), from the element of another range (copy, reverse_copy, rotate_copy,
    // copy_if, remove_copy_if, remove_copy, unique_copy, partition_copy, move, swap_ranges) or from the
    // result of a functor (transform, with and without projections). Nothing here is copyable, movable
    // or default constructible, and the elements written from are of a different type than the ones
    // written to. The elements of replace and remove_copy are additionally equality comparable with the
    // searched value of family 2, which is all their requires-clauses add.

    //----------------------------------------------------------------------------------------------
    // The writing algorithms; every callable takes its arguments by const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<writable_archetype, writable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::fill(std::forward<decltype(policy)>(policy), view, typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == 42;
        },
        "fill");

    // The output range of copy, move and swap_ranges is filled with the offset values, so that the value
    // the check reads is one which only the algorithm itself can have put there, see run_algo2_offset.
    run_algo2_offset_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc,
                                  1>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            const auto n = std::ranges::size(out_view);
            return std::ranges::begin(out_view)[7].val == std::ranges::begin(in_view)[7].val &&
                   std::ranges::begin(out_view)[n - 1].val == std::ranges::begin(in_view)[n - 1].val;
        },
        "copy");

    // reverse_copy and rotate_copy are constrained by std::indirectly_copyable just like copy, so they
    // may only assign the element of the input range to the one of the output range. Both storages start
    // as 0, 1, 2, ..., so a position which ends up holding a different value tells the assignment really
    // happened: reverse_copy writes the last input element at the front of the output range, and
    // rotate_copy writes the one the middle iterator points at.
    run_algo2_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc, 2>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::reverse_copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == std::ranges::begin(in_view)[n - 1].val &&
                   std::ranges::begin(out_view)[n - 1].val == std::ranges::begin(in_view)[0].val;
        },
        "reverse_copy");

    // KSATODO: std::indirectly_copyable only asks for *__out = *__in, i.e. for an assignment from
    // iter_reference_t of the input iterator, which is a non-const lvalue for archetype_view. The device
    // path assigns from a const prvalue instead, so the call does not compile:
    //  - unseq_backend_sycl.h:885 - __rotate_copy::operator() writes __rng2[__idx] = __rng1[__shifted]
    //    with __rng1 a const all_view of access mode read, whose subscript returns const _Elem by value.
    // Assigning through a non-const reference to the input element fixes it; the host path already does.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::rotate_copy(std::forward<decltype(policy)>(policy), in_view,
                                           std::ranges::begin(in_view) + 10, out_view);
        };
        auto check = [](auto&& in_view, auto&& out_view, auto) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == std::ranges::begin(in_view)[10].val &&
                   std::ranges::begin(out_view)[n - 10].val == std::ranges::begin(in_view)[0].val;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "rotate_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_ROTATE_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 3>(call, check, "rotate_copy");
#endif
    }

    run_algo2_offset_all_policies<move_in_archetype, move_out_archetype, move_in_archetype_dc, move_out_archetype_dc,
                                  4>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::move(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 7; }, "move");

    // Both ranges of swap_ranges hold the very same element type, so the offset fill is the only thing
    // which tells them apart: after the swap the first range holds the offset values and the second one
    // the ascending ones.
    run_algo2_offset_all_policies<swap_archetype, swap_archetype, swap_archetype_dc, swap_archetype_dc, 5>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::swap_ranges(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto) {
            return std::ranges::begin(view1)[7].val == (int)archetype_test_size + 7 &&
                   std::ranges::begin(view2)[7].val == 7;
        },
        "swap_ranges");

    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 6>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; }, "transform");

    // The same overload with a non-identity projection: the functor is invoked with the projected
    // value, which is neither the element nor the output element type.
    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 7>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, projection");

    // The binary overload takes two input ranges, so the output range is allocated inside the call
    // and the check is done there as well.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 8>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary");

    // The binary overload has a projection of its own for either input.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 9>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_projected_binary_op{}, transform_proj{}, transform_proj{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, projections");

    // The conditionally copying algorithms. Their callable is spelled over the projected input iterator
    // alone, so std::indirectly_copyable remains the whole output requirement: the implementation may
    // only assign an input element to an output one. Both storages start as 0, 1, 2, ..., so an output
    // position which ends up holding a different value proves the assignment really happened.
    // KSATODO: the device paths of copy_if, remove_copy_if and remove_copy (which are copy_if with a
    // negated predicate, respectively with a negated equality against the searched value), unique_copy
    // and partition_copy all assign a const copy of the input element to the output one,
    // which std::indirectly_copyable never asks for: it only requires *__out = *__in, with *__in the
    // non-const lvalue iter_reference_t of the input iterator. The places to fix are
    //  - utils.h:170-172 - __pstl_assign::operator() takes the source by const lvalue reference, so
    //    every writer below hands the output element a const source;
    //  - parallel_backend_sycl.h:298 - the single group copy_if functor materializes a copy of the
    //    input element with static_cast<__tuple_type>(__in_rng[__idx]) and assigns from that prvalue;
    //  - parallel_backend_sycl_reduce_then_scan.h:164,182 - __write_to_id_if::operator() assigns from
    //    the element it reads out of the const tuple __v gathered beforehand, i.e. from a const copy as
    //    well; it is reached from lines 1695 and 1709 for every one of the four algorithms;
    //  - parallel_backend_sycl_reduce_then_scan.h:2158 - the unique pattern additionally copies the
    //    0th element with __write_op.__assign(__in_rng[0], __out_rng[0]);
    //  - parallel_backend_sycl_reduce_then_scan.h:238,262 - partition_copy does not use __pstl_assign,
    //    but __write_partitioned::operator() destructures the very same const tuple and assigns
    //    __tuple_type_cast(__value, ...), which is a prvalue copy, to either output range.
    // Taking the source of __pstl_assign by forwarding reference and dropping the casts to a value
    // fixes all of them; the host paths already assign from the reference itself.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            // copy_pred keeps every third element, so the output holds 0, 3, 6, ...
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "copy_if");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 10>(call, check, "copy_if");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            // The negation of copy_pred, so the output holds 1, 2, 4, 5, 7, ...
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == 1 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "remove_copy_if");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 11>(call, check, "remove_copy_if");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::unique_copy(std::forward<decltype(policy)>(policy), in_view, out_view, copy_equiv{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            // copy_equiv groups the input into buckets of three, so the output holds 0, 3, 6, ...
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "unique_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_UNIQUE_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 12>(call, check, "unique_copy");
#endif
    }

    // partition_copy needs a second output range, which is allocated inside the call, so the check is
    // done there as well.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_true_view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(in_view)>>;
            auto out_false_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_false_view = out_false_storage.view();
            auto res = dpl_ranges::partition_copy(std::forward<decltype(policy)>(policy), in_view, out_true_view,
                                                  out_false_view, copy_pred{});
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_true_view)[1].val == 3 && std::ranges::begin(out_false_view)[0].val == 1 &&
                   (std::size_t)(res.out1 - std::ranges::begin(out_true_view)) == (n + 2) / 3 &&
                   (std::size_t)(res.out2 - std::ranges::begin(out_false_view)) == n - (n + 2) / 3;
        };
        auto check = [](auto&&, auto&&, auto res) { return res; };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "partition_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 13>(call, check, "partition_copy");
#endif
    }

    // replace_if writes the unrelated value type into the range itself, exactly like fill, and takes a
    // predicate over the element on top of it.
    run_algo_all_policies<writable_archetype, writable_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace_if(std::forward<decltype(policy)>(policy), view, write_pred{},
                                          typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 && std::ranges::begin(view)[1].val == 1 &&
                   std::ranges::begin(view)[3].val == 42;
        },
        "replace_if");

    // replace is the write family and the value family at once: the old value is compared with
    // std::ranges::equal_to and the new value is written into the range as replace_if does. The two
    // value types are unrelated to each other and to the element, so an implementation which confuses
    // them, or which assigns an element instead of the new value, does not compile.
    run_algo_all_policies<replaceable_archetype, replaceable_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace(std::forward<decltype(policy)>(policy), view, search_value{3},
                                       typename elem_t::value_arg{42});
        },
        // The storage holds 0, 1, 2, ..., so only the element equal to the old value 3 is overwritten.
        [](auto&& view, auto) {
            return std::ranges::begin(view)[3].val == 42 && std::ranges::begin(view)[2].val == 2;
        },
        "replace");

    // remove_copy is the copying family and the value family at once: it drops the elements equal to
    // the searched value and assigns the surviving ones to the output range. It delegates to copy_if
    // with a negated equality predicate, so its device path assigns a const copy of the input element
    // exactly like the conditionally copying algorithms above and is guarded for the same reason.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            // The value 3 occurs exactly once, so the output holds 0, 1, 2, 4, 5, ...
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[3].val == 4 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - 1;
        };

        run_algo2_host_policies<remove_copy_in_archetype, copy_out_archetype>(call, check, "remove_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_HETERO
        run_algo2_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 16>(call, check, "remove_copy");
#endif
    }

    // replace_copy_if and replace_copy write either an input element or the new value into the output
    // range, so the output element is assignable from both and from nothing else. Both cases are checked:
    // a replaced position against the new value, and a kept one against the input element, which the
    // offset fill of the output range makes a real check, see run_algo2_offset.
    // KSATODO: every host path of the two but the serial scalar one stores the new value by value and
    // therefore copy constructs it, which std::indirectly_writable<iterator_t<_OutR>, const _T&> never
    // asks for: it only needs *__out = __value for a const lvalue value. The places to fix are
    //  - utils.h:451,455 - __replace_copy_functor holds the value as a const _Tp member and copy
    //    constructs it in its constructor;
    //  - algorithm_ranges_impl.h:1753 - __pattern_replace_copy_if instantiates that functor with _T
    //    deduced from its own const _T& parameter, which drops the __ref_or_copy reference the CPO hands
    //    it at glue_algorithm_ranges_impl.h:1182.
    // Holding the __ref_or_copy type instead of a value fixes both. The serial scalar overload
    // (algorithm_ranges_impl.h:1761) already forwards the reference to std::ranges::replace_copy_if, and
    // the device path (hetero/algorithm_ranges_impl_hetero.h:952) legitimately copies the value into the
    // kernel, so the serial scalar branch, i.e. seq, is the only conforming host one. The gap macro covers
    // the host side as a whole, so all four host policies are guarded here.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred{},
                                               typename out_t::value_arg{42});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            // copy_pred holds for every third element, which is replaced with 42.
            return std::ranges::begin(out_view)[0].val == 42 && std::ranges::begin(out_view)[3].val == 42 &&
                   std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST
        run_algo2_offset_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "replace_copy_if");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 17>(call, check,
                                                                                         "replace_copy_if");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3},
                                            typename out_t::value_arg{42});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            // Only the element equal to the old value 3 is replaced with 42.
            return std::ranges::begin(out_view)[3].val == 42 && std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST
        run_algo2_offset_host_policies<remove_copy_in_archetype, copy_out_archetype>(call, check, "replace_copy");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 18>(call, check,
                                                                                                "replace_copy");
#endif
    }

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 19>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; },
        "transform, non-const callable");

    // The projection is the one taking the element by non-const reference here: the functor is
    // invoked with the projected prvalue and cannot take it by non-const reference at all.
    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 20>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, non-const projection");

    // The binary overload with a functor taking both input elements by non-const reference. It takes
    // two input ranges, so the output range is allocated inside the call and checked there as well.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 21>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op_mut{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, non-const callable");

    // The binary overload with a non-const projection for either input.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 22>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                      transform_projected_binary_op{}, transform_proj_mut{}, transform_proj_mut{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, non-const projections");

    // The conditionally copying algorithms with a predicate, respectively an equivalence relation,
    // taking the input element by non-const reference: the projected reference of archetype_view is a
    // non-const lvalue, so an implementation which copies the element, or hands a const one to the
    // callable, does not compile.
    // The three algorithms whose device path is guarded above are guarded here for the very same
    // reason: the const copy of the input element breaks the call before the predicate is reached.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "copy_if, non-const predicate");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 23>(call, check,
                                                                                   "copy_if, non-const predicate");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view,
                                              copy_pred_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == 1 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check,
                                                                       "remove_copy_if, non-const predicate");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 24>(
            call, check, "remove_copy_if, non-const predicate");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::unique_copy(std::forward<decltype(policy)>(policy), in_view, out_view,
                                           copy_equiv_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "unique_copy, non-const relation");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_UNIQUE_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 25>(call, check,
                                                                                   "unique_copy, non-const relation");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_true_view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(in_view)>>;
            auto out_false_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_false_view = out_false_storage.view();
            auto res = dpl_ranges::partition_copy(std::forward<decltype(policy)>(policy), in_view, out_true_view,
                                                  out_false_view, copy_pred_mut{});
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_true_view)[1].val == 3 && std::ranges::begin(out_false_view)[0].val == 1 &&
                   (std::size_t)(res.out1 - std::ranges::begin(out_true_view)) == (n + 2) / 3 &&
                   (std::size_t)(res.out2 - std::ranges::begin(out_false_view)) == n - (n + 2) / 3;
        };
        auto check = [](auto&&, auto&&, auto res) { return res; };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check,
                                                                       "partition_copy, non-const predicate");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 26>(
            call, check, "partition_copy, non-const predicate");
#endif
    }

    run_algo_all_policies<writable_archetype, writable_archetype_dc, 27>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace_if(std::forward<decltype(policy)>(policy), view, write_pred_mut{},
                                          typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 && std::ranges::begin(view)[1].val == 1 &&
                   std::ranges::begin(view)[3].val == 42;
        },
        "replace_if, non-const predicate");

    // replace and remove_copy with a projection taking the element by non-const reference: the value
    // itself is compared with std::ranges::equal_to, so the projection is the only user callable here.
    run_algo_all_policies<replaceable_archetype, replaceable_archetype_dc, 28>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace(std::forward<decltype(policy)>(policy), view, search_value{3},
                                       typename elem_t::value_arg{42}, replace_proj_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[3].val == 42 && std::ranges::begin(view)[2].val == 2;
        },
        "replace, non-const projection");

    // remove_copy is guarded here for the very same reason as above: the const copy of the input element
    // breaks the device call before the projection is ever reached.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3},
                                           replace_proj_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[3].val == 4 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - 1;
        };

        run_algo2_host_policies<remove_copy_in_archetype, copy_out_archetype>(call, check,
                                                                               "remove_copy, non-const projection");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_HETERO
        run_algo2_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 29>(
            call, check, "remove_copy, non-const projection");
#endif
    }

    // The two replacing copies are guarded here for the very same reason as above: the copy of the new
    // value breaks the call before the predicate, respectively the projection, is ever reached.
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view,
                                               copy_pred_mut{}, typename out_t::value_arg{42});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[0].val == 42 && std::ranges::begin(out_view)[3].val == 42 &&
                   std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST
        run_algo2_offset_host_policies<copy_in_archetype, copy_out_archetype>(
            call, check, "replace_copy_if, non-const predicate");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 30>(
            call, check, "replace_copy_if, non-const predicate");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3},
                                            typename out_t::value_arg{42}, replace_proj_mut{});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[3].val == 42 && std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST
        run_algo2_offset_host_policies<remove_copy_in_archetype, copy_out_archetype>(
            call, check, "replace_copy, non-const projection");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 31>(
            call, check, "replace_copy, non-const projection");
#endif
    }

    // The writing pattern over plain_archetype_view, i.e. over ranges without the members
    // std::ranges::view_interface provides; see the plain range section of the read test for what this
    // proves. copy is the representative shape here: the number of elements to write is the smaller of
    // the two range sizes, which the implementation has to obtain through std::ranges::size and not
    // through a size() member of the user range. Both storages start as 0, 1, 2, ..., so only the two
    // returned iterators say something here; that the assignment happens at all is what the call at id 1
    // above checks.
    run_algo2_plain_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc,
                                 32>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto res) {
            return res.in == std::ranges::begin(in_view) + std::ranges::size(in_view) &&
                   res.out == std::ranges::begin(out_view) + std::ranges::size(out_view);
        },
        "copy, plain ranges");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
