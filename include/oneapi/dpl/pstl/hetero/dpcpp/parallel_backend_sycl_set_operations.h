// -*- C++ -*-
//===-- parallel_backend_sycl_set_operations.h ----------------------------===//
//
// Copyright (C) Intel Corporation
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_SET_OPERATIONS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_SET_OPERATIONS_H

#include <algorithm>   // std::min/max
#include <cstddef>     // std::size_t
#include <cmath>       // std::abs
#include <functional>  // std::invoke
#include <tuple>
#include <type_traits>
#include <utility>

#include "parallel_backend_sycl_utils.h"
#include "utils_ranges_sycl.h"
#include "unseq_backend_sycl.h" // set operation tags
#include "parallel_backend_sycl_merge.h" // __find_start_point
#include "parallel_backend_sycl_reduce_then_scan_tools.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// Describe final and OOB positions in source ranges together for bounded set operations
// (where tracking of OOB position is needed to determine the effective final position
// in source ranges based on output range size).
template <typename _Range1, typename _Range2>
struct _SetOpFinalAndOOBPosTypeImpl
{
    using _Size1 = oneapi::dpl::__internal::__difference_t<_Range1>;
    using _Size2 = oneapi::dpl::__internal::__difference_t<_Range2>;
    using _PositionT = oneapi::dpl::__internal::tuple<_Size1, _Size2>;
    using __final_pos_t = _PositionT;

    _PositionT __final_pos = {}; // Describes final position after set operation in source ranges
    _PositionT __oob_pos = {};   // Describes OOB position during set operation in source ranges

    // Compute the stop (end) position of the set operation result in the source ranges.
    // The effective stop position is the smaller of:
    //   - the final position produced by the set operation (without output range size limitations), and
    //   - the out-of-bounds (OOB) position marking the end of the available output range.
    std::tuple<_Size1, _Size2>
    __compute_stop_pos() const
    {
        return {std::min(std::get<0>(__final_pos), std::get<0>(__oob_pos)),
                std::min(std::get<1>(__final_pos), std::get<1>(__oob_pos))};
    }

    template <typename _SetTag>
    static _SetOpFinalAndOOBPosTypeImpl
    __create_initial_state(_SetTag __set_tag, const _Range1& __range1, const _Range2& __range2)
    {
        const _Size1 __n1 = oneapi::dpl::__ranges::__size(__range1);
        const _Size2 __n2 = oneapi::dpl::__ranges::__size(__range2);

        _PositionT __initial_final_pos = {__n1, __n2}; // set_union and set_symmetric_difference stop at the ends
        if constexpr (std::is_same_v<_SetTag, unseq_backend::_DifferenceTag>)
            __initial_final_pos = {__n1, 0}; // for set_difference, position in the second range is obtained later
        else if constexpr (std::is_same_v<_SetTag, unseq_backend::_IntersectionTag>)
            __initial_final_pos = {0, 0}; // for set_intersection, both positions are obtained later

        // Initial OOB state initialized by the size of the ranges because we will use std::min() for their state
        // and final positions on host side after finish Kernel's code.
        _PositionT __initial_oob_pos{__n1, __n2};

        return _SetOpFinalAndOOBPosTypeImpl{__initial_final_pos, __initial_oob_pos};
    }
};

template <typename _Range1, typename _Range2>
using _SetOpFinalAndOOBPosType = _SetOpFinalAndOOBPosTypeImpl<std::decay_t<_Range1>, std::decay_t<_Range2>>;

// The return type of set operation implementation for bounded and unbounded cases.
// For bounded case we need to return final and OOB positions in source ranges,
// for unbounded case we need to return only the size of the result.
template <bool _Bounded, typename _Range1, typename _Range2, typename _Range3>
using __set_op_impl_return_t =
    std::conditional_t<_Bounded,
                       oneapi::dpl::__internal::tuple<oneapi::dpl::__internal::__difference_t<_Range1>,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>,
                       oneapi::dpl::__internal::__difference_t<_Range3>>;

// Create the result of set operation implementation for bounded and unbounded cases
// based on the final and OOB positions in source ranges or the size of the result respectively.
template <bool _Bounded, typename _Range1, typename _Range2, typename _Range3>
__set_op_impl_return_t<_Bounded, _Range1, _Range2, _Range3>
__create_set_op_impl_result(oneapi::dpl::__internal::__difference_t<_Range1> __idx1,
                            oneapi::dpl::__internal::__difference_t<_Range2> __idx2,
                            oneapi::dpl::__internal::__difference_t<_Range3> __idx3)
{
    if constexpr (_Bounded)
        return {__idx1, __idx2, __idx3};
    else
        return __idx3;
}

template <bool _Bounded, typename _SetTag, typename _Range1, typename _Range2>
std::conditional_t<_Bounded, _SetOpFinalAndOOBPosType<_Range1, _Range2>, std::size_t>
__create_initial_final_and_oob_pos_state(_SetTag __set_tag, const _Range1& __range1, const _Range2& __range2)
{
    if constexpr (_Bounded)
        return _SetOpFinalAndOOBPosType<_Range1, _Range2>::__create_initial_state(__set_tag, __range1, __range2);
    else
        return std::size_t{0};
}

// Writes multiple elements from temp data to the output range. The values to write are stored in `__temp_data` from a
// previous operation, and must be written to the output range in the appropriate location. The zeroth element of `__v`
// will contain the index of one past the last element to write, and the first element of `__v` will contain the number
// of elements to write. Used for __parallel_set_write_a_b_op.
struct __write_multiple_to_id
{
    // The OOB position reported by this write operation is an offset within the temporary data of a diagonal.
    using __position_type = __temp_data_array_idx_t;

    template <typename _ValueType>
    friend _ValueType
    __transform_result(const __write_multiple_to_id& __write_op,
                       const oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __carry)
    {
        return std::min<_ValueType>(__carry.__get_cref(), __write_op.__out_size);
    }

    template <typename _ValueType>
    bool
    __oob_write_possible(std::size_t __max_write_offset, std::size_t /*__start_idx*/,
                         const oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __prefix) const
    {
        const std::size_t __carry_in = __prefix.__has_value() ? __prefix.__get_cref() : 0;
        return (__carry_in + __max_write_offset > __out_size);
    }

    template <typename _OutRng, typename _SizeType, typename _ValueType, typename _TempData>
    void
    operator()(_OutRng& __out_rng, const _SizeType, const _ValueType& __v, _TempData& __temp_data) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__temp_data.get_and_destroy(0))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        const __temp_data_array_idx_t __n = std::get<1>(__v);
        for (__temp_data_array_idx_t __i = 0; __i < __n; ++__i)
        {
            __assign(static_cast<_ConvertedTupleType>(__temp_data.get_and_destroy(__i)),
                     __out_rng[std::get<0>(__v) - std::get<1>(__v) + __i]);
        }
    }

    template <typename _OutRng, typename _SizeType, typename _ValueType, typename _TempData, typename _OnOOBReached>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v, _TempData& __temp_data,
               _OnOOBReached __on_oob_reached) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__temp_data.get_and_destroy(0))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        const __temp_data_array_idx_t __n = std::get<1>(__v);
        for (__temp_data_array_idx_t __i = 0; __i < __n; ++__i)
        {
            // Retrieve and destroy the temporary slot unconditionally, before the bounds check. Each slot was
            // constructed by the generator via placement-new (__temp_data_array::set) and requires an explicit
            // matching destruction. Since __temp_data is reused for the next scanned element, destroying only on the
            // in-bounds path would leak slots whose output index falls outside __out_size (the bounded/overflow case)
            // and let the next set() placement-new over a still-live object.
            auto&& __val = __temp_data.get_and_destroy(__i);

            const std::size_t __out_idx = std::get<0>(__v) - std::get<1>(__v) + __i;
            if (__out_idx < __out_size)
                __assign(static_cast<_ConvertedTupleType>(std::forward<decltype(__val)>(__val)), __out_rng[__out_idx]);

            // Report the source id of the current diagonal together with the local element offset within the
            // temporary data. This is enough to recover the source index pair later (by re-running the serial
            // generator for this diagonal) without re-running any sub-group collective operation.
            if (__out_idx == __out_size)
                __on_oob_reached(__id, __i);
        }
    }
    const std::size_t __out_size;
    oneapi::dpl::__internal::__pstl_assign __assign;
};

// Set operation generic implementation, used for serial set operation of intersection, difference, union, and
// symmetric difference.
template <typename _SetTag>
struct __set_operation
{
    template <typename _InRng1, typename _InRng2, typename _TempOutput, typename _Compare, typename _Proj1,
              typename _Proj2, typename _FinalPosSaver>
    __temp_data_array_idx_t
    operator()(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t __idx1, std::size_t __idx2,
               const __temp_data_array_idx_t __num_eles_min, _TempOutput& __temp_out, const _Compare __comp,
               _Proj1 __proj1, _Proj2 __proj2, _FinalPosSaver __final_pos_saver) const
    {
        constexpr bool __is_set_difference = std::is_same_v<_SetTag, unseq_backend::_DifferenceTag>;
        constexpr bool __is_set_intersection = std::is_same_v<_SetTag, unseq_backend::_IntersectionTag>;
        constexpr bool __is_set_union = std::is_same_v<_SetTag, unseq_backend::_UnionTag>;
        constexpr bool __is_set_symmetric_difference = std::is_same_v<_SetTag, unseq_backend::_SymmetricDifferenceTag>;

        using _ValueTypeRng1 = typename oneapi::dpl::__internal::__value_t<_InRng1>;
        using _ValueTypeRng2 = typename oneapi::dpl::__internal::__value_t<_InRng2>;

        const auto __size1 = oneapi::dpl::__ranges::__size(__in_rng1);
        const auto __size2 = oneapi::dpl::__ranges::__size(__in_rng2);

        auto __write_temp_element = [&](const __temp_data_array_idx_t __count_arg, const auto& __value,
                                        std::size_t __idx1, std::size_t __idx2) {
            if constexpr (_TempOutput::__capture_indexes_flag)
                __temp_out.set(__count_arg, __value, {__idx1, __idx2});
            else
                __temp_out.set(__count_arg, __value);
        };

        // Make sense to calculate final positions only for set_intersection and set_difference operations,
        // and only if the user provided a callback to save the final positions.
        // Stop positions for set_union and set_symmetric_difference are not needed, because they are known in advance.
        constexpr bool __need_call_final_pos_saver =
            !std::is_same_v<std::decay_t<_FinalPosSaver>, __internal::__no_callback_tag> &&
            (__is_set_intersection || __is_set_difference);

        // Merge-path indices captured at iteration entry. The global merge path is partitioned across
        // work-items and walked exactly once, so the single step that first reaches the operation's natural
        // termination boundary occurs in exactly one work-item at exactly one iteration. Comparing the entry
        // indices against the post-step indices detects precisely that transition (the edge crossing), which
        // lets a single work-item write the final source position directly, with no reduction and no atomics.
        // These are refreshed at the top of every bounds-checked iteration below.
        [[maybe_unused]] std::size_t __idx1_at_entry = __idx1;
        [[maybe_unused]] std::size_t __idx2_at_entry = __idx2;

        auto __process_final_pos = [&](std::size_t __idx1, std::size_t __idx2) {
            if constexpr (__need_call_final_pos_saver)
            {
                // For set_intersection the operation terminates once either input is exhausted: the edge crossing
                // is the step that moves from "strictly inside both inputs" to "at least one input exhausted".
                if constexpr (__is_set_intersection)
                {
                    if (__idx1_at_entry < __size1 && __idx2_at_entry < __size2 &&
                        (__idx1 == __size1 || __idx2 == __size2))
                        __final_pos_saver({__idx1, __idx2});
                }
                // For set_difference the operation terminates once the first input (set A) is exhausted: the edge
                // crossing is the step that moves idx1 to the end of set A. This also covers the tail-drain of set
                // A performed after set B is exhausted.
                else if constexpr (__is_set_difference)
                {
                    if (__idx1_at_entry < __size1 && __idx1 == __size1)
                        __final_pos_saver({__idx1, __idx2});
                }
            }
        };

        __temp_data_array_idx_t __count = 0;
        __temp_data_array_idx_t __idx = 0;

        constexpr bool _CopyMatch = __is_set_intersection || __is_set_union;
        constexpr bool _CopyDiffSetA = !__is_set_intersection;
        constexpr bool _CopyDiffSetB = __is_set_union || __is_set_symmetric_difference;

        // Bounds checks are enabled only when this diagonal can reach the end of either range.
        const bool __check_bounds = (__idx1 + __num_eles_min >= __size1) || (__idx2 + __num_eles_min >= __size2);
        while (__idx < __num_eles_min)
        {
            if (__check_bounds)
            {
                if constexpr (__need_call_final_pos_saver)
                {
                    __idx1_at_entry = __idx1;
                    __idx2_at_entry = __idx2;
                }

                if (__idx1 == __size1)
                {
                    if constexpr (_CopyDiffSetB)
                    {
                        // If we are at the end of rng1, copy the rest of rng2 within our diagonal's bounds
                        for (; __idx2 < __size2 && __idx < __num_eles_min; ++__idx2, ++__idx)
                        {
                            __write_temp_element(__count, __in_rng2[__idx2], __idx1, __idx2);
                            ++__count;
                        }
                    }
                    __idx = __num_eles_min;
                    if constexpr (__need_call_final_pos_saver)
                        __process_final_pos(__idx1, __idx2);
                    continue;
                }

                if (__idx2 == __size2)
                {
                    if constexpr (_CopyDiffSetA)
                    {
                        // If we are at the end of rng2, copy the rest of rng1 within our diagonal's bounds
                        for (; __idx1 < __size1 && __idx < __num_eles_min; ++__idx1, ++__idx)
                        {
                            __write_temp_element(__count, __in_rng1[__idx1], __idx1, __idx2);
                            ++__count;
                        }
                    }
                    __idx = __num_eles_min;
                    if constexpr (__need_call_final_pos_saver)
                        __process_final_pos(__idx1, __idx2);
                    continue;
                }
            }

            const _ValueTypeRng1& __ele_rng1 = __in_rng1[__idx1];
            const _ValueTypeRng2& __ele_rng2 = __in_rng2[__idx2];
            if (std::invoke(__comp, std::invoke(__proj1, __ele_rng1), std::invoke(__proj2, __ele_rng2)))
            {
                if constexpr (_CopyDiffSetA)
                {
                    __write_temp_element(__count, __ele_rng1, __idx1, __idx2);
                    ++__count;
                }
                ++__idx1;
                ++__idx;
            }
            else if (std::invoke(__comp, std::invoke(__proj2, __ele_rng2), std::invoke(__proj1, __ele_rng1)))
            {
                if constexpr (_CopyDiffSetB)
                {
                    __write_temp_element(__count, __ele_rng2, __idx1, __idx2);
                    ++__count;
                }
                ++__idx2;
                ++__idx;
            }
            else // if neither element is less than the other, they are equal
            {
                if constexpr (_CopyMatch)
                {
                    __write_temp_element(__count, __ele_rng1, __idx1, __idx2);
                    ++__count;
                }
                ++__idx1;
                ++__idx2;
                __idx += 2;
            }
            // The edge crossing can only occur on a diagonal that can reach a range end, i.e. when bounds are
            // checked. Keep the interior hot path free of the extra comparisons.
            if constexpr (__need_call_final_pos_saver)
            {
                if (__check_bounds)
                    __process_final_pos(__idx1, __idx2);
            }
        }

        return __count;
    }
};

template <bool __return_star, typename _Rng, typename _IdxT>
auto
__decode_balanced_path_temp_data_impl(const _Rng& __rng, const _IdxT __id,
                                      const __temp_data_array_idx_t __diagonal_spacing)
{
    using SizeT = decltype(oneapi::dpl::__ranges::__size(__rng));
    using SignedSizeT = std::make_signed_t<decltype(oneapi::dpl::__ranges::__size(__rng))>;
    const SignedSizeT __tmp = __rng[__id];
    const SizeT __star_offset = oneapi::dpl::__internal::__dpl_signbit(__tmp) ? 1 : 0;
    const SizeT __rng1_idx = std::abs(__tmp);
    const SizeT __rng2_idx = __id * __diagonal_spacing - __rng1_idx + __star_offset;
    if constexpr (__return_star)
    {
        return std::make_tuple(__rng1_idx, __rng2_idx, __star_offset);
    }
    else
    {
        return std::make_tuple(__rng1_idx, __rng2_idx);
    }
}

template <typename _Rng, typename _IdxT>
std::tuple<_IdxT, _IdxT>
__decode_balanced_path_temp_data_no_star(const _Rng& __rng, const _IdxT __id,
                                         const __temp_data_array_idx_t __diagonal_spacing)
{
    return __decode_balanced_path_temp_data_impl<false>(__rng, __id, __diagonal_spacing);
}

template <typename _Rng, typename _IdxT>
std::tuple<_IdxT, _IdxT, decltype(oneapi::dpl::__ranges::__size(std::declval<_Rng>()))>
__decode_balanced_path_temp_data(const _Rng& __rng, const _IdxT __id, const __temp_data_array_idx_t __diagonal_spacing)
{
    return __decode_balanced_path_temp_data_impl<true>(__rng, __id, __diagonal_spacing);
}

template <typename _IdxT>
std::make_signed_t<_IdxT>
__encode_balanced_path_temp_data(const _IdxT __rng1_idx, const bool __star)
{
    using signed_t = std::make_signed_t<_IdxT>;

    // Convert to signed representation - we know this is positive and can be represented in the signed portion
    signed_t __signed_idx{static_cast<signed_t>(__rng1_idx)};

    // Branchless negation: (1 - 2 * __star) gives 1 if __star is false, -1 if __star is true
    return __signed_idx * (signed_t{1} - signed_t{2} * signed_t{__star});
}

struct __get_bounds_partitioned
{
    template <typename _Rng, typename _IndexT>
    auto // Returns a tuple of the form (start1, end1, start2, end2)
    operator()(const _Rng& __in_rng, const _IndexT __id) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        auto __rng_tmp_diag = std::get<2>(__tuple); // set a temp storage sequence

        using _SizeType = std::common_type_t<
            std::make_unsigned_t<decltype(oneapi::dpl::__ranges::__size(std::get<0>(__in_rng.base())))>,
            std::make_unsigned_t<decltype(oneapi::dpl::__ranges::__size(std::get<1>(__in_rng.base())))>,
            std::make_unsigned_t<decltype(oneapi::dpl::__ranges::__size(__rng_tmp_diag))>>;

        // Establish bounds of ranges for the tile from sparse partitioning pass kernel

        // diagonal index of the tile begin
        const _SizeType __wg_begin_idx = (__id / __tile_size) * __tile_size;
        const _SizeType __signed_tile_size = static_cast<_SizeType>(__tile_size);
        const _SizeType __wg_end_idx = std::min<_SizeType>(((__id / __signed_tile_size) + 1) * __signed_tile_size,
                                                           oneapi::dpl::__ranges::__size(__rng_tmp_diag) - 1);

        const auto [begin_rng1, begin_rng2] =
            __decode_balanced_path_temp_data_no_star(__rng_tmp_diag, __wg_begin_idx, __diagonal_spacing);
        const auto [end_rng1, end_rng2] =
            __decode_balanced_path_temp_data_no_star(__rng_tmp_diag, __wg_end_idx, __diagonal_spacing);
        return std::make_tuple(_SizeType{begin_rng1}, _SizeType{end_rng1}, _SizeType{begin_rng2}, _SizeType{end_rng2});
    }
    __temp_data_array_idx_t __diagonal_spacing;
    std::size_t __tile_size;
    std::size_t __partition_threshold;
};

struct __get_bounds_simple
{
    template <typename _Rng, typename _IndexT>
    auto // Returns a tuple of the form (start1, end1, start2, end2)
    operator()(const _Rng& __in_rng, const _IndexT) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        const auto __rng1 = std::get<0>(__tuple); // first sequence
        const auto __rng2 = std::get<1>(__tuple); // second sequence

        using _SizeType = oneapi::dpl::__ranges::__common_size_t<decltype(__rng1), decltype(__rng2)>;

        return std::make_tuple(_SizeType{0}, static_cast<_SizeType>(oneapi::dpl::__ranges::__size(__rng1)),
                               _SizeType{0}, static_cast<_SizeType>(oneapi::dpl::__ranges::__size(__rng2)));
    }
};

// Reduce then scan building block for set balanced path which is used in the reduction kernel to calculate the
// balanced path intersection, store it to temporary data with "star" status, then count the number of elements to write
// to the output for the reduction operation.
// _RetType is the scanned type for the scan so it must be the size type of the output range.
template <typename _SetOpCount, typename _BoundsProvider, typename _RetType, typename _Compare, typename _Proj1,
          typename _Proj2>
struct __gen_set_balanced_path
{
    // Locates and returns the "intersection" of a diagonal on the balanced path, based on merge path coordinates.
    // It returns coordinates in each set of the intersection with a boolean representing if the diagonal is "starred",
    // meaning that the balanced path "intersection" point does not lie directly on the diagonal, but one step forward in
    // the second set.
    // Some diagonals must be "starred" to ensure that matching elements between rng1 and rng2 are processed in pairs
    // starting from the first of repeating value(s) in each range and a matched pair are not split between work-items.
    template <typename _Rng1, typename _Rng2, typename _Index>
    std::tuple<_Index, _Index, bool>
    __find_balanced_path_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __merge_path_rng1,
                                     const _Index __merge_path_rng2, const _Index __rng1_begin,
                                     const _Index __rng2_begin, const _Index __rng2_end) const
    {
        // back up to balanced path divergence with a biased binary search
        bool __star = false;
        if (__merge_path_rng1 == 0 || __merge_path_rng2 == oneapi::dpl::__ranges::__size(__rng2))
        {
            return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
        }

        if (std::invoke(__comp, std::invoke(__proj1, __rng1[__merge_path_rng1 - 1]),
                        std::invoke(__proj2, __rng2[__merge_path_rng2])))
        {
            // There is no chance that the balanced path differs from the merge path here, because the previous element of
            // rng1 does not match the next element of rng2. We can just return the merge path.
            return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
        }

        // find first element of repeating sequence in the first set of the previous element
        _Index __rng1_repeat_start = oneapi::dpl::__internal::__biased_lower_bound_idx</*__last_bias=*/true>(
            __rng1, __rng1_begin, __merge_path_rng1, __rng1, __merge_path_rng1 - 1, __comp, __proj1, __proj1);
        // find first element of repeating sequence in the second set of the next element
        _Index __rng2_repeat_start = oneapi::dpl::__internal::__biased_lower_bound_idx</*__last_bias=*/true>(
            __rng2, __rng2_begin, __merge_path_rng2, __rng1, __merge_path_rng1 - 1, __comp, __proj2, __proj1);

        _Index __rng1_repeats = __merge_path_rng1 - __rng1_repeat_start;
        _Index __rng2_repeats_bck = __merge_path_rng2 - __rng2_repeat_start;

        if (__rng2_repeats_bck >= __rng1_repeats)
        {
            // If we have at least as many repeated elements in rng2, we end up back on merge path
            return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
        }

        // Calculate the number of "unmatched" repeats in the first set, dividing by two and rounding up to account for a
        // possible star diagonal.
        _Index __fwd_search_count = oneapi::dpl::__internal::__dpl_ceiling_div(__rng1_repeats - __rng2_repeats_bck, 2);

        // Calculate the max location to search in the second set for future repeats, limiting to the edge of the range
        _Index __fwd_search_bound = std::min(__merge_path_rng2 + __fwd_search_count, __rng2_end);

        _Index __balanced_path_intersection_rng2 = oneapi::dpl::__internal::__pstl_upper_bound_idx(
            __rng2, __merge_path_rng2, __fwd_search_bound, __rng1, __merge_path_rng1 - 1, __comp, __proj2, __proj1);

        // Calculate the number of matchable "future" repeats in the second set
        _Index __matchable_forward_ele_rng2 = __balanced_path_intersection_rng2 - __merge_path_rng2;
        _Index __total_matched_rng2 = __balanced_path_intersection_rng2 - __rng2_repeat_start;

        // Update balanced path intersection for rng1, must account for cases where there are more repeating elements in
        // rng1 than matched elements of rng2
        _Index __balanced_path_intersection_rng1 =
            __rng1_repeat_start + std::max(__total_matched_rng2, __rng1_repeats - __matchable_forward_ele_rng2);

        // If we needed to step off the diagonal to find the balanced path, mark the diagonal as "starred"
        __star = __balanced_path_intersection_rng1 + __balanced_path_intersection_rng2 !=
                 __merge_path_rng1 + __merge_path_rng2;

        return std::make_tuple(__balanced_path_intersection_rng1, __balanced_path_intersection_rng2, __star);
    }

    template <typename _InRng, typename _IndexT, typename _BoundsProviderLocal>
    std::tuple<_IndexT, _IndexT, bool>
    calc_and_store_balanced_path(_InRng& __in_rng, _IndexT __id, _BoundsProviderLocal __get_bounds_local) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        const auto __rng1 = std::get<0>(__tuple); // first sequence
        const auto __rng2 = std::get<1>(__tuple); // second sequence

        auto __rng1_temp_diag = std::get<2>(__tuple); // set a temp storage sequence

        using _SizeType = oneapi::dpl::__ranges::__common_size_t<decltype(__rng1), decltype(__rng2)>;
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2))
        {
            // ensure we do not go out of bounds
            __i_elem = oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2) - 1;
        }
        auto [__rng1_lower, __rng1_upper, __rng2_lower, __rng2_upper] = __get_bounds_local(__in_rng, __id);
        //find merge path intersection
        auto [__rng1_pos, __rng2_pos] = oneapi::dpl::__par_backend_hetero::__find_start_point(
            __rng1, __rng1_lower, __rng1_upper, __rng2, __rng2_lower, __rng2_upper, __i_elem, __comp, __proj1, __proj2);

        //Find balanced path for diagonal start
        auto [__rng1_balanced_pos, __rng2_balanced_pos, __star] = __find_balanced_path_start_point(
            __rng1, __rng2, __rng1_pos, __rng2_pos, __rng1_lower, __rng2_lower, __rng2_upper);

        // Use sign bit to represent star offset. Temp storage is a signed type equal to the difference_type of the
        // input iterator range. The index will fit into the positive portion of the type, so the sign may be used to
        // indicate the star offset.
        __rng1_temp_diag[__id] =
            oneapi::dpl::__par_backend_hetero::__encode_balanced_path_temp_data(__rng1_balanced_pos, __star);

        return std::make_tuple(__rng1_balanced_pos, __rng2_balanced_pos, __star);
    }

    //Entry point for partitioning phase
    template <typename _InRng, typename _IndexT>
    void
    __calc_partition_bounds(const _InRng& __in_rng, _IndexT __id) const
    {
        calc_and_store_balanced_path(__in_rng, __id, oneapi::dpl::__par_backend_hetero::__get_bounds_simple{});
    }

    // Entry point for reduce then scan reduce input
    template <typename _InRng, typename _IndexT>
    _RetType
    operator()(const _InRng& __in_rng, _IndexT __id) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        const auto __rng1 = std::get<0>(__tuple);   // first sequence
        const auto __rng2 = std::get<1>(__tuple);   // second sequence
        auto __rng_tmp_diag = std::get<2>(__tuple); // temp diag sequence

        _IndexT __rng1_balanced_pos = 0;
        _IndexT __rng2_balanced_pos = 0;
        bool __star = false;

        const auto __total_size = oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2);
        const bool __is_partitioned = __total_size >= __get_bounds.__partition_threshold;

        if (__id * __diagonal_spacing >= __total_size)
            return _RetType{0};
        if (!__is_partitioned)
        {
            // If not partitioned, just use the bounds of the full range to limit balanced path intersection search
            auto [__idx_rng1, __idx_rng2, __local_star] =
                calc_and_store_balanced_path(__in_rng, __id, oneapi::dpl::__par_backend_hetero::__get_bounds_simple{});
            __rng1_balanced_pos = __idx_rng1;
            __rng2_balanced_pos = __idx_rng2;
            __star = __local_star;
        }
        else if (__id % __get_bounds.__tile_size != 0)
        {
            // If partitioned, but not on the boundary, we must calculate intersection with the balanced path, and
            // we can use bounds for our search established in the partitioning phase by __get_bounds.
            auto [__idx_rng1, __idx_rng2, __local_star] = calc_and_store_balanced_path(__in_rng, __id, __get_bounds);
            __rng1_balanced_pos = __idx_rng1;
            __rng2_balanced_pos = __idx_rng2;
            __star = __local_star;
        }
        else // if we are at the start of a tile, we can decode the balanced path from the existing temporary data
        {
            auto [__idx_rng1, __idx_rng2, __local_star] =
                __decode_balanced_path_temp_data(__rng_tmp_diag, __id, __diagonal_spacing);
            __rng1_balanced_pos = __idx_rng1;
            __rng2_balanced_pos = __idx_rng2;
            __star = __local_star;
        }

        __noop_temp_data __temp_data{}; // lvalue is required to call __set_op_count
        const __temp_data_array_idx_t __eles_to_process = static_cast<__temp_data_array_idx_t>(
            std::min(_IndexT{__diagonal_spacing} - (__star ? _IndexT{1} : _IndexT{0}),
                     oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2) -
                         _IndexT{__id * __diagonal_spacing - 1}));

        return _RetType{__set_op_count(__rng1, __rng2, __rng1_balanced_pos, __rng2_balanced_pos, __eles_to_process,
                                       __temp_data, __comp, __proj1, __proj2, __internal::__no_callback_tag{})};
    }
    _SetOpCount __set_op_count;
    __temp_data_array_idx_t __diagonal_spacing;
    _BoundsProvider __get_bounds;
    _Compare __comp;
    _Proj1 __proj1;
    _Proj2 __proj2;
};

// Reduce then scan building block for set balanced path which is used in the scan kernel to decode the stored balanced
// path intersection, perform the serial set operation for the diagonal, counting the number of elements and writing
// the output to temporary data in registers to be ready for the scan and write operations to follow.
// _RetType is the scanned type for the scan so it must be the size type of the output range.
template <typename _SetOpCount, typename _TempData, typename _RetType, typename _Compare, typename _Proj1,
          typename _Proj2>
struct __gen_set_op_from_known_balanced_path
{
    template <typename _InRng>
    using __result_t = std::tuple<_RetType, __temp_data_array_idx_t>;

    using TempData = _TempData;
    template <typename _InRng, typename _IndexT, typename _TempDataArg, typename _FinalPosSaver>
    __result_t<_InRng>
    operator()(const _InRng& __in_rng, _IndexT __id, _TempDataArg& __output_data,
               _FinalPosSaver __final_pos_saver) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        const auto __rng1 = std::get<0>(__tuple); // first sequence
        const auto __rng2 = std::get<1>(__tuple); // second sequence

        // set a temp storage sequence, star value in sign bit
        const auto __rng1_temp_diag = std::get<2>(__tuple);

        using _SizeType =
            oneapi::dpl::__ranges::__common_size_t<decltype(__rng1), decltype(__rng2), decltype(__rng1_temp_diag)>;
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2))
            return __result_t<_InRng>{_RetType{0}, __temp_data_array_idx_t{0}};
        auto [__rng1_idx, __rng2_idx, __star_offset] =
            oneapi::dpl::__par_backend_hetero::__decode_balanced_path_temp_data(__rng1_temp_diag, __id,
                                                                                __diagonal_spacing);

        __temp_data_array_idx_t __eles_to_process = static_cast<__temp_data_array_idx_t>(
            std::min(static_cast<_SizeType>(__diagonal_spacing - __star_offset),
                     static_cast<_SizeType>(oneapi::dpl::__ranges::__size(__rng1) +
                                            oneapi::dpl::__ranges::__size(__rng2) - __i_elem + 1)));

        __temp_data_array_idx_t __count = __set_op_count(__rng1, __rng2, __rng1_idx, __rng2_idx, __eles_to_process,
                                                         __output_data, __comp, __proj1, __proj2, __final_pos_saver);

        return __result_t<_InRng>{_RetType{__count}, __count};
    }
    _SetOpCount __set_op_count;
    __temp_data_array_idx_t __diagonal_spacing;
    _Compare __comp;
    _Proj1 __proj1;
    _Proj2 __proj2;
};

// kernel for balanced path to partition the input into tiles by calculating balanced path on diagonals of tile bounds
template <typename _GenInput, typename _KernelName>
struct __partition_set_balanced_path_submitter;
template <typename _GenInput, typename... _KernelName>
struct __partition_set_balanced_path_submitter<_GenInput, __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _InInOutRng>
    sycl::event
    operator()(sycl::queue& __q, _InInOutRng&& __in_in_out_rng, std::size_t __num_diagonals) const
    {
        const std::size_t __tile_size = __gen_input.__get_bounds.__tile_size;
        const std::size_t __n =
            oneapi::dpl::__internal::__dpl_ceiling_div(__num_diagonals + __tile_size - 1, __tile_size);
        return __q.submit([&__in_in_out_rng, this, __n, __num_diagonals](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_in_out_rng);

            __cgh.parallel_for<_KernelName...>(
                sycl::range</*dim=*/1>(__n), [=, *this](sycl::item</*dim=*/1> __item_id) {
                    auto __global_idx = __item_id.get_linear_id();
                    const std::size_t __tile_size = __gen_input.__get_bounds.__tile_size;
                    std::size_t __id = (__global_idx * __tile_size < __num_diagonals) ? __global_idx * __tile_size
                                                                                      : __num_diagonals - 1;
                    __gen_input.__calc_partition_bounds(__in_in_out_rng, __id);
                });
        });
    }
    _GenInput __gen_input;
};

template <typename... _Name>
class __set_op_partition_kernel;

template <typename _CustomName, typename _InInOutRng, typename _GenReduceInput>
sycl::event
__parallel_set_balanced_path_partition(sycl::queue& __q, _InInOutRng&& __in_in_out_rng, std::size_t __num_diagonals,
                                       _GenReduceInput __gen_reduce_input)
{
    using _PartitionKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __set_op_partition_kernel<_CustomName>>;
    using _PartitionSubmitter = __partition_set_balanced_path_submitter<_GenReduceInput, _PartitionKernel>;

    _PartitionSubmitter __partition_submitter{__gen_reduce_input};

    return __partition_submitter(__q, std::forward<_InInOutRng>(__in_in_out_rng), __num_diagonals);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_SET_OPERATIONS_H
