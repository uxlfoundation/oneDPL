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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_SCAN_HELPERS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_SCAN_HELPERS_H
#include <cstdint>
#include <tuple>
#include <utility>
#include <type_traits>
#include <cstddef>
#include <algorithm>
#include <cmath>

#include "../../tuple_impl.h"
#include "../../utils.h"
#include "../../utils_ranges.h"
#include "unseq_backend_sycl.h"

// This file contains helper "building block" structures for use with reduce_then_scan operations in the SYCL backend.

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// *** Utilities ***

// Temporary data structure which is used to store results to registers during a reduce then scan operation.
template <std::uint16_t elements, typename _ValueT>
struct __temp_data_array
{
    template <typename _ValueT2>
    void
    set(std::uint16_t __idx, const _ValueT2& __ele)
    {
        __data[__idx].__setup(__ele);
    }

    _ValueT
    get_and_destroy(std::uint16_t __idx)
    {
        _ValueT __ele = std::move(__data[__idx].__v);
        __data[__idx].__destroy();
        return __ele;
    }

    oneapi::dpl::__internal::__lazy_ctor_storage<_ValueT> __data[elements];
};

// This is a stand-in for a temporary data structure which is used to turn set() into a no-op. This is used in the case
// where no temporary register data is needed within reduce then scan kern
struct __noop_temp_data
{
    template <typename _ValueT>
    void
    set(std::uint16_t, const _ValueT&) const
    {
    }
};

// Extracts a range from a zip iterator based on the element ID
template <std::size_t _EleId>
struct __extract_range_from_zip
{
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng) const
    {
        return std::get<_EleId>(__in_rng.tuple());
    }
};

// Extracts the zeroth element from a tuple or pair
struct __get_zeroth_element
{
    template <typename _Tp>
    auto&
    operator()(_Tp&& __a) const
    {
        return std::get<0>(std::forward<_Tp>(__a));
    }
};

// *** Write Operations ***

// Writes a single element to the output range at the specified index, `__id`. The value to write is passed in as `__v`.
// Used in __parallel_transform_scan.
struct __simple_write_to_id
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _ValueType& __v, const _TempData&) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__v)>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
        __out_rng[__id] = static_cast<_ConvertedTupleType>(__v);
    }
};

// Writes a single element `get<2>(__v)` to the output range at the index, `get<0>(__v) - 1 + __offset`, but only if the
// condition `get<0>(__v)` is `true`. Used in __parallel_copy_if, __parallel_unique_copy, and
// __parallel_set_reduce_then_scan_set_a_write
template <std::int32_t __offset, typename _Assign>
struct __write_to_id_if
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _SizeType, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v, const _TempData&) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
        if (std::get<1>(__v))
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)), __out_rng[std::get<0>(__v) - 1 + __offset]);
    }
    _Assign __assign;
};

// Writes a single element `get<2>(__v)` to the output range at the index, `get<0>(__v) - 1`, but only if the
// condition `get<1>(__v)` is `true`. Otherwise, writes the element to the output range at the index,
// `__id - get<0>(__v)`. Used for __parallel_partition_copy.
template <typename _Assign>
struct __write_to_id_if_else
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _SizeType, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v, const _TempData&) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
        if (std::get<1>(__v))
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)), std::get<0>(__out_rng[std::get<0>(__v) - 1]));
        else
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)),
                     std::get<1>(__out_rng[__id - std::get<0>(__v)]));
    }
    _Assign __assign;
};

// Writes operation for reduce_by_segment, writes first key if the id is 0. Also, if the segment end is reached, writes
// the current value and then the next key if it exists. Used for __parallel_reduce_by_segment_reduce_then_scan.
template <typename _BinaryPred>
struct __write_red_by_seg
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _Tup>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _Tup& __tup, const _TempData&) const
    {
        using std::get;
        auto __out_keys = get<0>(__out_rng.tuple());
        auto __out_values = get<1>(__out_rng.tuple());

        const auto& __next_key = get<2>(__tup);
        const auto& __current_key = get<3>(__tup);
        const auto& __current_value = get<1>(get<0>(__tup));
        const bool __is_seg_end = get<1>(__tup);
        const std::size_t __out_idx = get<0>(get<0>(__tup));

        // With the exception of the first key which is output by index 0, the first key in each segment is written
        // by the work item that outputs the previous segment's reduction value. This is because the reduce_by_segment
        // API requires that the first key in a segment is output and is important for when keys in a segment might not
        // be the same (but satisfy the predicate). The last segment does not output a key as there are no future
        // segments process.
        if (__id == 0)
            __out_keys[0] = __current_key;
        if (__is_seg_end)
        {
            __out_values[__out_idx] = __current_value;
            if (__id != __n - 1)
                __out_keys[__out_idx + 1] = __next_key;
        }
    }
    _BinaryPred __binary_pred;
    std::size_t __n;
};

// Writes multiple elements from temp data to the output range. The values to write are stored in `__temp_data` from a
// previous operation, and must be written to the output range in the appropriate location. The zeroth element of `__v`
// will contain the index of one past the last element to write, and the first element of `__v` will contain the number
// of elements to write. Used for __parallel_set_reduce_then_scan.
template <typename _Assign>
struct __write_multiple_to_id
{
    template <typename _OutRng, typename _SizeType, typename _ValueType, typename _TempData>
    void
    operator()(_OutRng& __out_rng, _SizeType, const _ValueType& __v, _TempData& __temp_data) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__temp_data.get_and_destroy(0))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        for (std::size_t __i = 0; __i < std::get<1>(__v); ++__i)
        {
            __assign(static_cast<_ConvertedTupleType>(__temp_data.get_and_destroy(__i)),
                     __out_rng[std::get<0>(__v) - std::get<1>(__v) + __i]);
        }
    }
    _Assign __assign;
};

// *** Algorithm Specific Helpers, Input Generators to Reduction and Scan Operations ***

// __parallel_transform_scan

// A generator which applies a unary operation to the input range element at an index and returns the result.
template <typename _UnaryOp>
struct __gen_transform_input
{
    using TempData = __noop_temp_data;
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        // We explicitly convert __in_rng[__id] to the value type of _InRng to properly handle the case where we
        // process zip_iterator input where the reference type is a tuple of a references. This prevents the caller
        // from modifying the input range when altering the return of this functor.
        using _ValueType = oneapi::dpl::__internal::__value_t<_InRng>;
        return __unary_op(_ValueType{__in_rng[__id]});
    }
    _UnaryOp __unary_op;
};

// Scan copy algorithms (partition_copy, copy_if, unique_copy, set_reduce_then_scan_set_a_write)

// A mask generator to filter the input range based on a predicate, returning true if satisfied at an index.
template <typename _Predicate, typename _RangeTransform = oneapi::dpl::__internal::__no_op>
struct __gen_mask
{
    template <typename _InRng>
    bool
    operator()(_InRng&& __in_rng, std::size_t __id) const
    {
        return __pred((__rng_transform(std::forward<_InRng>(__in_rng)))[__id]);
    }
    _Predicate __pred;
    _RangeTransform __rng_transform;
};

// Wrapper for a mask generator, converting the mask generator to a counting operation.
template <typename _GenMask>
struct __gen_count_mask
{
    using TempData = __noop_temp_data;
    template <typename _InRng, typename _SizeType>
    _SizeType
    operator()(_InRng&& __in_rng, _SizeType __id, TempData&) const
    {
        return __gen_mask(std::forward<_InRng>(__in_rng), __id) ? _SizeType{1} : _SizeType{0};
    }
    _GenMask __gen_mask;
};

// A generator which expands the mask generator to return a tuple containing the count, mask, and the element at the
// specified index.
template <typename _GenMask, typename _RangeTransform = oneapi::dpl::__internal::__no_op>
struct __gen_expand_count_mask
{
    using TempData = __noop_temp_data;
    template <typename _InRng, typename _SizeType>
    auto
    operator()(_InRng&& __in_rng, _SizeType __id, TempData&) const
    {
        auto __transformed_input = __rng_transform(__in_rng);
        // Explicitly creating this element type is necessary to avoid modifying the input data when _InRng is a
        //  zip_iterator which will return a tuple of references when dereferenced. With this explicit type, we copy
        //  the values of zipped input types rather than their references.
        using _ElementType = oneapi::dpl::__internal::__value_t<decltype(__transformed_input)>;
        _ElementType ele = __transformed_input[__id];
        bool mask = __gen_mask(std::forward<_InRng>(__in_rng), __id);
        return std::tuple(mask ? _SizeType{1} : _SizeType{0}, mask, ele);
    }
    _GenMask __gen_mask;
    _RangeTransform __rng_transform;
};

// __parallel_unique_copy

// A mask generator to filter the input range based on if the element at an index is unique compared to the previous.
// Used in .
template <typename _BinaryPredicate>
struct __gen_unique_mask
{
    template <typename _InRng>
    bool
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // Starting index is offset to 1 for "unique" patterns and 0th element
        // copy is handled separately, which allows us to do this without
        // branching each access to protect underflow
        return !__pred(__in_rng[__id], __in_rng[__id - 1]);
    }
    _BinaryPredicate __pred;
};

//__parallel_set_reduce_then_scan_set_a_write

// A mask generator for set operations (difference or intersection) to determine if an element from Set A should be
// written to the output sequence based on its presence in Set B and the operation type (difference or intersection).
template <typename _IsOpDifference, typename _Compare>
struct __gen_set_mask
{
    template <typename _InRng>
    bool
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __set_a = std::get<0>(__in_rng.tuple());    // first sequence
        auto __set_b = std::get<1>(__in_rng.tuple());    // second sequence
        auto __set_mask = std::get<2>(__in_rng.tuple()); // mask sequence

        std::size_t __nb = __set_b.size();

        auto __val_a = __set_a[__id];

        auto __res = oneapi::dpl::__internal::__pstl_lower_bound(__set_b, std::size_t{0}, __nb, __val_a, __comp);

        bool bres =
            _IsOpDifference::value; //initialization is true in case of difference operation; false - intersection.
        if (__res == __nb || __comp(__val_a, __set_b[__res]))
        {
            // there is no __val_a in __set_b, so __set_b in the difference {__set_a}/{__set_b};
        }
        else
        {
            auto __val_b = __set_b[__res];

            //Difference operation logic: if number of duplication in __set_a on left side from __id > total number of
            //duplication in __set_b then a mask is 1

            //Intersection operation logic: if number of duplication in __set_a on left side from __id <= total number of
            //duplication in __set_b then a mask is 1

            const std::size_t __count_a_left =
                __id - oneapi::dpl::__internal::__pstl_left_bound(__set_a, std::size_t{0}, __id, __val_a, __comp) + 1;

            const std::size_t __count_b =
                oneapi::dpl::__internal::__pstl_right_bound(__set_b, __res, __nb, __val_b, __comp) -
                oneapi::dpl::__internal::__pstl_left_bound(__set_b, std::size_t{0}, __res, __val_b, __comp);

            if constexpr (_IsOpDifference::value)
                bres = __count_a_left > __count_b; /*difference*/
            else
                bres = __count_a_left <= __count_b; /*intersection*/
        }
        __set_mask[__id] = bres;
        return bres;
    }
    _Compare __comp;
};

// __parallel_set_reduce_then_scan

// Returns by reference: iterations consumed, and the number of elements copied to temp output.
template <bool _CopyMatch, bool _CopyDiffSetA, bool _CopyDiffSetB, bool _CheckBounds, typename _InRng1,
          typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare>
void
__set_generic_operation_iteration(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t& __idx1,
                                  std::size_t& __idx2, _SizeType __num_eles_min, _TempOutput& __temp_out,
                                  _SizeType& __idx, std::uint16_t& __count, _Compare __comp)
{
    using _ValueTypeRng1 = typename oneapi::dpl::__internal::__value_t<_InRng1>;
    using _ValueTypeRng2 = typename oneapi::dpl::__internal::__value_t<_InRng2>;

    if constexpr (_CheckBounds)
    {
        if (__idx1 == __in_rng1.size())
        {
            if constexpr (_CopyDiffSetB)
            {
                // If we are at the end of rng1, copy the rest of rng2 within our diagonal's bounds
                for (; __idx2 < __in_rng2.size() && __idx < __num_eles_min; ++__idx2, ++__idx)
                {
                    __temp_out.set(__count, __in_rng2[__idx2]);
                    ++__count;
                }
            }
            __idx = __num_eles_min;
            return;
        }
        if (__idx2 == __in_rng2.size())
        {
            if constexpr (_CopyDiffSetA)
            {
                // If we are at the end of rng2, copy the rest of rng1 within our diagonal's bounds
                for (; __idx1 < __in_rng1.size() && __idx < __num_eles_min; ++__idx1, ++__idx)
                {
                    __temp_out.set(__count, __in_rng1[__idx1]);
                    ++__count;
                }
            }
            __idx = __num_eles_min;
            return;
        }
    }

    const _ValueTypeRng1& __ele_rng1 = __in_rng1[__idx1];
    const _ValueTypeRng2& __ele_rng2 = __in_rng2[__idx2];
    if (__comp(__ele_rng1, __ele_rng2))
    {
        if constexpr (_CopyDiffSetA)
        {
            __temp_out.set(__count, __ele_rng1);
            ++__count;
        }
        ++__idx1;
        ++__idx;
    }
    else if (__comp(__ele_rng2, __ele_rng1))
    {
        if constexpr (_CopyDiffSetB)
        {
            __temp_out.set(__count, __ele_rng2);
            ++__count;
        }
        ++__idx2;
        ++__idx;
    }
    else // if neither element is less than the other, they are equal
    {
        if constexpr (_CopyMatch)
        {
            __temp_out.set(__count, __ele_rng1);
            ++__count;
        }
        ++__idx1;
        ++__idx2;
        __idx += 2;
    }
}

// Set operation generic implementation, used for serial set operation of intersection, difference, union, and
// symmetric difference.
template <bool _CopyMatch, bool _CopyDiffSetA, bool _CopyDiffSetB>
struct __set_generic_operation
{
    template <typename _InRng1, typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare>
    std::uint16_t
    operator()(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t __idx1, std::size_t __idx2,
               _SizeType __num_eles_min, _TempOutput& __temp_out, _Compare __comp) const
    {

        std::uint16_t __count = 0;
        _SizeType __idx = 0;
        bool __can_reach_rng1_end = __idx1 + __num_eles_min >= __in_rng1.size();
        bool __can_reach_rng2_end = __idx2 + __num_eles_min >= __in_rng2.size();

        if (!__can_reach_rng1_end && !__can_reach_rng2_end)
        {
            while (__idx < __num_eles_min)
            {
                // no bounds checking
                __set_generic_operation_iteration<_CopyMatch, _CopyDiffSetA, _CopyDiffSetB, false>(
                    __in_rng1, __in_rng2, __idx1, __idx2, __num_eles_min, __temp_out, __idx, __count, __comp);
            }
        }
        else
        {
            while (__idx < __num_eles_min)
            {
                //bounds check all
                __set_generic_operation_iteration<_CopyMatch, _CopyDiffSetA, _CopyDiffSetB, true>(
                    __in_rng1, __in_rng2, __idx1, __idx2, __num_eles_min, __temp_out, __idx, __count, __comp);
            }
        }
        return __count;
    }
};

// Set operation implementations using the generic implementation
using __set_intersection = __set_generic_operation<true, false, false>;
using __set_difference = __set_generic_operation<false, true, false>;
using __set_union = __set_generic_operation<true, true, true>;
using __set_symmetric_difference = __set_generic_operation<false, true, true>;

template <typename _SetTag>
struct __get_set_operation;

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_IntersectionTag<std::true_type>> : public __set_intersection
{
};

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_DifferenceTag<std::true_type>> : public __set_difference
{
};
template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_UnionTag<std::true_type>> : public __set_union
{
};

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_SymmetricDifferenceTag<std::true_type>>
    : public __set_symmetric_difference
{
};

// Locates and returns the "intersection" of a diagonal on the balanced path, based on merge path coordinates.
// It returns coordinates in each set of the intersection with a boolean representing if the diagonal is "starred",
// meaning that the balanced path "intersection" point does not lie directly on the diagonal, but one step forward in
// the second set.
// Some diagonals must be "starred" to ensure that matching elements between rng1 and rng2 are processed in pairs
// starting from the first of repeating value(s) in each range and a matched pair are not split between work-items.
template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
auto
__find_balanced_path_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __merge_path_rng1,
                                 const _Index __merge_path_rng2, _Compare __comp)
{
    // back up to balanced path divergence with a biased binary search
    bool __star = false;
    if (__merge_path_rng1 == 0 || __merge_path_rng2 == __rng2.size())
    {
        return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
    }

    auto __ele_val = __rng1[__merge_path_rng1 - 1];

    if (__comp(__ele_val, __rng2[__merge_path_rng2]))
    {
        // There is no chance that the balanced path differs from the merge path here, because the previous element of
        // rng1 does not match the next element of rng2. We can just return the merge path.
        return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
    }

    // find first element of repeating sequence in the first set of the previous element
    _Index __rng1_repeat_start = oneapi::dpl::__internal::__biased_lower_bound</*__last_bias=*/true>(
        __rng1, _Index{0}, __merge_path_rng1, __ele_val, __comp);
    // find first element of repeating sequence in the second set of the next element
    _Index __rng2_repeat_start = oneapi::dpl::__internal::__biased_lower_bound</*__last_bias=*/true>(
        __rng2, _Index{0}, __merge_path_rng2, __ele_val, __comp);

    _Index __rng1_repeats = __merge_path_rng1 - __rng1_repeat_start;
    _Index __rng2_repeats_bck = __merge_path_rng2 - __rng2_repeat_start;

    if (__rng2_repeats_bck >= __rng1_repeats)
    {
        // If we have at least as many repeated elements in rng2, we end up back on merge path
        return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
    }

    // Calculate the number of "unmatched" repeats in the first set, add one and divide by two to round up for a
    // possible star diagonal.
    _Index __fwd_search_count = (__rng1_repeats - __rng2_repeats_bck + 1) / 2;

    // Calculate the max location to search in the second set for future repeats, limiting to the edge of the range
    _Index __fwd_search_bound = std::min(__merge_path_rng2 + __fwd_search_count, __rng2.size());

    _Index __balanced_path_intersection_rng2 =
        oneapi::dpl::__internal::__pstl_upper_bound(__rng2, __merge_path_rng2, __fwd_search_bound, __ele_val, __comp);

    // Calculate the number of matchable "future" repeats in the second set
    _Index __matchable_forward_ele_rng2 = __balanced_path_intersection_rng2 - __merge_path_rng2;
    _Index __total_matched_rng2 = __balanced_path_intersection_rng2 - __rng2_repeat_start;

    // Update balanced path intersection for rng1, must account for cases where there are more repeating elements in
    // rng1 than matched elements of rng2
    _Index __balanced_path_intersection_rng1 =
        __rng1_repeat_start + std::max(__total_matched_rng2, __rng1_repeats - __matchable_forward_ele_rng2);

    // If we needed to step off the diagonal to find the balanced path, mark the diagonal as "starred"
    __star =
        __balanced_path_intersection_rng1 + __balanced_path_intersection_rng2 != __merge_path_rng1 + __merge_path_rng2;

    return std::make_tuple(__balanced_path_intersection_rng1, __balanced_path_intersection_rng2, __star);
}

// Reduce then scan building block for set balanced path which is used in the reduction kernel to calculate the
// balanced path intersection, store it to temporary data with "star" status, then count the number of elements to write
// to the output for the reduction operation.
template <typename _SetOpCount, typename _Compare>
struct __gen_set_balanced_path
{
    using TempData = __noop_temp_data;
    template <typename _InRng, typename _IndexT>
    std::uint16_t
    operator()(const _InRng& __in_rng, _IndexT __id, TempData& __temp_data) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        auto __rng1_temp_diag = std::get<2>(__in_rng.tuple()); // set a temp storage sequence

        using _SizeType = decltype(__rng1.size());
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= __rng1.size() + __rng2.size())
            return 0;
        //find merge path intersection
        auto [__rng1_pos, __rng2_pos] = oneapi::dpl::__par_backend_hetero::__find_start_point(
            __rng1, _SizeType{0}, __rng1.size(), __rng2, _SizeType{0}, __rng2.size(), __i_elem, __comp);

        //Find balanced path for diagonal start
        auto [__rng1_balanced_pos, __rng2_balanced_pos, __star_offset] =
            __find_balanced_path_start_point(__rng1, __rng2, __rng1_pos, __rng2_pos, __comp);

        // Use sign bit to represent star offset. Temp storage is a signed type equal to the difference_type of the
        // input iterator range. The index will fit into the positive portion of the type, so the sign may be used to
        // indicate the star offset.
        __rng1_temp_diag[__id] = __rng1_balanced_pos * (__star_offset ? -1 : 1);

        _SizeType __eles_to_process = std::min(__diagonal_spacing - (__star_offset ? _SizeType{1} : _SizeType{0}),
                                               __rng1.size() + __rng2.size() - (__i_elem - 1));

        std::uint16_t __count = __set_op_count(__rng1, __rng2, __rng1_balanced_pos, __rng2_balanced_pos,
                                               __eles_to_process, __temp_data, __comp);
        return __count;
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
    _Compare __comp;
};

// Reduce then scan building block for set balanced path which is used in the scan kernel to decode the stored balanced
// path intersection, perform the serial set operation for the diagonal, counting the number of elements and writing
// the output to temporary data in registers to be ready for the scan and write operations to follow.
template <typename _SetOpCount, typename _TempData, typename _Compare>
struct __gen_set_op_from_known_balanced_path
{
    using TempData = _TempData;
    template <typename _InRng, typename _IndexT>
    auto
    operator()(const _InRng& __in_rng, _IndexT __id, _TempData& __output_data) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        auto __rng1_temp_diag = std::get<2>(__in_rng.tuple()); // set a temp storage sequence, star value in sign bit
        using _SizeType = decltype(__rng1.size());
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= __rng1.size() + __rng2.size())
            return std::make_tuple(std::uint32_t{0}, std::uint16_t{0});
        _SizeType __star_offset = oneapi::dpl::__internal::__dpl_signbit(__rng1_temp_diag[__id]) ? 1 : 0;
        auto __rng1_temp_diag_abs = std::abs(__rng1_temp_diag[__id]);
        auto __rng2_temp_diag = __i_elem - __rng1_temp_diag_abs + __star_offset;

        _SizeType __eles_to_process =
            std::min(_SizeType{__diagonal_spacing} - __star_offset, __rng1.size() + __rng2.size() - (__i_elem - 1));

        std::uint16_t __count = __set_op_count(__rng1, __rng2, __rng1_temp_diag_abs, __rng2_temp_diag,
                                               __eles_to_process, __output_data, __comp);
        return std::make_tuple(std::uint32_t{__count}, __count);
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
    _Compare __comp;
};

// __parallel_reduce_by_segment_reduce_then_scan

// Generates input for a reduction operation by applying a binary predicate to the keys of the input range.
template <typename _BinaryPred>
struct __gen_red_by_seg_reduce_input
{
    using TempData = __noop_temp_data;
    // Returns the following tuple:
    // (new_seg_mask, value)
    // size_t new_seg_mask : 1 for a start of a new segment, 0 otherwise
    // ValueType value     : Current element's value for reduction
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        const auto __in_keys = std::get<0>(__in_rng.tuple());
        const auto __in_vals = std::get<1>(__in_rng.tuple());
        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        // The first segment start (index 0) is not marked with a 1. This is because we need the first
        // segment's key and value output index to be 0. We begin marking new segments only after the
        // first.
        const std::size_t __new_seg_mask = __id > 0 && !__binary_pred(__in_keys[__id - 1], __in_keys[__id]);
        return oneapi::dpl::__internal::make_tuple(__new_seg_mask, _ValueType{__in_vals[__id]});
    }
    _BinaryPred __binary_pred;
};

// Generates input for a scan operation by applying a binary predicate to the keys of the input range.
template <typename _BinaryPred>
struct __gen_red_by_seg_scan_input
{
    using TempData = __noop_temp_data;
    // Returns the following tuple:
    // ((new_seg_mask, value), output_value, next_key, current_key)
    // size_t new_seg_mask : 1 for a start of a new segment, 0 otherwise
    // ValueType value     : Current element's value for reduction
    // bool output_value   : Whether this work-item should write an output (end of segment)
    // KeyType next_key    : The key of the next segment to write if output_value is true
    // KeyType current_key : The current element's key. This is only ever used by work-item 0 to write the first key
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        const auto __in_keys = std::get<0>(__in_rng.tuple());
        const auto __in_vals = std::get<1>(__in_rng.tuple());
        using _KeyType = oneapi::dpl::__internal::__value_t<decltype(__in_keys)>;
        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        const _KeyType& __current_key = __in_keys[__id];
        const _ValueType& __current_val = __in_vals[__id];
        // Ordering the most common condition first has yielded the best results.
        if (__id > 0 && __id < __n - 1)
        {
            const _KeyType& __prev_key = __in_keys[__id - 1];
            const _KeyType& __next_key = __in_keys[__id + 1];
            const std::size_t __new_seg_mask = !__binary_pred(__prev_key, __current_key);
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(__new_seg_mask, __current_val),
                !__binary_pred(__current_key, __next_key), __next_key, __current_key);
        }
        else if (__id == __n - 1)
        {
            const _KeyType& __prev_key = __in_keys[__id - 1];
            const std::size_t __new_seg_mask = !__binary_pred(__prev_key, __current_key);
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(__new_seg_mask, __current_val), true, __current_key,
                __current_key); // Passing __current_key as the next key for the last element is a placeholder
        }
        else // __id == 0
        {
            const _KeyType& __next_key = __in_keys[__id + 1];
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(std::size_t{0}, __current_val),
                !__binary_pred(__current_key, __next_key), __next_key, __current_key);
        }
    }
    _BinaryPred __binary_pred;
    // For correctness of the function call operator, __n must be greater than 1.
    std::size_t __n;
};

// Reduction operation for reduce-by-segment
template <typename _BinaryOp>
struct __red_by_seg_op
{
    // Consider the following segment / value pairs that would be processed in reduce-then-scan by a sub-group of size 8:
    // ----------------------------------------------------------
    // Keys:   0 0 1 1 2 2 2 2
    // Values: 1 1 1 1 1 1 1 1
    // ----------------------------------------------------------
    // The reduce and scan input generation phase flags new segments (excluding index 0) for use in the sub-group scan
    // operation. The above key, value pairs correspond to the following flag, value pairs:
    // ----------------------------------------------------------
    // Flags:  0 0 1 0 1 0 0 0
    // Values: 1 1 1 1 1 1 1 1
    // ----------------------------------------------------------
    // The sub-group scan operation looks back by powers-of-2 applying encountered prefixes. The __red_by_seg_op
    // operation performs a standard inclusive scan over the flags to compute output indices while performing a masked
    // scan over values to avoid applying a previous segment's partial reduction. Previous value elements are reduced
    // so long as the current index's flag is 0, indicating that input within its segment is still being processed
    // ----------------------------------------------------------
    // Start:
    // ----------------------------------------------------------
    // Flags:  0 0 1 0 1 0 0 0
    // Values: 1 1 1 1 1 1 1 1
    // ----------------------------------------------------------
    // After step 1 (apply the i-1th value if the ith flag is 0):
    // ----------------------------------------------------------
    // Flags:  0 0 1 1 1 1 0 0
    // Values: 1 2 1 2 1 2 2 2
    // ----------------------------------------------------------
    // After step 2 (apply the i-2th value if the ith flag is 0):
    // ----------------------------------------------------------
    // Flags:  0 0 1 1 2 2 1 1
    // Values: 1 2 1 2 1 2 3 4
    // ----------------------------------------------------------
    // After step 3 (apply the i-4th value if the ith flag is 0):
    // ----------------------------------------------------------
    // Flags:  0 0 1 1 2 2 2 2
    // Values: 1 2 1 2 1 2 3 4
    //           ^   ^       ^
    // ----------------------------------------------------------
    // Note that the scan of segment flags results in the desired output index of the reduce_by_segment operation in
    // each segment and the item corresponding to the final key in a segment contains its output reduction value. This
    // operation is first applied within a sub-group and then across sub-groups, work-groups, and blocks to
    // reduce-by-segment across the full input. The result of these operations combined with cached key data in
    // __gen_red_by_seg_scan_input enables the write phase to output keys and reduction values.
    // =>
    // Segments : 0 1 2
    // Values   : 2 2 4
    template <typename _Tup1, typename _Tup2>
    auto
    operator()(const _Tup1& __lhs_tup, const _Tup2& __rhs_tup) const
    {
        using std::get;
        using _OpReturnType = decltype(__binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        // The left-hand side has processed elements from the same segment, so update the reduction value.
        if (get<0>(__rhs_tup) == 0)
        {
            return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup),
                                                       __binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        }
        // We are looking at elements from a previous segment so just update the output index.
        return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup) + get<0>(__rhs_tup),
                                                   _OpReturnType{get<1>(__rhs_tup)});
    }
    _BinaryOp __binary_op;
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_SCAN_HELPERS_H
