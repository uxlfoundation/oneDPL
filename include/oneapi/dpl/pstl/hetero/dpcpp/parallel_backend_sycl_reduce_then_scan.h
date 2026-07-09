// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_then_scan.h ---------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <cmath>
#include <cassert>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"
#include "../../functional_impl.h" // for oneapi::dpl::identity

#include "../../tuple_impl.h"
#include "../../utils_ranges.h"
#include "../utils_hetero.h"
#include "parallel_backend_sycl_reduce_then_scan_pos_tools.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// *** Reduce then scan functional building blocks ***
// *** Utilities ***

// Temporary data structure which is used to store results to registers during a reduce then scan operation.
template <std::uint16_t elements, typename _ValueT>
struct __temp_data_array
{
    // The maximum number of output elements a single scanned element may emit through this temporary data.
    // For set operations a scanned element is a diagonal which can produce up to `elements` outputs, so the
    // bounded-write estimate must account for this many writes per scanned element.
    static constexpr std::uint16_t __max_outputs_per_input = elements;

    // We don't capture source data indexes in this structure
    static constexpr bool __capture_indexes_flag = false;

    template <typename _ValueT2>
    void
    set(std::uint16_t __idx, const _ValueT2& __ele)
    {
        __data[__idx].__setup(__ele);
    }

    _ValueT
    get_and_destroy(std::uint16_t __idx)
    {
        // Setting up temporary value to be destroyed as this function exits. The __scoped_destroyer calls destroy when
        // it leaves scope.
        oneapi::dpl::__internal::__scoped_destroyer<_ValueT> __destroy_when_leaving_scope{__data[__idx]};
        return __data[__idx].__v;
    }

    oneapi::dpl::__internal::__lazy_ctor_storage<_ValueT> __data[elements];
};

// This is a stand-in for a temporary data structure which is used to turn set() into a no-op. This is used in the case
// where no temporary register data is needed within reduce then scan kern
struct __noop_temp_data
{
    static constexpr std::uint16_t __max_outputs_per_input = 1;

    // We don't capture source data indexes in this structure
    static constexpr bool __capture_indexes_flag = false;

    template <typename _ValueT>
    void
    set(std::uint16_t, const _ValueT&) const
    {
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
    template <typename _OutRng, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _ValueType& __v) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__v)>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        __out_rng[__id] = static_cast<_ConvertedTupleType>(__v);
    }
};

// Writes a single element `get<2>(__v)` to the output range at the index, `get<0>(__v) - 1 + __offset`, but only if the
// condition `get<1>(__v)` is `true`. Used in __parallel_copy_if, __parallel_unique_copy
template <std::int32_t __offset, typename _Assign>
struct __write_to_id_if
{
    template <typename _OutRng, typename _SizeType, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        if (std::get<1>(__v))
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)), __out_rng[std::get<0>(__v) - 1 + __offset]);
    }

    template <typename _OutRng, typename _OutSize, typename _SizeType, typename _ValueType, typename _OnOOBReached>
    void
    operator()(_OutRng& __out_rng, const _OutSize __out_size, _SizeType __id, const _ValueType& __v,
               _OnOOBReached __on_oob_reached) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        if (std::get<1>(__v))
        {
            const auto __out_idx = std::get<0>(__v) - 1 + __offset;

            if (__out_idx < __out_size)
                __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)), __out_rng[__out_idx]);
            if (__out_idx == __out_size)
                __on_oob_reached(__id, __id);
        }
    }
    _Assign __assign;
};

// Writes a single element `get<2>(__v)` to the output range at the index, `get<0>(__v) - 1`, but only if the
// condition `get<1>(__v)` is `true`. Otherwise, writes the element to the output range at the index,
// `__id - get<0>(__v)`. Used for __parallel_partition_copy.
template <typename _Assign>
struct __write_to_id_if_else
{
    template <typename _OutRng, typename _SizeType, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
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
    template <typename _OutRng, typename _Tup>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _Tup& __tup) const
    {
        using std::get;

        // Get source tuple
        auto&& __tuple = __out_rng.base();

        auto __out_keys = get<0>(__tuple);
        auto __out_values = get<1>(__tuple);

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

template <bool __is_inclusive, typename _InitType, typename _BinaryOp>
struct __write_scan_by_seg
{
    _InitType __init_value;
    _BinaryOp __binary_op;

    template <typename _OutRng, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _ValueType& __v) const
    {
        using std::get;
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(get<1>(get<0>(__v)))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        if constexpr (__is_inclusive)
        {
            static_assert(std::is_same_v<_InitType,
                                         oneapi::dpl::unseq_backend::__no_init_value<typename _InitType::__value_type>>,
                          "inclusive_scan_by_segment must not have an initial element");
            __out_rng[__id] = static_cast<_ConvertedTupleType>(get<1>(get<0>(__v)));
        }
        else
        {
            static_assert(
                std::is_same_v<_InitType, oneapi::dpl::unseq_backend::__init_value<typename _InitType::__value_type>>,
                "exclusive_scan_by_segment must have an initial element");
            __out_rng[__id] =
                get<1>(__v) ? static_cast<_ConvertedTupleType>(__init_value.__value)
                            : static_cast<_ConvertedTupleType>(__binary_op(__init_value.__value, get<1>(get<0>(__v))));
        }
    }
};

// Writes multiple elements from temp data to the output range. The values to write are stored in `__temp_data` from a
// previous operation, and must be written to the output range in the appropriate location. The zeroth element of `__v`
// will contain the index of one past the last element to write, and the first element of `__v` will contain the number
// of elements to write. Used for __parallel_set_write_a_b_op.
template <typename _Assign>
struct __write_multiple_to_id
{
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
        const std::size_t __n = std::get<1>(__v);
        for (std::size_t __i = 0; __i < __n; ++__i)
        {
            __assign(static_cast<_ConvertedTupleType>(__temp_data.get_and_destroy(__i)),
                     __out_rng[std::get<0>(__v) - std::get<1>(__v) + __i]);
        }
    }

    template <typename _OutRng, typename _OutSize, typename _SizeType, typename _ValueType, typename _TempData,
              typename _OnOOBReached>
    void
    operator()(_OutRng& __out_rng, const _OutSize __out_size, _SizeType __id, const _ValueType& __v,
               _TempData& __temp_data, _OnOOBReached __on_oob_reached) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__temp_data.get_and_destroy(0))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        const std::size_t __n = std::get<1>(__v);
        for (std::size_t __i = 0; __i < __n; ++__i)
        {
            // Retrieve and destroy the temporary slot unconditionally, before the bounds check. Each slot was
            // constructed by the generator via placement-new (__temp_data_array::set) and requires an explicit
            // matching destruction. Since __temp_data is reused for the next scanned element, destroying only on the
            // in-bounds path would leak slots whose output index falls outside __out_size (the bounded/overflow case)
            // and let the next set() placement-new over a still-live object.
            auto&& __val = __temp_data.get_and_destroy(__i);

            const auto __out_idx = std::get<0>(__v) - std::get<1>(__v) + __i;
            if (__out_idx < __out_size)
                __assign(static_cast<_ConvertedTupleType>(std::forward<decltype(__val)>(__val)), __out_rng[__out_idx]);

            // Report the source id of the current diagonal together with the local element offset within the
            // temporary data. This is enough to recover the source index pair later (by re-running the serial
            // generator for this diagonal) without re-running any sub-group collective operation.
            if (__out_idx == __out_size)
                __on_oob_reached(__id, __i);
        }
    }
    _Assign __assign;
};

// *** Algorithm Specific Helpers, Input Generators to Reduction and Scan Operations ***

// __parallel_transform_scan

// A generator which applies a unary operation to the input range element at an index and returns the result
// converted to an underlying init type.
template <typename _UnaryOp, typename _InitType>
struct __gen_transform_input
{
    template <typename _InRng>
    using __result_t = _InitType;

    template <typename _InRng>
    __result_t<_InRng>
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // We explicitly convert __in_rng[__id] to the value type of _InRng to properly handle the case where we
        // process zip_iterator input where the reference type is a tuple of a references. This prevents the caller
        // from modifying the input range when altering the return of this functor.
        using _ValueType = oneapi::dpl::__internal::__value_t<_InRng>;
        return static_cast<_InitType>(__unary_op(_ValueType{__in_rng[__id]}));
    }
    _UnaryOp __unary_op;
};

// Scan copy algorithms (partition_copy, copy_if, unique_copy)

// A mask generator to filter the input range based on a predicate, returning true if satisfied at an index.
template <typename _Predicate, typename _RangeTransform = oneapi::dpl::identity>
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
template <typename _GenMask, typename _RetType>
struct __gen_count_mask
{
    template <typename _InRng>
    _RetType
    operator()(_InRng&& __in_rng, _RetType __id) const
    {
        return __gen_mask(std::forward<_InRng>(__in_rng), __id) ? _RetType{1} : _RetType{0};
    }
    _GenMask __gen_mask;
};

// A generator which expands the mask generator to return a tuple containing the count, mask, and the element at the
// specified index.
template <typename _GenMask, typename _RetType, typename _RangeTransform = oneapi::dpl::identity>
struct __gen_expand_count_mask
{
    template <typename _InRng>
    using __element_t =
        oneapi::dpl::__internal::__value_t<decltype(std::declval<const _RangeTransform&>()(std::declval<_InRng&>()))>;

    template <typename _InRng>
    using __result_t = std::tuple<_RetType, bool, __element_t<_InRng>>;

    template <typename _InRng>
    __result_t<_InRng>
    operator()(_InRng&& __in_rng, _RetType __id) const
    {
        auto __transformed_input = __rng_transform(__in_rng);
        // Explicitly creating this element by value is necessary to avoid modifying the input data when _InRng is a
        //  zip_iterator which will return a tuple of references when dereferenced. With this explicit type, we copy
        //  the values of zipped input types rather than their references.
        __element_t<_InRng> ele = __transformed_input[__id];
        bool mask = __gen_mask(std::forward<_InRng>(__in_rng), __id);
        return __result_t<_InRng>(mask ? _RetType{1} : _RetType{0}, mask, ele);
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

// __parallel_set_write_a_b_op

// Set operation generic implementation, used for serial set operation of intersection, difference, union, and
// symmetric difference.
template <typename _SetTag>
struct __set_operation
{
  protected:
    static constexpr bool __is_set_difference           = std::is_same_v<_SetTag, unseq_backend::_DifferenceTag>;
    static constexpr bool __is_set_intersection         = std::is_same_v<_SetTag, unseq_backend::_IntersectionTag>;
    static constexpr bool __is_set_symmetric_difference = std::is_same_v<_SetTag, unseq_backend::_SymmetricDifferenceTag>;
    static constexpr bool __is_set_union                = std::is_same_v<_SetTag, unseq_backend::_UnionTag>;

    static constexpr bool _CopyMatch = __is_set_intersection || __is_set_union;
    static constexpr bool _CopyDiffSetA = !__is_set_intersection;
    static constexpr bool _CopyDiffSetB = __is_set_union || __is_set_symmetric_difference;

  public:
    template <typename _InRng1, typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare,
              typename _Proj1, typename _Proj2, typename _FinalPosSaver>
    _SizeType
    operator()(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t __idx1, std::size_t __idx2,
               const _SizeType __num_eles_min, _TempOutput& __temp_out, const _Compare __comp, _Proj1 __proj1,
               _Proj2 __proj2, _FinalPosSaver __final_pos_saver) const
    {
        const auto __size1 = oneapi::dpl::__ranges::__size(__in_rng1);
        const auto __size2 = oneapi::dpl::__ranges::__size(__in_rng2);

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

        _SizeType __count = 0;
        _SizeType __idx = 0;

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
                            __write_temp_element(__temp_out, __count, __in_rng2[__idx2], __idx1, __idx2);
                            ++__count;
                        }
                    }
                    __idx = __num_eles_min;
                    if constexpr (__need_call_final_pos_saver)
                        if (__need_setup_final_pos(__size1, __size2, __idx1_at_entry, __idx2_at_entry, __idx1, __idx2))
                            __final_pos_saver({__idx1, __idx2});
                    continue;
                }

                if (__idx2 == __size2)
                {
                    if constexpr (_CopyDiffSetA)
                    {
                        // If we are at the end of rng2, copy the rest of rng1 within our diagonal's bounds
                        for (; __idx1 < __size1 && __idx < __num_eles_min; ++__idx1, ++__idx)
                        {
                            __write_temp_element(__temp_out, __count, __in_rng1[__idx1], __idx1, __idx2);
                            ++__count;
                        }
                    }
                    __idx = __num_eles_min;
                    if constexpr (__need_call_final_pos_saver)
                        if (__need_setup_final_pos(__size1, __size2, __idx1_at_entry, __idx2_at_entry, __idx1, __idx2))
                            __final_pos_saver({__idx1, __idx2});
                    continue;
                }
            }

            auto&& __ele_rng1 = __in_rng1[__idx1];
            auto&& __ele_rng2 = __in_rng2[__idx2];

            auto&& __ele_rng1_proj = std::invoke(__proj1, __ele_rng1);
            auto&& __ele_rng2_proj = std::invoke(__proj2, __ele_rng2);

            if (std::invoke(__comp, __ele_rng1_proj, __ele_rng2_proj))
            {
                if constexpr (_CopyDiffSetA)
                {
                    __write_temp_element(__temp_out, __count, std::forward<decltype(__ele_rng1)>(__ele_rng1), __idx1, __idx2);
                    ++__count;
                }
                ++__idx1;
                ++__idx;
            }
            else if (std::invoke(__comp, __ele_rng2_proj, __ele_rng1_proj))
            {
                if constexpr (_CopyDiffSetB)
                {
                    __write_temp_element(__temp_out, __count, std::forward<decltype(__ele_rng2)>(__ele_rng2), __idx1, __idx2);
                    ++__count;
                }
                ++__idx2;
                ++__idx;
            }
            else // if neither element is less than the other, they are equal
            {
                if constexpr (_CopyMatch)
                {
                    __write_temp_element(__temp_out, __count, std::forward<decltype(__ele_rng1)>(__ele_rng1), __idx1, __idx2);
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
                {
                    if constexpr (__need_call_final_pos_saver)
                        if (__need_setup_final_pos(__size1, __size2, __idx1_at_entry, __idx2_at_entry, __idx1, __idx2))
                            __final_pos_saver({__idx1, __idx2});
                }
            }
        }

        return __count;
    }

  protected:

    template <typename _TempOutput, typename _SizeType>
    void
    __write_temp_element(_TempOutput& __temp_out, const _SizeType __idx_tmp, const auto& __value,
                         const std::size_t __idx1, const std::size_t __idx2) const
    {
        if constexpr (_TempOutput::__capture_indexes_flag)
            __temp_out.set(__idx_tmp, __value, {__idx1, __idx2});
        else
            __temp_out.set(__idx_tmp, __value);
    }

    bool
    __need_setup_final_pos(const auto __size1, const auto __size2, const std::size_t __idx1_at_entry,
                           const std::size_t __idx2_at_entry, const std::size_t __idx1, const std::size_t __idx2) const
    {
        // For set_intersection the operation terminates once either input is exhausted: the edge crossing
        // is the step that moves from "strictly inside both inputs" to "at least one input exhausted".
        if constexpr (__is_set_intersection)
        {
            return __idx1_at_entry < __size1 && __idx2_at_entry < __size2 &&
                    (__idx1 == __size1 || __idx2 == __size2);
        }
        // For set_difference the operation terminates once the first input (set A) is exhausted: the edge
        // crossing is the step that moves idx1 to the end of set A. This also covers the tail-drain of set
        // A performed after set B is exhausted.
        else if constexpr (__is_set_difference)
        {
            return __idx1_at_entry < __size1 && __idx1 == __size1;
        }

        return false;
    }
};

template <bool __return_star, typename _Rng, typename _IdxT>
auto
__decode_balanced_path_temp_data_impl(const _Rng& __rng, const _IdxT __id, const std::uint16_t __diagonal_spacing)
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
__decode_balanced_path_temp_data_no_star(const _Rng& __rng, const _IdxT __id, const std::uint16_t __diagonal_spacing)
{
    return __decode_balanced_path_temp_data_impl<false>(__rng, __id, __diagonal_spacing);
}

template <typename _Rng, typename _IdxT>
std::tuple<_IdxT, _IdxT, decltype(oneapi::dpl::__ranges::__size(std::declval<_Rng>()))>
__decode_balanced_path_temp_data(const _Rng& __rng, const _IdxT __id, const std::uint16_t __diagonal_spacing)
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
    std::uint16_t __diagonal_spacing;
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
template <typename _SetOpCount, typename _BoundsProvider, typename _RetType, typename _Compare, typename _Proj1,
          typename _Proj2>
struct __gen_set_balanced_path
{
    using TempData = __noop_temp_data;

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

        // Calculate the number of "unmatched" repeats in the first set, add one and divide by two to round up for a
        // possible star diagonal.
        _Index __fwd_search_count = (__rng1_repeats - __rng2_repeats_bck + 1) / 2;

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
    template <typename _InRng, typename _IndexT, typename _FinalPosSaver>
    _RetType
    operator()(const _InRng& __in_rng, _IndexT __id, TempData& __temp_data, _FinalPosSaver __final_pos_saver) const
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
            return 0;
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

        _IndexT __eles_to_process =
            std::min(_IndexT{__diagonal_spacing} - (__star ? _IndexT{1} : _IndexT{0}),
                     oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2) -
                         _IndexT{__id * __diagonal_spacing - 1});

        return __set_op_count(__rng1, __rng2, __rng1_balanced_pos, __rng2_balanced_pos, __eles_to_process, __temp_data,
                              __comp, __proj1, __proj2, __final_pos_saver);
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
    _BoundsProvider __get_bounds;
    _Compare __comp;
    _Proj1 __proj1;
    _Proj2 __proj2;
};

// Reduce then scan building block for set balanced path which is used in the scan kernel to decode the stored balanced
// path intersection, perform the serial set operation for the diagonal, counting the number of elements and writing
// the output to temporary data in registers to be ready for the scan and write operations to follow.
template <typename _SetOpCount, typename _TempData, typename _RetType, typename _Compare, typename _Proj1,
          typename _Proj2>
struct __gen_set_op_from_known_balanced_path
{
    template <typename _InRng>
    using __result_t = std::tuple<_RetType, _RetType>;

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
            return __result_t<_InRng>{_RetType{0}, _RetType{0}};
        auto [__rng1_idx, __rng2_idx, __star_offset] =
            oneapi::dpl::__par_backend_hetero::__decode_balanced_path_temp_data(__rng1_temp_diag, __id,
                                                                                __diagonal_spacing);

        _RetType __eles_to_process = static_cast<_RetType>(
            std::min(static_cast<_SizeType>(__diagonal_spacing - __star_offset),
                     static_cast<_SizeType>(oneapi::dpl::__ranges::__size(__rng1) +
                                            oneapi::dpl::__ranges::__size(__rng2) - __i_elem + 1)));

        _RetType __count = __set_op_count(__rng1, __rng2, __rng1_idx, __rng2_idx, __eles_to_process, __output_data,
                                          __comp, __proj1, __proj2, __final_pos_saver);

        return __result_t<_InRng>{__count, __count};
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
    _Compare __comp;
    _Proj1 __proj1;
    _Proj2 __proj2;
};

template <typename _SetOpCount, typename _TempData, typename _RetType, typename _Compare, typename _Proj1,
          typename _Proj2>
struct __internal::__detect_oob_in_two_steps_selector<
    __gen_set_op_from_known_balanced_path<_SetOpCount, _TempData, _RetType, _Compare, _Proj1, _Proj2>> : std::true_type
{
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

// __parallel_reduce_by_segment_reduce_then_scan

// Generates input for a reduction operation by applying a binary predicate to the keys of the input range.
template <typename _BinaryPred>
struct __gen_red_by_seg_reduce_input
{
    // Returns the following tuple:
    // (new_seg_mask, value)
    // size_t new_seg_mask : 1 for a start of a new segment, 0 otherwise
    // ValueType value     : Current element's value for reduction
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        const auto __in_keys = std::get<0>(__tuple);
        const auto __in_vals = std::get<1>(__tuple);

        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        // The first segment start (index 0) is not marked with a 1. This is because we need the first
        // segment's key and value output index to be 0. We begin marking new segments only after the
        // first.
        const std::size_t __new_seg_mask = __id > 0 && !__binary_pred(__in_keys[__id - 1], __in_keys[__id]);
        return oneapi::dpl::__internal::make_tuple(__new_seg_mask, _ValueType{__in_vals[__id]});
    }
    _BinaryPred __binary_pred;
};

template <typename _BinaryPred>
struct __gen_scan_by_seg_reduce_input
{
    // Returns the following tuple:
    // (new_seg_mask, value)
    // bool new_seg_mask : true for a start of a new segment, false otherwise
    // ValueType value   : Current element's value for reduction
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // Get source base
        auto&& __tuple = __in_rng.base();

        const auto __in_keys = std::get<0>(__tuple);
        const auto __in_vals = std::get<1>(__tuple);

        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        const std::uint32_t __new_seg_mask = __id == 0 || !__binary_pred(__in_keys[__id - 1], __in_keys[__id]);
        return oneapi::dpl::__internal::make_tuple(__new_seg_mask, _ValueType{__in_vals[__id]});
    }
    _BinaryPred __binary_pred;
};

// Generates input for a scan operation by applying a binary predicate to the keys of the input range.
template <typename _BinaryPred>
struct __gen_red_by_seg_scan_input
{
    template <typename _InRng>
    using __key_t = oneapi::dpl::__internal::__value_t<decltype(std::get<0>(std::declval<_InRng>().base()))>;
    template <typename _InRng>
    using __val_t = oneapi::dpl::__internal::__value_t<decltype(std::get<1>(std::declval<_InRng>().base()))>;
    template <typename _InRng>
    using __result_t = oneapi::dpl::__internal::tuple<oneapi::dpl::__internal::tuple<std::size_t, __val_t<_InRng>>,
                                                      bool, __key_t<_InRng>, __key_t<_InRng>>;
    // Returns the following tuple:
    // ((new_seg_mask, value), output_value, next_key, current_key)
    // size_t new_seg_mask : 1 for a start of a new segment, 0 otherwise
    // ValueType value     : Current element's value for reduction
    // bool output_value   : Whether this work-item should write an output (end of segment)
    // KeyType next_key    : The key of the next segment to write if output_value is true
    // KeyType current_key : The current element's key. This is only ever used by work-item 0 to write the first key
    template <typename _InRng>
    __result_t<_InRng>
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        const auto __in_keys = std::get<0>(__tuple);
        const auto __in_vals = std::get<1>(__tuple);

        const __key_t<_InRng>& __current_key = __in_keys[__id];
        const __val_t<_InRng>& __current_val = __in_vals[__id];
        // Ordering the most common condition first has yielded the best results.
        if (__id > 0 && __id < __n - 1)
        {
            const __key_t<_InRng>& __prev_key = __in_keys[__id - 1];
            const __key_t<_InRng>& __next_key = __in_keys[__id + 1];
            const std::size_t __new_seg_mask = !__binary_pred(__prev_key, __current_key);
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(__new_seg_mask, __current_val),
                !__binary_pred(__current_key, __next_key), __next_key, __current_key);
        }
        else if (__id == __n - 1)
        {
            const __key_t<_InRng>& __prev_key = __in_keys[__id - 1];
            const std::size_t __new_seg_mask = !__binary_pred(__prev_key, __current_key);
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(__new_seg_mask, __current_val), true, __current_key,
                __current_key); // Passing __current_key as the next key for the last element is a placeholder
        }
        else // __id == 0
        {
            const __key_t<_InRng>& __next_key = __in_keys[__id + 1];
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(std::size_t{0}, __current_val),
                !__binary_pred(__current_key, __next_key), __next_key, __current_key);
        }
    }
    _BinaryPred __binary_pred;
    // For correctness of the function call operator, __n must be greater than 1.
    std::size_t __n;
};

template <typename _BinaryPred>
struct __gen_scan_by_seg_scan_input
{
    // Returns the following tuple:
    // ((new_seg_mask, value), new_seg_mask)
    // bool new_seg_mask : true for a start of a new segment, false otherwise
    // ValueType value   : Current element's value for reduction
    template <typename _InRng>
    using __val_t = oneapi::dpl::__internal::__value_t<decltype(std::get<1>(std::declval<_InRng>().base()))>;
    template <typename _InRng>
    using __result_t =
        oneapi::dpl::__internal::tuple<oneapi::dpl::__internal::tuple<std::uint32_t, __val_t<_InRng>>, std::uint32_t>;

    template <typename _InRng>
    __result_t<_InRng>
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // Get source tuple
        auto&& __tuple = __in_rng.base();

        const auto __in_keys = std::get<0>(__tuple);
        const auto __in_vals = std::get<1>(__tuple);

        // Mark the first index as a new segment as well as an indexing corresponding to any key
        // that does not satisfy the binary predicate with the previous key. The first tuple mask element
        // is scanned over, and the third is a placeholder for exclusive_scan_by_segment to perform init
        // handling in the output write.
        const std::uint32_t __new_seg_mask = __id == 0 || !__binary_pred(__in_keys[__id - 1], __in_keys[__id]);
        return __result_t<_InRng>{oneapi::dpl::__internal::make_tuple(__new_seg_mask, __val_t<_InRng>{__in_vals[__id]}),
                                  __new_seg_mask};
    }
    _BinaryPred __binary_pred;
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
        if (get<0>(__rhs_tup) == 0)
        {
            // The left-hand side and right-hand side are processing the same segment, so update the reduction value.
            // We additionally propagate the left-hand side's flag get<0>(__lhs_tup) forward to communicate in the next
            // iteration if the segment end has been found.
            return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup),
                                                       __binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        }
        // We are looking at elements from a previous segment so just update the output index.
        return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup) + get<0>(__rhs_tup),
                                                   _OpReturnType{get<1>(__rhs_tup)});
    }
    _BinaryOp __binary_op;
};

template <typename _BinaryOp>
struct __scan_by_seg_op
{
    template <typename _Tup1, typename _Tup2>
    auto
    operator()(const _Tup1& __lhs_tup, const _Tup2& __rhs_tup) const
    {
        using std::get;
        using _OpReturnType = decltype(__binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        if (get<0>(__rhs_tup) == 0)
        {
            // The left-hand side and right-hand side are processing on the same segment, so update the scan value. We
            // additionally propagate the left-hand side's flag get<0>(__lhs_tup) forward to communicate in the next
            // iteration if the segment end has been found.
            return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup),
                                                       __binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        }
        // We are looking at elements from a previous segment, so no operation is performed
        return oneapi::dpl::__internal::make_tuple(std::uint32_t{1}, _OpReturnType{get<1>(__rhs_tup)});
    }
    _BinaryOp __binary_op;
};

// *** Main reduce then scan infrastructure ***

// Sub-group communication wrappers with SLM fallback.
// The scan tag dictates what implementation paths are available in the kernel, and holds a slm pointer if applicable.

// For KT kernels, or after the runtime branch where only the subgroup operations are available
struct __subgroup_only_tag
{
};

// For non-trivially copyable types, only the SLM fallback is available.
template <typename _ValueType>
struct __slm_only_tag
{
    _ValueType* __value_ptr = nullptr;
};

// For trivially copyable types, both paths are available in the kernel, and a runtime parameter chooses between them.
// This enables CPU trivially copyable types to use the SLM-based path, which is considerably faster.
template <typename _ValueType>
struct __slm_or_subgroup_tag
{
    _ValueType* __value_ptr = nullptr;
};

// To workaround a hardware bug on certain Intel iGPUs with older driver versions and -O0 device compilation, use a
// sub-group size of 16. Note this function may only be called on the device as _ONEDPL_DETECT_SPIRV_COMPILATION is only
// valid here.
constexpr inline std::uint8_t
__get_reduce_then_scan_req_sg_sz_device()
{
#if _ONEDPL_DETECT_COMPILER_OPTIMIZATIONS_ENABLED || !_ONEDPL_DETECT_SPIRV_COMPILATION
    return 32; // best value
#else
    return 16; // workaround value
#endif
}

#if _ONEDPL_DETECT_SPIRV_COMPILATION

// On SPIR-V, the reqd_sub_group_size attribute is honored, and we ensure the workgroup size is a multiple of the
// subgroup size, so the sub-group size is a known fixed compile time constant for all subgroups.
constexpr inline std::uint8_t
__get_reduce_then_scan_actual_sub_group_size(const sycl::sub_group&)
{
    return __get_reduce_then_scan_req_sg_sz_device();
}
#else
// When not compiling for SPIR-V, we must use the real subgroup size obtained at runtime, which may be
// implementation-defined and may even differ between subgroups within a work-group per the SYCL spec.
// This will cause performance degradation, but is required for correctness according to the sycl specification.
// In practice, most implementations will have a fixed subgroup size, but this fallback ensures correctness even
// if that is not the case.
inline std::uint8_t
__get_reduce_then_scan_actual_sub_group_size(const sycl::sub_group& __sg)
{
    return __sg.get_local_range()[0];
}
#endif

// Work-group-local id of this sub-group's first work-item. This is the sub-group's contiguous base within
// the work-group; it is used to index work-group-sized SLM and is valid for any (even non-uniform)
// sub-group layout, unlike sub_group_id * size which assumes uniform sizes.
inline std::uint32_t
__get_sub_group_base(const sycl::nd_item<1>& __ndi)
{
    return static_cast<std::uint32_t>(__ndi.get_local_linear_id()) - __ndi.get_sub_group().get_local_linear_id();
}

inline std::uint32_t
__count_active_sub_groups(const sycl::nd_item<1>& __ndi, std::size_t __inputs_remaining,
                          std::uint32_t __inputs_per_item, std::uint32_t __inputs_per_work_group)
{
    // If all work-items are active, all subgroups are active.
    if (__inputs_remaining > __inputs_per_work_group - __inputs_per_item)
        return __ndi.get_sub_group().get_group_linear_range();
    // Calculate active subgroups with a broadcast. Past the branch above, __inputs_remaining is known to fit in uint32.
    // With a possible arbitrary layout of differently sized subgroups (from sycl spec), the sub-group id of a
    // given work-item is not known, but the last active work item is known.
    const std::uint32_t __active_work_items =
        oneapi::dpl::__internal::__dpl_ceiling_div(static_cast<std::uint32_t>(__inputs_remaining), __inputs_per_item);
    const std::uint32_t __last_active_sub_group_id = __dpl_sycl::__group_broadcast(
        __ndi.get_group(), __ndi.get_sub_group().get_group_linear_id(), __active_work_items - 1);
    return __last_active_sub_group_id + 1;
}

// Layer of abstraction that hoists the runtime "sub-group ops vs SLM fallback" decision to a single branch point.
// The communication tag passed in encodes which path(s) are available:
//   - __slm_or_subgroup_tag: both paths are compiled in; the runtime pointer (null => sub-group ops, non-null =>
//                            SLM) selects the concrete tag, so the body is instantiated once per path and the
//                            branch happens a single time at each call.
//   - __slm_only_tag / __subgroup_only_tag: only that single path is compiled; the body is invoked directly.
template <typename _ValueType, typename _Body>
void
__dispatch_comm_tag(__slm_or_subgroup_tag<_ValueType> __comm_tag, _Body&& __body)
{
    if (!__comm_tag.__value_ptr)
        __body(__subgroup_only_tag{});
    else
        __body(__slm_only_tag<_ValueType>{__comm_tag.__value_ptr});
}

template <typename _ValueType, typename _Body>
void
__dispatch_comm_tag(__slm_only_tag<_ValueType> __comm_tag, _Body&& __body)
{
    __body(__comm_tag);
}

template <typename _Body>
void
__dispatch_comm_tag(__subgroup_only_tag, _Body&& __body)
{
    __body(__subgroup_only_tag{});
}

template <typename _ValueType>
_ValueType
__shift_sub_group_right(const sycl::nd_item<1>& __ndi, _ValueType __value, std::uint32_t __shift, __subgroup_only_tag)
{
    return sycl::shift_group_right(__ndi.get_sub_group(), __value, __shift);
}

template <typename _ValueType>
_ValueType
__shift_sub_group_right(const sycl::nd_item<1>& __ndi, _ValueType __value, std::uint32_t __shift,
                        __slm_only_tag<_ValueType> __comm_slm)
{
    // SLM-based fallback: used for non-trivially-copyable types or when SLM communication is
    // preferred (e.g., CPU targets where sub-group ops are slow).
    const __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
    const std::uint32_t __sg_base = __get_sub_group_base(__ndi);
    std::uint32_t __local_id = __sub_group.get_local_linear_id();
    __comm_slm.__value_ptr[__sg_base + __local_id] = __value;
    sycl::group_barrier(__sub_group);
    _ValueType __result =
        __comm_slm.__value_ptr[__sg_base + ((__local_id >= __shift) ? __local_id - __shift : __local_id)];
    sycl::group_barrier(__sub_group);
    return __result;
}

template <typename _ValueType, typename _IdType>
_ValueType
__broadcast_sub_group(const sycl::nd_item<1>& __ndi, _ValueType __value, _IdType __broadcast_id, __subgroup_only_tag)
{
    return sycl::group_broadcast(__ndi.get_sub_group(), __value, __broadcast_id);
}

template <typename _ValueType, typename _IdType>
_ValueType
__broadcast_sub_group(const sycl::nd_item<1>& __ndi, _ValueType __value, _IdType __broadcast_id,
                      __slm_only_tag<_ValueType> __comm_slm)
{
    // SLM-based fallback: used for non-trivially-copyable types or when SLM communication is
    // preferred (e.g., CPU targets where sub-group ops are slow).
    const __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
    const std::uint32_t __sg_base = __get_sub_group_base(__ndi);
    __comm_slm.__value_ptr[__sg_base + __sub_group.get_local_linear_id()] = __value;
    sycl::group_barrier(__sub_group);
    _ValueType __result = __comm_slm.__value_ptr[__sg_base + __broadcast_id];
    sycl::group_barrier(__sub_group);
    return __result;
}

template <bool __is_inclusive, typename _MaskOp, typename _InitBroadcastId, typename _BinaryOp, typename _ValueType,
          typename _CommTag>
void
__sub_group_masked_scan(const sycl::nd_item<1>& __ndi, _MaskOp __mask_fn, _InitBroadcastId __init_broadcast_id,
                        _ValueType& __value, _BinaryOp __binary_op,
                        oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __init_and_carry,
                        _CommTag __comm_tag)
{
    std::uint8_t __sub_group_local_id = __ndi.get_sub_group().get_local_linear_id();
    const std::uint8_t __sub_group_size = __get_reduce_then_scan_actual_sub_group_size(__ndi.get_sub_group());
    for (std::uint8_t __shift = 1; __shift < __sub_group_size; __shift <<= 1)
    {
        _ValueType __partial_carry_in = __shift_sub_group_right(__ndi, __value, __shift, __comm_tag);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }

    // For an inclusive scan __old_init is never used, so avoid instantiating storage for it.
    // exclusive scan does have one instance of an invocation without an init, in scan_by_segment, so we must use
    // lazy storage to avoid default constructing in that case.
    std::conditional_t<__is_inclusive, oneapi::dpl::internal::ignore_copyable,
                       oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>>
        __old_init;
    if (__init_and_carry.__has_value())
    {
        __value = __binary_op(__init_and_carry.__get_cref(), __value);
        if constexpr (!__is_inclusive)
        {
            // For an exclusive scan, lane 0's incoming init becomes its result after the final right-shift below,
            // so it must be saved before being overwritten by the broadcast carry.
            if (__sub_group_local_id == 0)
                __old_init.__setup(__init_and_carry.__get_cref());
        }
    }
    __init_and_carry.__assign(__broadcast_sub_group(__ndi, __value, __init_broadcast_id, __comm_tag));

    if constexpr (!__is_inclusive)
    {
        // Shift the inclusive result right by one lane to produce the exclusive scan, then restore the saved init on
        // lane 0.
        __value = __shift_sub_group_right(__ndi, __value, 1, __comm_tag);
        if (__old_init.__has_value())
        {
            if (__sub_group_local_id == 0)
            {
                __value = __old_init.__get_cref();
            }
        }
    }
    //return by reference __value and __init_and_carry
}

template <bool __is_inclusive, typename _BinaryOp, typename _ValueType, typename _ScanOpsTag = __subgroup_only_tag>
void
__sub_group_scan(const sycl::nd_item<1>& __ndi, _ValueType& __value, _BinaryOp __binary_op,
                 oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __init_and_carry,
                 _ScanOpsTag __comm_tag = {})
{
    auto __mask_fn = [](auto __sub_group_local_id, auto __offset) { return __sub_group_local_id >= __offset; };
    std::uint8_t __init_broadcast_id = __get_reduce_then_scan_actual_sub_group_size(__ndi.get_sub_group()) - 1;
    __sub_group_masked_scan<__is_inclusive>(__ndi, __mask_fn, __init_broadcast_id, __value, __binary_op,
                                            __init_and_carry, __comm_tag);
}

template <bool __is_inclusive, typename _BinaryOp, typename _ValueType, typename _SizeType,
          typename _ScanOpsTag = __subgroup_only_tag>
void
__sub_group_scan_partial(const sycl::nd_item<1>& __ndi, _ValueType& __value, _BinaryOp __binary_op,
                         oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __init_and_carry,
                         _SizeType __elements_to_process, _ScanOpsTag __comm_tag = {})
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    std::uint8_t __init_broadcast_id = __elements_to_process - 1;
    __sub_group_masked_scan<__is_inclusive>(__ndi, __mask_fn, __init_broadcast_id, __value, __binary_op,
                                            __init_and_carry, __comm_tag);
}

template <bool __is_inclusive, typename _GenInput, typename _ScanInputTransform, typename _BinaryOp, typename _WriteOp,
          typename _ValueType, typename _InRng, typename _CommTag>
void
__scan_through_elements_helper_impl(const sycl::nd_item<1>& __ndi, _GenInput __gen_input,
                                    _ScanInputTransform __scan_input_transform, _BinaryOp __binary_op,
                                    _WriteOp __write_op,
                                    oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __sub_group_carry,
                                    const _InRng& __in_rng, std::size_t __start_id, std::size_t __n,
                                    std::uint32_t __iters_per_item, std::size_t __subgroup_start_id,
                                    _CommTag __comm_tag)
{
    using _GenInputType = std::invoke_result_t<_GenInput, _InRng, std::size_t>;

    const std::uint8_t __sub_group_size = __get_reduce_then_scan_actual_sub_group_size(__ndi.get_sub_group());

    // For partial thread, we need to handle the partial subgroup at the end of the range
    const std::uint32_t __subgroup_n = static_cast<std::uint32_t>(
        std::min<std::size_t>(__n - __subgroup_start_id, __iters_per_item * __sub_group_size));
    std::uint32_t __iters = oneapi::dpl::__internal::__dpl_ceiling_div(__subgroup_n, __sub_group_size);

    if (__iters > 1)
    {
        // peel first iteration out as workaround for issue set_union.pass and reduce_by_segment.pass
        // with some compilers and environments
        _GenInputType __v = __gen_input(__in_rng, __start_id);
        __sub_group_scan<__is_inclusive>(__ndi, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                                         __comm_tag);
        __write_op(__start_id, __v);

        for (std::uint32_t __j = 1; __j + 1 < __iters; __j++)
        {
            __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size);
            __sub_group_scan<__is_inclusive>(__ndi, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                                             __comm_tag);
            __write_op(__start_id + __j * __sub_group_size, __v);
        }
    }
    std::size_t __offset = __start_id + (__iters - 1) * __sub_group_size;
    std::size_t __local_id = std::min(__offset, __n - 1);
    _GenInputType __v = __gen_input(__in_rng, __local_id);
    std::uint32_t __elements_to_process = static_cast<std::uint32_t>(__subgroup_n - (__iters - 1) * __sub_group_size);
    __sub_group_scan_partial<__is_inclusive>(__ndi, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                                             __elements_to_process, __comm_tag);
    if constexpr (!std::is_same_v<_WriteOp, oneapi::dpl::__internal::__ignore_call_op>)
    {
        if (__offset < __n)
            __write_op(__offset, __v);
    }
}

// Detecting TempData type alias in the specified structure
template <typename, typename = void>
struct __temp_data_required
{
    static constexpr bool value = false;
    using type = __noop_temp_data;
};

template <typename _T>
struct __temp_data_required<_T, std::void_t<typename _T::TempData>>
{
    static constexpr bool value = true;
    using type = typename _T::TempData;
};

template <bool _Bounded, bool __is_inclusive, bool __is_unique_pattern_v, typename _GenInput,
          typename _ScanInputTransform, typename _BinaryOp, typename _WriteOp, typename _ValueType, typename _InRng,
          typename _OutRng, typename _CommTag, typename _OnOOBReached = __internal::__no_callback_tag,
          typename _FinalPosSaver = __internal::__no_callback_tag>
void
__scan_through_elements_helper(const sycl::nd_item<1>& __ndi, _GenInput __gen_input,
                               _ScanInputTransform __scan_input_transform, _BinaryOp __binary_op, _WriteOp __write_op,
                               oneapi::dpl::__internal::__opt_lazy_ctor_storage<_ValueType>& __sub_group_carry,
                               const _InRng& __in_rng, _OutRng& __out_rng, std::size_t __start_id, std::size_t __n,
                               std::uint32_t __iters_per_item, std::size_t __subgroup_start_id, _CommTag __comm_tag,
                               _OnOOBReached __on_oob_reached = {}, _FinalPosSaver __final_pos_saver = {})
{
    using __temp_data_required_t = __temp_data_required<_GenInput>;
    constexpr bool __is_temp_data_required = __temp_data_required_t::value;

    using _TempData = typename __temp_data_required_t::type;
    _TempData __temp_data{};

    auto __gen_input_impl = [&](const _InRng& __rng, std::size_t __id) {
        if constexpr (__is_temp_data_required)
            return __gen_input(__rng, __id, __temp_data, __final_pos_saver);
        else
            return __gen_input(__rng, __id);
    };

    using _OutRngSize = decltype(oneapi::dpl::__ranges::__size(__out_rng));

    // Hoist the sub-group-ops vs SLM-fallback decision to here. The element-scan body below is instantiated
    // once per available communication path; the branch is taken a single time per call to this helper.
    __dispatch_comm_tag(__comm_tag, [&](auto __comm_tag_concrete) {
        if constexpr (std::is_same_v<_WriteOp, oneapi::dpl::__internal::__ignore_call_op>)
        {
            __scan_through_elements_helper_impl<__is_inclusive>(
                __ndi, __gen_input_impl, __scan_input_transform, __binary_op,
                oneapi::dpl::__internal::__ignore_call_op{}, __sub_group_carry, __in_rng, __start_id, __n,
                __iters_per_item, __subgroup_start_id, __comm_tag_concrete);
        }
        else
        {
            if constexpr (_Bounded)
            {
                _OutRngSize __out_rng_size = oneapi::dpl::__ranges::__size(__out_rng);

                const std::size_t __carry_in = __sub_group_carry.__has_value() ? __sub_group_carry.__get_cref() : 0;
                const std::uint8_t __sub_group_size =
                    __get_reduce_then_scan_actual_sub_group_size(__ndi.get_sub_group());
                // A single scanned element may emit up to _TempData::__max_outputs_per_input output elements:
                // one for copy_if/unique, but up to __diagonal_spacing for set operations, where each scanned
                // element is a diagonal written through __write_multiple_to_id. The estimate must account for
                // this many writes per scanned element, otherwise the unchecked write path could be selected for
                // set operations and overrun __out_rng (corrupting memory and skipping OOB position detection).
                const std::size_t __max_writes_this_sub_group =
                    std::size_t{__iters_per_item} * __sub_group_size * _TempData::__max_outputs_per_input;
                if (__carry_in + __max_writes_this_sub_group + __is_unique_pattern_v > __out_rng_size)
                {
                    auto __bounded_write_op = [&](std::size_t __id, const auto& __v) {
                        if constexpr (__is_temp_data_required)
                            __write_op(__out_rng, __out_rng_size, __id, __v, __temp_data, __on_oob_reached);
                        else
                            __write_op(__out_rng, __out_rng_size, __id, __v, __on_oob_reached);
                    };
                    __scan_through_elements_helper_impl<__is_inclusive>(
                        __ndi, __gen_input_impl, __scan_input_transform, __binary_op, __bounded_write_op,
                        __sub_group_carry, __in_rng, __start_id, __n, __iters_per_item, __subgroup_start_id,
                        __comm_tag_concrete);
                    return;
                }
            }

            auto __unbounded_write_op = [&](std::size_t __id, const auto& __v) {
                if constexpr (__is_temp_data_required)
                    __write_op(__out_rng, __id, __v, __temp_data);
                else
                    __write_op(__out_rng, __id, __v);
            };
            __scan_through_elements_helper_impl<__is_inclusive>(
                __ndi, __gen_input_impl, __scan_input_transform, __binary_op, __unbounded_write_op, __sub_group_carry,
                __in_rng, __start_id, __n, __iters_per_item, __subgroup_start_id, __comm_tag_concrete);
        }
    });
}

template <typename _ScanOpsTag, typename _InitValueType>
struct __comm_slm_handler
{
    auto
    __get_accessor_or_placeholder(const std::uint32_t __work_group_size, sycl::handler& __cgh) const
    {
        if (!__use_subgroup_ops)
            return __dpl_sycl::__local_accessor<_InitValueType>(__work_group_size, __cgh);
        else
            return __dpl_sycl::__local_accessor<_InitValueType>(0, __cgh); // Dummy accessor, won't actually be used
    }

    template <typename _Acc>
    _ScanOpsTag
    __get_tag_with_workspace(const _Acc& __comm_slm_acc_opt) const
    {
        if (!__use_subgroup_ops)
            return _ScanOpsTag{__dpl_sycl::__get_accessor_ptr(__comm_slm_acc_opt)};
        else
            return _ScanOpsTag{nullptr};
    }
    bool __use_subgroup_ops = false;
};

template <typename _InitValueType>
struct __comm_slm_handler<__subgroup_only_tag, _InitValueType>
{
    __comm_slm_handler(bool /*__use_subgroup_ops*/) {}

    __subgroup_only_tag
    __get_accessor_or_placeholder(const std::uint32_t /*__work_group_size*/, sycl::handler& /*__cgh*/) const
    {
        return __subgroup_only_tag{};
    }

    __subgroup_only_tag
    __get_tag_with_workspace(__subgroup_only_tag) const
    {
        // Noop, type carries dispatch info for broadcast and shift left subgroup ops
        return __subgroup_only_tag{};
    }
};

template <typename... _Name>
class __reduce_then_scan_partition_kernel;

template <typename... _Name>
class __reduce_then_scan_reduce_kernel;

template <typename... _Name>
class __reduce_then_scan_scan_kernel;

template <bool _Bounded, bool __is_inclusive, bool __is_unique_pattern_v, typename _ScanOpsTag,
          typename _GenReduceInput, typename _ReduceOp, typename _InitType, typename _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter;

template <bool _Bounded, bool __is_inclusive, bool __is_unique_pattern_v, typename _ScanOpsTag,
          typename _GenReduceInput, typename _ReduceOp, typename _InitType, typename... _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter<_Bounded, __is_inclusive, __is_unique_pattern_v, _ScanOpsTag,
                                                    _GenReduceInput, _ReduceOp, _InitType,
                                                    __internal::__optional_kernel_name<_KernelName...>>
{
    // Step 1 - SubGroupReduce is expected to perform sub-group reductions to global memory
    // input buffer
    template <typename _InRng, typename _TmpStorageAcc, typename _StopPosStorage, typename _StopPosInitState>
    sycl::event
    operator()(sycl::queue& __q, const sycl::nd_range<1> __nd_range, _InRng&& __in_rng,
               _TmpStorageAcc& __scratch_container, const sycl::event& __prior_event,
               const std::uint32_t __inputs_per_item, const std::size_t __block_num,
               _StopPosStorage& __stop_pos_storage, _StopPosInitState __stop_pos_initial_state) const
    {
        using _InitValueType = typename _InitType::__value_type;
        const std::uint32_t __inputs_per_work_group = __inputs_per_item * __work_group_size;
        return __q.submit([&, this](sycl::handler& __cgh) {
            __dpl_sycl::__local_accessor<_InitValueType> __sub_group_partials(__max_num_sub_groups_local, __cgh);
            // SLM for sub-group communication (shift_group_right / group_broadcast).
            // Used for non-trivially-copyable types or when SLM communication is preferred (e.g., CPU targets).
            __comm_slm_handler<_ScanOpsTag, _InitValueType> __comm_handler{__use_subgroup_ops};
            auto __comm_acc_or_placeholder = __comm_handler.__get_accessor_or_placeholder(__work_group_size, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng);
            auto __temp_acc = __get_accessor(sycl::write_only, __scratch_container, __cgh, __dpl_sycl::__no_init{});
            auto __stop_pos_acc =
                __internal::__get_stop_pos_accessor_opt<_Bounded>(sycl::write_only, __cgh, __stop_pos_storage);
            __cgh.parallel_for<_KernelName...>(__nd_range, [=, *this](sycl::nd_item<1> __ndi)
                    [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(__get_reduce_then_scan_req_sg_sz_device())]] {
                const __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
                const std::uint8_t __sub_group_size = __get_reduce_then_scan_actual_sub_group_size(__sub_group);

                _InitValueType* __temp_ptr = __temp_acc.__data();
                // The sub-group-ops vs SLM-fallback decision is dispatched at each sub-group-scan region
                // (see __scan_through_elements_helper and the carry-computation block below).
                const _ScanOpsTag __comm_scan_tag = __comm_handler.__get_tag_with_workspace(__comm_acc_or_placeholder);
                std::size_t __group_id = __ndi.get_group(0);
                std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
                std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

                std::size_t __group_start_id =
                    (__block_num * __max_block_size) + (__group_id * __inputs_per_work_group);
                if constexpr (__is_unique_pattern_v)
                {
                    // for unique patterns, the first element is always copied to the output, so we need to skip it
                    __group_start_id += 1;
                }

                std::uint32_t __active_subgroups = __count_active_sub_groups(
                    __ndi, __n - __group_start_id, __inputs_per_item, __inputs_per_work_group);
                std::size_t __subgroup_start_id =
                    __group_start_id + (std::size_t{__get_sub_group_base(__ndi)} * __inputs_per_item);

                std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

                if (__sub_group_id < __active_subgroups)
                {
                    oneapi::dpl::__internal::__opt_lazy_ctor_storage<_InitValueType> __sub_group_carry;
                    // adjust for lane-id
                    // compute sub-group local prefix on T0..63, K samples/T, send to accumulator kernel
                    __scan_through_elements_helper</*_Bounded*/ false, __is_inclusive, __is_unique_pattern_v>(
                        __ndi, __gen_reduce_input, oneapi::dpl::identity{}, __reduce_op,
                        oneapi::dpl::__internal::__ignore_call_op{}, __sub_group_carry, __in_rng, /*unused*/ __in_rng,
                        __start_id, __n, __inputs_per_item, __subgroup_start_id, __comm_scan_tag);
                    if (__sub_group_local_id == 0)
                        __sub_group_partials[__sub_group_id] = __sub_group_carry.__get_cref();
                }
                sycl::group_barrier(__ndi.get_group());

                // compute sub-group local prefix sums on (T0..63) carries
                // and store to scratch space at the end of dst; next
                // accumulator kernel takes M thread carries from scratch
                // to compute a prefix sum on global carries
                if (__sub_group_id == 0)
                {
                    oneapi::dpl::__internal::__opt_lazy_ctor_storage<_InitValueType> __summary_carry;
                    // Each group's region in the global temp is strided by __max_num_sub_groups_local (the allocated
                    // upper bound on sub-group count), not the actual count, so that regions of distinct groups never
                    // overlap regardless of how many sub-groups a given group actually has.
                    __start_id = __group_id * __max_num_sub_groups_local;
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    // Dispatch the sub-group-ops vs SLM-fallback decision once for this carry-computation region.
                    __dispatch_comm_tag(__comm_scan_tag, [&](auto __comm_tag_concrete) {
                        std::uint32_t __reduction_scan_id = __sub_group_local_id;
                        if (__iters > 1)
                        {
                            // peel first iteration out as workaround for issue set_union.pass and
                            // reduce_by_segment.pass with some compilers and environments
                            _InitValueType __v = __sub_group_partials[__reduction_scan_id];
                            __sub_group_scan</*__is_inclusive=*/true>(__ndi, __v, __reduce_op, __summary_carry,
                                                                      __comm_tag_concrete);
                            __temp_ptr[__start_id + __reduction_scan_id] = __v;
                            __reduction_scan_id += __sub_group_size;

                            for (std::uint32_t __i = 1; __i < __iters - 1; __i++)
                            {
                                __v = __sub_group_partials[__reduction_scan_id];
                                __sub_group_scan</*__is_inclusive=*/true>(__ndi, __v, __reduce_op, __summary_carry,
                                                                          __comm_tag_concrete);
                                __temp_ptr[__start_id + __reduction_scan_id] = __v;
                                __reduction_scan_id += __sub_group_size;
                            }
                        }
                        // If we are past the input range, then the previous value of v is passed to the sub-group scan.
                        // It does not affect the result as our sub_group_scan will use a mask to only process in-range elements.

                        // fill with unused dummy values to avoid overrunning input
                        std::uint32_t __load_id = std::min(__reduction_scan_id, __max_num_sub_groups_local - 1);

                        _InitValueType __v = __sub_group_partials[__load_id];
                        __sub_group_scan_partial</*__is_inclusive=*/true>(
                            __ndi, __v, __reduce_op, __summary_carry,
                            __active_subgroups - ((__iters - 1) * __sub_group_size), __comm_tag_concrete);
                        if (__reduction_scan_id < __max_num_sub_groups_local)
                            __temp_ptr[__start_id + __reduction_scan_id] = __v;
                    });
                    // Write this group's TOTAL carry-out to a CANONICAL slot -- the last slot of the group's
                    // max-strided region -- independent of the actual sub-group count. A later group's cross-group
                    // gather (scan kernel step 2) reads totals at stride=max, offset=max-1 without needing to know any
                    // per-group count. __summary_carry holds the inclusive total of all active partials,
                    // broadcast to every lane, so lane 0 may write it. When __active_subgroups == __max_num_sub_groups_local
                    // this slot already holds the same value from the per-sub-group writes above.
                    if (__sub_group_local_id == 0)
                        __temp_ptr[__start_id + (__max_num_sub_groups_local - 1)] = __summary_carry.__get_cref();
                }

                if constexpr (!std::is_same_v<std::remove_cv_t<decltype(__stop_pos_acc)>,
                                              __internal::__no_stop_pos_acc_tag>)
                {
                    if (__block_num == 0 && __ndi.get_global_linear_id() == 0)
                    {
                        __stop_pos_acc.__data()[0] = __stop_pos_initial_state;
                    }
                }
            });
        });
    }

    // Constant parameters throughout all blocks
    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::size_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::size_t __n;

    const _GenReduceInput __gen_reduce_input;
    const _ReduceOp __reduce_op;
    _InitType __init;
    const bool __use_subgroup_ops;
};

template <bool _Bounded, typename _ValueType, typename _StopPosType>
using __transform_reduce_then_scan_result_t =
    std::conditional_t<_Bounded,
                       std::tuple<sycl::event, __combined_storage<_ValueType>, __result_storage<_StopPosType>>,
                       std::tuple<sycl::event, __combined_storage<_ValueType>>>;

template <bool _Bounded, bool __is_inclusive, bool __is_unique_pattern_v, typename _ScanOpsTag, typename _ReduceOp,
          typename _GenScanInput, typename _ScanInputTransform, typename _WriteOp, typename _InitType,
          typename _TransformResult, typename _KernelName>
struct __parallel_reduce_then_scan_scan_submitter;

template <bool _Bounded, bool __is_inclusive, bool __is_unique_pattern_v, typename _ScanOpsTag, typename _ReduceOp,
          typename _GenScanInput, typename _ScanInputTransform, typename _WriteOp, typename _InitType,
          typename _TransformResult, typename... _KernelName>
struct __parallel_reduce_then_scan_scan_submitter<_Bounded, __is_inclusive, __is_unique_pattern_v, _ScanOpsTag,
                                                  _ReduceOp, _GenScanInput, _ScanInputTransform, _WriteOp, _InitType,
                                                  _TransformResult, __internal::__optional_kernel_name<_KernelName...>>
{
    using _InitValueType = typename _InitType::__value_type;

    template <typename _TmpAcc>
    _InitValueType
    __get_block_carry_in(const std::size_t __block_num, _TmpAcc __tmp_acc,
                         const std::size_t __num_sub_groups_global) const
    {
        return __tmp_acc[__num_sub_groups_global + (__block_num % 2)];
    }

    template <typename _TmpAcc, typename _ValueType>
    void
    __set_block_carry_out(const std::size_t __block_num, _TmpAcc __tmp_acc, const _ValueType __block_carry_out,
                          const std::size_t __num_sub_groups_global) const
    {
        __tmp_acc[__num_sub_groups_global + 1 - (__block_num % 2)] = __block_carry_out;
    }

    template <typename _InRng, typename _OutRng, typename _TmpStorageAcc, typename _StopPosStorage>
    sycl::event
    operator()(sycl::queue& __q, const sycl::nd_range<1> __nd_range, _InRng&& __in_rng, _OutRng&& __out_rng,
               _TmpStorageAcc& __scratch_container, const sycl::event& __prior_event,
               const std::uint32_t __inputs_per_item, const std::size_t __block_num,
               _StopPosStorage& __stop_pos_storage) const
    {
        // Size-independent, host-computed inputs-per-work-item; see the reduce submitter for why it is passed in
        // rather than re-derived on the device.
        const std::uint32_t __inputs_per_work_group = __inputs_per_item * __work_group_size;
        std::size_t __num_remaining = __n - __block_num * __max_block_size;
        // for unique patterns, the first element is always copied to the output, so we need to skip it
        if constexpr (__is_unique_pattern_v)
        {
            assert(__num_remaining > 0);
            __num_remaining -= 1;
        }
        std::uint32_t __inputs_in_block = std::min(__num_remaining, __max_block_size);
        return __q.submit([&, this](sycl::handler& __cgh) {
            // We need __num_sub_groups_local + 1 temporary SLM locations to store intermediate results:
            //   __num_sub_groups_local for each sub-group partial from the reduce kernel +
            //   1 element for the accumulated block-local carry-in from previous groups in the block
            __dpl_sycl::__local_accessor<_InitValueType> __sub_group_partials(__max_num_sub_groups_local + 1, __cgh);
            // SLM for sub-group communication (shift_group_right / group_broadcast).
            // Used for non-trivially-copyable types or when SLM communication is preferred (e.g., CPU targets).
            __comm_slm_handler<_ScanOpsTag, _InitValueType> __comm_handler{__use_subgroup_ops};
            auto __comm_acc_or_placeholder = __comm_handler.__get_accessor_or_placeholder(__work_group_size, __cgh);

            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng, __out_rng);
            auto __temp_acc = __get_accessor(sycl::read_write, __scratch_container, __cgh);
            auto __res_acc =
                __get_result_accessor(sycl::write_only, __scratch_container, __cgh, __dpl_sycl::__no_init{});
            auto __stop_pos_acc =
                __internal::__get_stop_pos_accessor_opt<_Bounded>(sycl::read_write, __cgh, __stop_pos_storage);

            __cgh.parallel_for<_KernelName...>(__nd_range, [=, *this](sycl::nd_item<1> __ndi)
                    [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(__get_reduce_then_scan_req_sg_sz_device())]] {
                _ScanOpsTag __comm_scan_tag = __comm_handler.__get_tag_with_workspace(__comm_acc_or_placeholder);
                const __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
                const std::uint8_t __sub_group_size = __get_reduce_then_scan_actual_sub_group_size(__sub_group);

                const std::uint32_t __active_groups =
                    oneapi::dpl::__internal::__dpl_ceiling_div(__inputs_in_block, __inputs_per_work_group);

                _InitValueType* __tmp_ptr = __temp_acc.__data();
                _InitValueType* __res_ptr = __res_acc.__data();
                std::uint32_t __group_id = __ndi.get_group(0);
                std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
                std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

                std::size_t __group_start_id =
                    (__block_num * __max_block_size) + (__group_id * __inputs_per_work_group);
                if constexpr (__is_unique_pattern_v)
                {
                    // for unique patterns, the first element is always copied to the output, so we need to skip it
                    __group_start_id += 1;
                }
                std::uint32_t __active_subgroups = __count_active_sub_groups(
                    __ndi, __n - __group_start_id, __inputs_per_item, __inputs_per_work_group);
                oneapi::dpl::__internal::__opt_lazy_ctor_storage<_InitValueType> __carry_last;

                // propagate carry in from previous block
                oneapi::dpl::__internal::__opt_lazy_ctor_storage<_InitValueType> __sub_group_carry;

                // on the first sub-group in a work-group (assuming S subgroups in a work-group):
                // 1. load S sub-group local carry prefix sums (T0..TS-1) to SLM
                // 2. load 32, 64, 96, etc. TS-1 work-group carry-outs (32 for WG num<32, 64 for WG num<64, etc.),
                //    and then compute the prefix sum to generate global carry out
                //    for each WG, i.e., prefix sum on TS-1 carries over all WG.
                // 3. on each WG select the adjacent neighboring WG carry in
                // 4. on each WG add the global carry-in to S sub-group local prefix sums to
                //    get a T-local global carry in
                // 5. recompute T-local prefix values, add the T-local global carries,
                //    and then write back the final values to memory
                if (__sub_group_id == 0)
                {
                    // step 1) load to SLM the WG-local S prefix sums
                    //         on WG T-local carries
                    //            0: T0 carry, 1: T0 + T1 carry, 2: T0 + T1 + T2 carry, ...
                    //           S: sum(T0 carry...TS carry)
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    // This group's own per-sub-group partials occupy the max-strided region [g*max, g*max+active).
                    // The reduce kernel wrote them at g*max + sg_id (both kernels share the per-group sub-group layout),
                    // so we read with the same max stride.
                    std::size_t __subgroups_before_my_group = __group_id * __max_num_sub_groups_local;
                    std::uint32_t __load_reduction_id = __sub_group_local_id;
                    std::uint8_t __i = 0;
                    for (; __i < __iters - 1; __i++)
                    {
                        __sub_group_partials[__load_reduction_id] =
                            __tmp_ptr[__subgroups_before_my_group + __load_reduction_id];
                        __load_reduction_id += __sub_group_size;
                    }
                    if (__load_reduction_id < __active_subgroups)
                    {
                        __sub_group_partials[__load_reduction_id] =
                            __tmp_ptr[__subgroups_before_my_group + __load_reduction_id];
                    }

                    // step 2) load 32, 64, 96, etc. work-group carry outs on every work-group; then
                    //         compute the prefix in a sub-group to get global work-group carries
                    //         memory accesses: gather(63, 127, 191, 255, ...)
                    // Each preceding group's TOTAL carry-out lives at the CANONICAL last slot of its max-strided
                    // region (g*max + max-1). Gathering at stride=max, offset=max-1 needs no knowledge of any
                    // group's actual sub-group count, which can vary under non-uniform sub-group sizes.
                    std::uint32_t __offset = __max_num_sub_groups_local - 1;
                    // only need 32 carries for WGs0..WG32, 64 for WGs32..WGs64, etc.
                    if (__group_id > 0)
                    {
                        // one canonical total per preceding group
                        const std::size_t __elements_to_process = __group_id;
                        const std::size_t __pre_carry_iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__elements_to_process, __sub_group_size);
                        // Dispatch the sub-group-ops vs SLM-fallback decision once for this carry-computation region.
                        __dispatch_comm_tag(__comm_scan_tag, [&](auto __comm_tag_concrete) {
                            std::uint32_t __reduction_id = __max_num_sub_groups_local * __sub_group_local_id + __offset;
                            if (__pre_carry_iters > 1)
                            {
                                // peel first iteration out as workaround for issue set_union.pass and
                                // reduce_by_segment.pass with some compilers and environments
                                std::uint32_t __reduction_id_increment = __max_num_sub_groups_local * __sub_group_size;
                                _InitValueType __value = __tmp_ptr[__reduction_id];
                                __sub_group_scan</*__is_inclusive=*/true>(__ndi, __value, __reduce_op, __carry_last,
                                                                          __comm_tag_concrete);
                                __reduction_id += __reduction_id_increment;
                                // then some number of full iterations
                                for (std::uint32_t __i = 1; __i < __pre_carry_iters - 1; __i++)
                                {
                                    __value = __tmp_ptr[__reduction_id];
                                    __sub_group_scan</*__is_inclusive=*/true>(__ndi, __value, __reduce_op, __carry_last,
                                                                              __comm_tag_concrete);
                                    __reduction_id += __reduction_id_increment;
                                }
                            }
                            // final partial iteration

                            std::size_t __remaining_elements =
                                __elements_to_process - ((__pre_carry_iters - 1) * __sub_group_size);
                            // fill with unused dummy values to avoid overrunning input
                            std::size_t __final_reduction_id =
                                std::min(std::size_t{__reduction_id}, __subgroups_before_my_group - 1);
                            _InitValueType __value = __tmp_ptr[__final_reduction_id];
                            __sub_group_scan_partial</*__is_inclusive=*/true>(
                                __ndi, __value, __reduce_op, __carry_last, __remaining_elements, __comm_tag_concrete);
                        });

                        // steps 3+4) load global carry in from neighbor work-group
                        //            and apply to local sub-group prefix carries
                        std::size_t __carry_offset = __sub_group_local_id;

                        std::uint8_t __iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);

                        std::uint8_t __i = 0;
                        for (; __i < __iters - 1; ++__i)
                        {
                            __sub_group_partials[__carry_offset] =
                                __reduce_op(__carry_last.__get_cref(), __sub_group_partials[__carry_offset]);
                            __carry_offset += __sub_group_size;
                        }
                        if (__i * __sub_group_size + __sub_group_local_id < __active_subgroups)
                        {
                            __sub_group_partials[__carry_offset] =
                                __reduce_op(__carry_last.__get_cref(), __sub_group_partials[__carry_offset]);
                            __carry_offset += __sub_group_size;
                        }
                        if (__sub_group_local_id == 0)
                            __sub_group_partials[__active_subgroups] = __carry_last.__get_cref();
                    }
                }

                sycl::group_barrier(__ndi.get_group());

                // Get inter-work group and adjusted for intra-work group prefix
                if (__block_num == 0)
                {
                    if (__sub_group_id > 0)
                    {
                        _InitValueType __value =
                            __sub_group_partials[std::min(__sub_group_id - 1, __active_subgroups - 1)];
                        oneapi::dpl::unseq_backend::__init_processing<_InitValueType>{}(__init, __value, __reduce_op);
                        __sub_group_carry.__setup(__value);
                    }
                    else if (__group_id > 0)
                    {
                        _InitValueType __value = __sub_group_partials[__active_subgroups];
                        oneapi::dpl::unseq_backend::__init_processing<_InitValueType>{}(__init, __value, __reduce_op);
                        __sub_group_carry.__setup(__value);
                    }
                    else // zeroth block, group and subgroup
                    {
                        if constexpr (__is_unique_pattern_v)
                        {
                            if (__sub_group_local_id == 0)
                            {
                                // For unique patterns, always copy the 0th element to the output
                                __write_op.__assign(__in_rng[0], __out_rng[0]);
                            }
                        }

                        if constexpr (!std::is_same_v<_InitType,
                                                      oneapi::dpl::unseq_backend::__no_init_value<_InitValueType>>)
                        {
                            __sub_group_carry.__setup(__init.__value);
                        }
                    }
                }
                else
                {
                    if (__sub_group_id > 0)
                    {
                        _InitValueType __value =
                            __sub_group_partials[std::min(__sub_group_id - 1, __active_subgroups - 1)];
                        __sub_group_carry.__setup(__reduce_op(
                            __get_block_carry_in(__block_num, __tmp_ptr, __max_num_sub_groups_global), __value));
                    }
                    else if (__group_id > 0)
                    {
                        __sub_group_carry.__setup(
                            __reduce_op(__get_block_carry_in(__block_num, __tmp_ptr, __max_num_sub_groups_global),
                                        __sub_group_partials[__active_subgroups]));
                    }
                    else
                    {
                        __sub_group_carry.__setup(
                            __get_block_carry_in(__block_num, __tmp_ptr, __max_num_sub_groups_global));
                    }
                }

                // step 5) apply global carries
                // Cumulative element offset of all preceding sub-groups, valid for any (even non-uniform) sub-group
                // layout: __sg_base (work-group-local id of this sub-group's first work-item) times the
                // size-independent __inputs_per_item. See the reduce kernel for the same derivation.
                if (__sub_group_id < __active_subgroups)
                {
                    const std::size_t __subgroup_start_id =
                        __group_start_id + (std::size_t{__get_sub_group_base(__ndi)} * __inputs_per_item);
                    std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

                    using _PosTools =
                        __internal::__parallel_reduce_then_scan_stop_oob_pos_tools<_Bounded, _GenScanInput,
                                                                                   _StopPosStorage>;
                    if constexpr (_Bounded)
                    {
                        using __src_final_pos_t = typename _PosTools::__src_final_pos_t;

                        // Two pass processing: if the OOB position is reached in the first pass, then on the second
                        // pass we recover the source indexes for the diagonal where it happened and store the OOB
                        // position from them. The OOB position may be reached only in one work-item, so no
                        // synchronization is needed to update the shared OOB position in the second pass.
                        std::size_t __start_id_reached_on_oob = __start_id;
                        typename _PosTools::__oob_pos_t __oob_position = _PosTools::__initial_oob_pos();

                        auto __on_oob_reached = [&](std::size_t __start_id, typename _PosTools::__oob_pos_t __id) {
                            __start_id_reached_on_oob = __start_id;
                            __oob_position = __id;
                        };

                        if constexpr (_PosTools::__has_src_final_pos)
                        {
                            __scan_through_elements_helper<_Bounded, __is_inclusive, __is_unique_pattern_v>(
                                __ndi, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op,
                                __sub_group_carry, __in_rng, __out_rng, __start_id, __n, __inputs_per_item,
                                __subgroup_start_id, __comm_scan_tag, __on_oob_reached,
                                [&](__src_final_pos_t __final_pos) {
                                    // Exactly one work-item reaches the edge crossing, so no synchronization
                                    // is needed to store the shared final position.
                                    _PosTools::__store_final_pos(__stop_pos_acc, __final_pos);
                                });
                        }
                        else
                        {
                            __scan_through_elements_helper<_Bounded, __is_inclusive, __is_unique_pattern_v>(
                                __ndi, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op,
                                __sub_group_carry, __in_rng, __out_rng, __start_id, __n, __inputs_per_item,
                                __subgroup_start_id, __comm_scan_tag, __on_oob_reached,
                                __internal::__no_callback_tag{});
                        }

                        _PosTools::__finalize_and_store_oob_pos(__in_rng, __oob_position, __start_id_reached_on_oob,
                                                                __gen_scan_input, __stop_pos_acc);
                    }
                    else
                    {
                        __scan_through_elements_helper<_Bounded, __is_inclusive, __is_unique_pattern_v>(
                            __ndi, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op, __sub_group_carry,
                            __in_rng, __out_rng, __start_id, __n, __inputs_per_item, __subgroup_start_id,
                            __comm_scan_tag, __internal::__no_callback_tag{}, __internal::__no_callback_tag{});
                    }
                }
                // If within the last active group and sub-group of the block, use the 0th work-item of the sub-group
                // to write out the last carry out for either the return value or the next block
                if (__sub_group_local_id == 0 && (__active_groups == __group_id + 1) &&
                    (__active_subgroups == __sub_group_id + 1))
                {
                    if (__block_num + 1 == __num_blocks)
                    {
                        if constexpr (__is_unique_pattern_v)
                        {
                            // unique patterns automatically copy the 0th element and scan starting at index 1
                            __res_ptr[0] = __transform_result(__sub_group_carry.__get_cref() + 1);
                        }
                        else
                        {
                            __res_ptr[0] = __transform_result(__sub_group_carry.__get_cref());
                        }
                    }
                    else
                    {
                        // capture the last carry out for the next block
                        __set_block_carry_out(__block_num, __tmp_ptr, __sub_group_carry.__get_cref(),
                                              __max_num_sub_groups_global);
                    }
                }
            });
        });
    }

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::size_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::uint32_t __max_num_sub_groups_global;
    const std::size_t __num_blocks;
    const std::size_t __n;

    const _ReduceOp __reduce_op;
    const _GenScanInput __gen_scan_input;
    const _ScanInputTransform __scan_input_transform;
    const _WriteOp __write_op;
    _InitType __init;
    const bool __use_subgroup_ops;
    const _TransformResult __transform_result;
};

// Helper for __parallel_transform_reduce_then_scan templated on the choice of sub-group communication
// strategy via _ScanOpsTag, which selects which communication path(s) are compiled into the kernel. The
// runtime __use_subgroup_ops flag then chooses between them when both are available.
template <bool _Bounded, typename _ScanOpsTag, std::uint32_t __bytes_per_work_item_iter, typename _CustomName,
          typename _InRng, typename _OutRng, typename _GenReduceInput, typename _ReduceOp, typename _GenScanInput,
          typename _ScanInputTransform, typename _WriteOp, typename _InitType, typename _Inclusive,
          typename _IsUniquePattern, typename _TransformResult, typename _StopPosInitState>
__transform_reduce_then_scan_result_t<_Bounded, typename _InitType::__value_type, _StopPosInitState>
__parallel_transform_reduce_then_scan_impl(sycl::queue& __q, const std::size_t __n, _InRng&& __in_rng,
                                           _OutRng&& __out_rng, _GenReduceInput __gen_reduce_input,
                                           _ReduceOp __reduce_op, _GenScanInput __gen_scan_input,
                                           _ScanInputTransform __scan_input_transform, _WriteOp __write_op,
                                           _InitType __init, _Inclusive, _IsUniquePattern, bool __use_subgroup_ops,
                                           _TransformResult __transform_result,
                                           _StopPosInitState __stop_pos_initial_state, sycl::event __prior_event)
{
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_reduce_kernel<_ScanOpsTag, _CustomName>>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_scan_kernel<_ScanOpsTag, _CustomName>>;
    using _ValueType = typename _InitType::__value_type;

    // Query the device's supported sub-group sizes to allocate storage conservatively and round
    // the work-group size appropriately. The actual sub-group size used by each kernel is determined
    // on the device (see __get_reduce_then_scan_actual_sub_group_size): a compile-time constant on the SPIR-V path and
    // the actual per-sub-group get_local_range() otherwise.
    const auto __supported_sg_sizes = __q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    const auto [__min_sg_it, __max_sg_it] =
        std::minmax_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());
    const std::uint8_t __min_sub_group_size = *__min_sg_it;
    const std::uint8_t __max_sub_group_size = *__max_sg_it;
    // Empirically determined maximum. May be less for non-full blocks.
    const bool __target_is_gpu = __q.get_device().is_gpu();
    constexpr bool __inclusive = _Inclusive::value;
    constexpr bool __is_unique_pattern_v = _IsUniquePattern::value;
    // empirical derived caps for workgroup size based upon target
    const std::uint32_t __wg_size_cap = __target_is_gpu ? 1024 : 128;
    const std::uint32_t __max_work_group_size = oneapi::dpl::__internal::__max_work_group_size(__q, __wg_size_cap);
    // Round down to nearest multiple of the max subgroup size to ensure compatibility with all sub-group sizes
    const std::uint32_t __work_group_size = (__max_work_group_size / __max_sub_group_size) * __max_sub_group_size;

    // use work groups to match the number of compute units
    const std::uint32_t __max_compute_units =
        __q.get_device().template get_info<sycl::info::device::max_compute_units>();

    std::size_t __inputs_remaining = __n;
    if constexpr (__is_unique_pattern_v)
    {
        // skip scan of zeroth element in unique patterns
        __inputs_remaining -= 1;
    }

    std::uint32_t __num_work_groups = 0;
    std::uint32_t __max_inputs_per_item = 0;

    std::size_t __last_level_cache_size_bytes =
        __q.get_device().template get_info<sycl::info::device::global_mem_cache_size>();
    if (__last_level_cache_size_bytes == 0)
    {
        // If the device doesn't report a cache size, assume 32K per CU, likely older device if not reporting LLC size
        __last_level_cache_size_bytes = std::size_t{32} * 1024 * __max_compute_units;
    }
    const std::size_t __target_last_level_cache_size_bytes = __last_level_cache_size_bytes / 2;

    if (__target_is_gpu)
    {
        std::size_t __cache_target_num_blocks = 0;
        // for intel hardware there are 8 compute units per Xe core
        const std::uint32_t __num_xe_cores = std::max(1u, __max_compute_units / 8);

        if (__num_xe_cores * __work_group_size * __bytes_per_work_item_iter > __last_level_cache_size_bytes)
        {
            // if we can't avoid spilling from LLC, use a large block and 2 work groups per core
            __num_work_groups = __num_xe_cores * 2;
            __cache_target_num_blocks = 1;
        }
        else
        {
            __cache_target_num_blocks = oneapi::dpl::__internal::__dpl_ceiling_div(
                __inputs_remaining * __bytes_per_work_item_iter, __target_last_level_cache_size_bytes);
            if (__last_level_cache_size_bytes < 2 * __num_xe_cores * __work_group_size * __bytes_per_work_item_iter)
            {
                // if we can avoid spilling from LLC by only launching 1 work group per core, do that
                __num_work_groups = __num_xe_cores;
            }
            else
            {
                // otherwise, we can launch 2 work groups per core and still avoid spilling from LLC, try to balance
                // the block sizes, and maximize the number of inputs per work item while fitting in a ratio of the cache
                __num_work_groups = __num_xe_cores * 2;
            }
        }
        std::uint32_t __inputs_per_item_limit = std::numeric_limits<std::uint32_t>::max() / (__work_group_size * 2);

        __max_inputs_per_item = std::max<std::uint32_t>(
            1, std::min<std::uint32_t>(
                   __inputs_per_item_limit,
                   oneapi::dpl::__internal::__dpl_ceiling_div(
                       __inputs_remaining, __cache_target_num_blocks * __num_work_groups * __work_group_size)));
    }
    else // target is cpu
    {
        // use a large multiple of the number of cores (each WG is executed by a single thread)
        __num_work_groups = oneapi::dpl::__internal::__dpl_bit_ceil(__max_compute_units * 64);
        // use a large number of inputs per item to amortize the overhead
        __max_inputs_per_item = std::max<std::uint32_t>(1, 2048u / __bytes_per_work_item_iter);
    }

    // Need to calculate actual number of blocks to avoid empty blocks due to floor calculations
    const std::size_t __num_blocks = oneapi::dpl::__internal::__dpl_ceiling_div(
        __inputs_remaining, std::size_t{__max_inputs_per_item} * __work_group_size * __num_work_groups);

    // Allocate sufficient temporary storage for the worst case (smallest sub-group size = most sub-groups).
    const std::uint32_t __max_num_sub_groups_local =
        oneapi::dpl::__internal::__dpl_ceiling_div(__work_group_size, __min_sub_group_size);
    const std::uint32_t __max_num_sub_groups_global = __max_num_sub_groups_local * __num_work_groups;
    // reduce_then_scan kernel is not built to handle "empty" scans which includes `__n == 1` for unique patterns.
    // These trivial end cases should be handled at a higher level.
    assert(__inputs_remaining > 0);
    const std::size_t __work_items_per_block = std::size_t{__num_work_groups} * __work_group_size;
    const std::size_t __max_inputs_per_block = __work_items_per_block * __max_inputs_per_item;
    // Determine the fewest blocks that keep every block within the hardware maximum, then balance the inputs evenly
    // across them. Avoids the last block being much smaller than the others, resulting in a loss of performance.
    std::uint32_t __inputs_per_item =
        __inputs_remaining >= __max_inputs_per_block
            ? __max_inputs_per_item
            : oneapi::dpl::__internal::__dpl_ceiling_div(oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                                                         __work_items_per_block);
    const std::size_t __block_size = std::min(__inputs_remaining, __max_inputs_per_block);

    // We need temporary storage for reductions of each sub-group (__num_sub_groups_global).
    // Additionally, we need two elements for the block carry-out to prevent a race condition
    // between reading and writing the block carry-out within a single kernel.
    __combined_storage<_ValueType> __result_and_scratch{__q, __max_num_sub_groups_global + 2, 1};

    // Reduce and scan step implementations
    using _ReduceSubmitter =
        __parallel_reduce_then_scan_reduce_submitter<_Bounded, __inclusive, __is_unique_pattern_v, _ScanOpsTag,
                                                     _GenReduceInput, _ReduceOp, _InitType, _ReduceKernel>;
    using _ScanSubmitter =
        __parallel_reduce_then_scan_scan_submitter<_Bounded, __inclusive, __is_unique_pattern_v, _ScanOpsTag, _ReduceOp,
                                                   _GenScanInput, _ScanInputTransform, _WriteOp, _InitType,
                                                   _TransformResult, _ScanKernel>;
    _ReduceSubmitter __reduce_submitter{__num_work_groups,
                                        __work_group_size,
                                        __max_inputs_per_block,
                                        __max_num_sub_groups_local,
                                        __n,
                                        __gen_reduce_input,
                                        __reduce_op,
                                        __init,
                                        __use_subgroup_ops};
    _ScanSubmitter __scan_submitter{__num_work_groups,
                                    __work_group_size,
                                    __max_inputs_per_block,
                                    __max_num_sub_groups_local,
                                    __max_num_sub_groups_global,
                                    __num_blocks,
                                    __n,
                                    __reduce_op,
                                    __gen_scan_input,
                                    __scan_input_transform,
                                    __write_op,
                                    __init,
                                    __use_subgroup_ops,
                                    __transform_result};

    // Allocate storage for stop pos and out-of-bounds position if needed
    auto __create_stop_pos_storage_opt = [](sycl::queue& __q) {
        if constexpr (_Bounded)
            return __result_storage<_StopPosInitState>(__q, 1);
        else
            return __internal::__no_stop_pos_acc_tag{};
    };
    auto __stop_pos_storage = __create_stop_pos_storage_opt(__q);

    // Data is processed in 2-kernel blocks to allow contiguous input segment to persist in LLC between the first and second kernel for accelerators
    // with sufficiently large L2 / L3 caches.
    for (std::size_t __b = 0; __b < __num_blocks; ++__b)
    {
        std::uint32_t __workitems_in_block = oneapi::dpl::__internal::__dpl_ceiling_div(
            std::min(__inputs_remaining, std::size_t{__max_inputs_per_block}), __inputs_per_item);
        std::uint32_t __workitems_in_block_round_up_workgroup =
            oneapi::dpl::__internal::__dpl_ceiling_div(__workitems_in_block, __work_group_size) * __work_group_size;
        auto __global_range = sycl::range<1>(__workitems_in_block_round_up_workgroup);
        auto __local_range = sycl::range<1>(__work_group_size);
        auto __kernel_nd_range = sycl::nd_range<1>(__global_range, __local_range);
        // 1. Reduce step - Reduce assigned input per sub-group, compute and apply intra-wg carries, and write to global memory.
        __prior_event = __reduce_submitter(__q, __kernel_nd_range, __in_rng, __result_and_scratch, __prior_event,
                                           __inputs_per_item, __b, __stop_pos_storage, __stop_pos_initial_state);
        // 2. Scan step - Compute intra-wg carries, determine sub-group carry-ins, and perform full input block scan.
        __prior_event = __scan_submitter(__q, __kernel_nd_range, __in_rng, __out_rng, __result_and_scratch,
                                         __prior_event, __inputs_per_item, __b, __stop_pos_storage);
        __inputs_remaining -= std::min(__inputs_remaining, __block_size);
        if (__b + 2 == __num_blocks)
        {
            __inputs_per_item =
                __inputs_remaining >= __max_inputs_per_block
                    ? __max_inputs_per_item
                    : oneapi::dpl::__internal::__dpl_ceiling_div(
                          oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining), __work_items_per_block);
        }
    }

    if constexpr (_Bounded)
        return {std::move(__prior_event), std::move(__result_and_scratch), std::move(__stop_pos_storage)};
    else
        return {std::move(__prior_event), std::move(__result_and_scratch)};
}

// General scan-like algorithm helpers
// _GenReduceInput - a function which accepts the input range and index to generate the data needed by the main output
//                   used in the reduction operation (to calculate the global carries)
// _GenScanInput - a function which accepts the input range and index to generate the data needed by the final scan
//                 and write operations, for scan patterns
// _ScanInputTransform - a unary function applied to the output of `_GenScanInput` to extract the component used in the
//             scan, but not the part only required for the final write operation
// _ReduceOp - a binary function which is used in the reduction and scan operations
// _WriteOp - a function which accepts output range, index, and output of `_GenScanInput` applied to the input range
//            and performs the final write to output operation
template <bool _Bounded, std::uint32_t __bytes_per_work_item_iter, typename _CustomName, typename _InRng,
          typename _OutRng, typename _GenReduceInput, typename _ReduceOp, typename _GenScanInput,
          typename _ScanInputTransform, typename _WriteOp, typename _InitType, typename _Inclusive,
          typename _IsUniquePattern, typename _StopPosInitState = oneapi::dpl::__internal::__difference_t<_InRng>,
          typename _TransformResult = oneapi::dpl::identity>
__transform_reduce_then_scan_result_t<_Bounded, typename _InitType::__value_type, _StopPosInitState>
__parallel_transform_reduce_then_scan(sycl::queue& __q, const std::size_t __n, _InRng&& __in_rng, _OutRng&& __out_rng,
                                      _GenReduceInput __gen_reduce_input, _ReduceOp __reduce_op,
                                      _GenScanInput __gen_scan_input, _ScanInputTransform __scan_input_transform,
                                      _WriteOp __write_op, _InitType __init, _Inclusive __inclusive,
                                      _IsUniquePattern __is_unique_pattern,
                                      _StopPosInitState __stop_pos_initial_state = {},
                                      _TransformResult __transform_result = {}, sycl::event __prior_event = {})
{
    using _ValueType = typename _InitType::__value_type;

    // This static assert clarifies a cryptic error for "no matching function" due to mismatched type
    using _GenScanInputResult = typename _GenScanInput::template __result_t<std::decay_t<_InRng>>;
    using _ScannedValueType = std::decay_t<std::invoke_result_t<_ScanInputTransform, _GenScanInputResult&>>;
    static_assert(std::is_same_v<_ScannedValueType, _ValueType>,
                  "reduce-then-scan: the init value type must match the type produced by applying the scan input "
                  "transform to the scan input generator.");

    // Native sycl sub-group operations can only be used on trivially copyable types, for other types, use SLM variant
    if constexpr (std::is_trivially_copyable_v<_ValueType>)
    {
        bool __use_subgroup_ops = __q.get_device().is_gpu();
        return __parallel_transform_reduce_then_scan_impl<_Bounded, __slm_or_subgroup_tag<_ValueType>,
                                                          __bytes_per_work_item_iter, _CustomName>(
            __q, __n, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __gen_reduce_input, __reduce_op,
            __gen_scan_input, __scan_input_transform, __write_op, __init, __inclusive, __is_unique_pattern,
            __use_subgroup_ops, __transform_result, __stop_pos_initial_state, std::move(__prior_event));
    }
    else
    {
        return __parallel_transform_reduce_then_scan_impl<_Bounded, __slm_only_tag<_ValueType>,
                                                          __bytes_per_work_item_iter, _CustomName>(
            __q, __n, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __gen_reduce_input, __reduce_op,
            __gen_scan_input, __scan_input_transform, __write_op, __init, __inclusive, __is_unique_pattern,
            /*__use_subgroup_ops=*/false, __transform_result, __stop_pos_initial_state, std::move(__prior_event));
    }
}

template <typename _CustomName, typename _InInOutRng, typename _GenReduceInput>
sycl::event
__parallel_set_balanced_path_partition(sycl::queue& __q, _InInOutRng&& __in_in_out_rng, std::size_t __num_diagonals,
                                       _GenReduceInput __gen_reduce_input)
{
    using _PartitionKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_partition_kernel<_CustomName>>;
    using _PartitionSubmitter = __partition_set_balanced_path_submitter<_GenReduceInput, _PartitionKernel>;

    _PartitionSubmitter __partition_submitter{__gen_reduce_input};

    return __partition_submitter(__q, std::forward<_InInOutRng>(__in_in_out_rng), __num_diagonals);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H
