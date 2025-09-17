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
#include "../../utils.h"

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

template <bool __is_inclusive, typename _InitType, typename _BinaryOp>
struct __write_scan_by_seg
{
    using _TempData = __noop_temp_data;
    _InitType __init_value;
    _BinaryOp __binary_op;

    template <typename _OutRng, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _ValueType& __v, const _TempData&) const
    {
        using std::get;
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(get<1>(get<0>(__v)))>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
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
    _Assign __assign;
};

// *** Algorithm Specific Helpers, Input Generators to Reduction and Scan Operations ***

// __parallel_transform_scan

// A generator which applies a unary operation to the input range element at an index and returns the result
// converted to an underlying init type.
template <typename _UnaryOp, typename _InitType>
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
        return static_cast<_InitType>(__unary_op(_ValueType{__in_rng[__id]}));
    }
    _UnaryOp __unary_op;
};

// Scan copy algorithms (partition_copy, copy_if, unique_copy, set_reduce_then_scan_set_a_write)

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
template <typename _GenMask, typename _RangeTransform = oneapi::dpl::identity>
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
template <typename _SetTag, typename _Compare, typename _Proj1, typename _Proj2>
struct __gen_set_mask
{
    template <typename _InRng>
    bool
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __set_a = std::get<0>(__in_rng.tuple());    // first sequence, use with __proj1
        auto __set_b = std::get<1>(__in_rng.tuple());    // second sequence, use with __proj2
        auto __set_mask = std::get<2>(__in_rng.tuple()); // mask sequence

        std::size_t __nb = __set_b.size();

        // This reference extends the lifetime of a temporary object returned by operator[]
        // so that it can be safely used with identity projections
        auto&& __val_a = __set_a[__id];
        auto&& __val_a_proj = std::invoke(__proj1, std::forward<decltype(__val_a)>(__val_a));

        auto __res =
            oneapi::dpl::__internal::__pstl_lower_bound(__set_b, std::size_t{0}, __nb, __val_a_proj, __comp, __proj2);
        constexpr bool __is_difference = std::is_same_v<_SetTag, oneapi::dpl::unseq_backend::_DifferenceTag>;

        //initialization is true in case of difference operation; false - intersection.
        bool bres = __is_difference;

        if (__res == __nb || std::invoke(__comp, __val_a_proj, std::invoke(__proj2, __set_b[__res])))
        {
            // there is no __val_a in __set_b, so __set_b in the difference {__set_a}/{__set_b};
        }
        else
        {
            // This reference extends the lifetime of a temporary object returned by operator[]
            // so that it can be safely used with identity projections
            auto&& __val_b = __set_b[__res];
            auto&& __val_b_proj = std::invoke(__proj2, std::forward<decltype(__val_b)>(__val_b));

            //Difference operation logic: if number of duplication in __set_a on left side from __id > total number of
            //duplication in __set_b then a mask is 1

            //Intersection operation logic: if number of duplication in __set_a on left side from __id <= total number of
            //duplication in __set_b then a mask is 1

            const std::size_t __count_a_left =
                __id - oneapi::dpl::__internal::__pstl_left_bound(__set_a, std::size_t{0}, __id, __val_a_proj, __comp, __proj1) + 1;

            const std::size_t __count_b =
                oneapi::dpl::__internal::__pstl_right_bound(__set_b, __res, __nb, __val_b_proj, __comp, __proj2) -
                oneapi::dpl::__internal::__pstl_left_bound(__set_b, std::size_t{0}, __res, __val_b_proj, __comp,
                                                           __proj2);

            if constexpr (__is_difference)
                bres = __count_a_left > __count_b; /*difference*/
            else
                bres = __count_a_left <= __count_b; /*intersection*/
        }
        __set_mask[__id] = bres;
        return bres;
    }
    _Compare __comp;
    _Proj1 __proj1;
    _Proj2 __proj2;
};

// __parallel_set_write_a_b_op

// Returns by reference: iterations consumed, and the number of elements copied to temp output.
template <bool _CopyMatch, bool _CopyDiffSetA, bool _CopyDiffSetB, bool _CheckBounds, typename _InRng1,
          typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare, typename _Proj1,
          typename _Proj2>
void
__set_generic_operation_iteration(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t& __idx1,
                                  std::size_t& __idx2, const _SizeType __num_eles_min, _TempOutput& __temp_out,
                                  _SizeType& __idx, std::uint16_t& __count, const _Compare __comp, _Proj1 __proj1,
                                  _Proj2 __proj2)
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
    if (std::invoke(__comp, std::invoke(__proj1, __ele_rng1), std::invoke(__proj2, __ele_rng2)))
    {
        if constexpr (_CopyDiffSetA)
        {
            __temp_out.set(__count, __ele_rng1);
            ++__count;
        }
        ++__idx1;
        ++__idx;
    }
    else if (std::invoke(__comp, std::invoke(__proj2, __ele_rng2), std::invoke(__proj1, __ele_rng1)))
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
    template <typename _InRng1, typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare,
              typename _Proj1, typename _Proj2>
    std::uint16_t
    operator()(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t __idx1, std::size_t __idx2,
               const _SizeType __num_eles_min, _TempOutput& __temp_out, const _Compare __comp, _Proj1 __proj1,
               _Proj2 __proj2) const
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
                    __in_rng1, __in_rng2, __idx1, __idx2, __num_eles_min, __temp_out, __idx, __count, __comp, __proj1,
                    __proj2);
            }
        }
        else
        {
            while (__idx < __num_eles_min)
            {
                //bounds check all
                __set_generic_operation_iteration<_CopyMatch, _CopyDiffSetA, _CopyDiffSetB, true>(
                    __in_rng1, __in_rng2, __idx1, __idx2, __num_eles_min, __temp_out, __idx, __count, __comp, __proj1,
                    __proj2);
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
struct __get_set_operation<oneapi::dpl::unseq_backend::_IntersectionTag> : __set_intersection
{
};

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_DifferenceTag> : __set_difference
{
};
template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_UnionTag> : __set_union
{
};

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_SymmetricDifferenceTag> : __set_symmetric_difference
{
};

template <bool __return_star, typename _Rng, typename _IdxT>
auto
__decode_balanced_path_temp_data_impl(const _Rng& __rng, const _IdxT __id, const std::uint16_t __diagonal_spacing)
{
    using SizeT = decltype(__rng.size());
    using SignedSizeT = std::make_signed_t<decltype(__rng.size())>;
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
std::tuple<_IdxT, _IdxT, decltype(std::declval<_Rng>().size())>
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
        auto __rng_tmp_diag = std::get<2>(__in_rng.tuple()); // set a temp storage sequence

        using _SizeType = std::common_type_t<std::make_unsigned_t<decltype(std::get<0>(__in_rng.tuple()).size())>,
                                             std::make_unsigned_t<decltype(std::get<1>(__in_rng.tuple()).size())>,
                                             std::make_unsigned_t<decltype(__rng_tmp_diag.size())>>;

        // Establish bounds of ranges for the tile from sparse partitioning pass kernel

        // diagonal index of the tile begin
        const _SizeType __wg_begin_idx = (__id / __tile_size) * __tile_size;
        const _SizeType __signed_tile_size = static_cast<_SizeType>(__tile_size);
        const _SizeType __wg_end_idx =
            std::min<_SizeType>(((__id / __signed_tile_size) + 1) * __signed_tile_size, __rng_tmp_diag.size() - 1);

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
        const auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        const auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        using _SizeType = std::common_type_t<std::make_unsigned_t<decltype(__rng1.size())>,
                                             std::make_unsigned_t<decltype(__rng2.size())>>;

        return std::make_tuple(_SizeType{0}, static_cast<_SizeType>(__rng1.size()), _SizeType{0},
                               static_cast<_SizeType>(__rng2.size()));
    }
};

// Reduce then scan building block for set balanced path which is used in the reduction kernel to calculate the
// balanced path intersection, store it to temporary data with "star" status, then count the number of elements to write
// to the output for the reduction operation.
template <typename _SetOpCount, typename _BoundsProvider, typename _Compare, typename _Proj1, typename _Proj2>
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
        if (__merge_path_rng1 == 0 || __merge_path_rng2 == __rng2.size())
        {
            return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
        }

        // This reference extends the lifetime of a temporary object returned by operator[]
        // so that it can be safely used with identity projections
        auto&& __ele_val = __rng1[__merge_path_rng1 - 1];
        auto&& __ele_val_proj = std::invoke(__proj1, std::forward<decltype(__ele_val)>(__ele_val));

        if (std::invoke(__comp, __ele_val_proj, std::invoke(__proj2, __rng2[__merge_path_rng2])))
        {
            // There is no chance that the balanced path differs from the merge path here, because the previous element of
            // rng1 does not match the next element of rng2. We can just return the merge path.
            return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
        }

        // find first element of repeating sequence in the first set of the previous element
        _Index __rng1_repeat_start = oneapi::dpl::__internal::__biased_lower_bound</*__last_bias=*/true>(
            __rng1, __rng1_begin, __merge_path_rng1, __ele_val_proj, __comp, __proj1);
        // find first element of repeating sequence in the second set of the next element
        _Index __rng2_repeat_start = oneapi::dpl::__internal::__biased_lower_bound</*__last_bias=*/true>(
            __rng2, __rng2_begin, __merge_path_rng2, __ele_val_proj, __comp, __proj2);

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

        _Index __balanced_path_intersection_rng2 = oneapi::dpl::__internal::__pstl_upper_bound(
            __rng2, __merge_path_rng2, __fwd_search_bound, __ele_val_proj, __comp, __proj2);

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
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        const auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        const auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        auto __rng1_temp_diag = std::get<2>(__in_rng.tuple()); // set a temp storage sequence

        using _SizeType = std::common_type_t<std::make_unsigned_t<decltype(__rng1.size())>,
                                             std::make_unsigned_t<decltype(__rng2.size())>>;
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= __rng1.size() + __rng2.size())
            __i_elem = __rng1.size() + __rng2.size() - 1; // ensure we do not go out of bounds
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
    std::uint16_t
    operator()(const _InRng& __in_rng, _IndexT __id, TempData& __temp_data) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        const auto __rng1 = std::get<0>(__in_rng.tuple());   // first sequence
        const auto __rng2 = std::get<1>(__in_rng.tuple());   // second sequence
        auto __rng_tmp_diag = std::get<2>(__in_rng.tuple()); // temp diag sequence

        _IndexT __rng1_balanced_pos = 0;
        _IndexT __rng2_balanced_pos = 0;
        bool __star = false;

        const auto __total_size = __rng1.size() + __rng2.size();
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

        _IndexT __eles_to_process = std::min(_IndexT{__diagonal_spacing} - (__star ? _IndexT{1} : _IndexT{0}),
                                             __rng1.size() + __rng2.size() - _IndexT{__id * __diagonal_spacing - 1});

        std::uint16_t __count = __set_op_count(__rng1, __rng2, __rng1_balanced_pos, __rng2_balanced_pos,
                                               __eles_to_process, __temp_data, __comp, __proj1, __proj2);
        return __count;
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
template <typename _SetOpCount, typename _TempData, typename _Compare, typename _Proj1, typename _Proj2>
struct __gen_set_op_from_known_balanced_path
{
    using TempData = _TempData;
    template <typename _InRng, typename _IndexT>
    std::tuple<std::uint32_t, std::uint16_t>
    operator()(const _InRng& __in_rng, _IndexT __id, _TempData& __output_data) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        const auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        const auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        const auto __rng1_temp_diag =
            std::get<2>(__in_rng.tuple()); // set a temp storage sequence, star value in sign bit
        using _SizeType = std::common_type_t<std::make_unsigned_t<decltype(__rng1.size())>,
                                             std::make_unsigned_t<decltype(__rng2.size())>,
                                             std::make_unsigned_t<decltype(__rng1_temp_diag.size())>>;
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= __rng1.size() + __rng2.size())
            return std::make_tuple(std::uint32_t{0}, std::uint16_t{0});
        auto [__rng1_idx, __rng2_idx, __star_offset] =
            oneapi::dpl::__par_backend_hetero::__decode_balanced_path_temp_data(__rng1_temp_diag, __id,
                                                                                __diagonal_spacing);

        std::uint16_t __eles_to_process =
            static_cast<std::uint16_t>(std::min(static_cast<_SizeType>(__diagonal_spacing - __star_offset),
                                                static_cast<_SizeType>(__rng1.size() + __rng2.size() - __i_elem + 1)));

        std::uint16_t __count = __set_op_count(__rng1, __rng2, __rng1_idx, __rng2_idx, __eles_to_process, __output_data,
                                               __comp, __proj1, __proj2);

        return std::make_tuple(std::uint32_t{__count}, __count);
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
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

template <typename _BinaryPred>
struct __gen_scan_by_seg_reduce_input
{
    using TempData = __noop_temp_data;
    // Returns the following tuple:
    // (new_seg_mask, value)
    // bool new_seg_mask : true for a start of a new segment, false otherwise
    // ValueType value   : Current element's value for reduction
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        const auto __in_keys = std::get<0>(__in_rng.tuple());
        const auto __in_vals = std::get<1>(__in_rng.tuple());
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

template <typename _BinaryPred>
struct __gen_scan_by_seg_scan_input
{
    using TempData = __noop_temp_data;
    // Returns the following tuple:
    // ((new_seg_mask, value), new_seg_mask)
    // bool new_seg_mask : true for a start of a new segment, false otherwise
    // ValueType value   : Current element's value for reduction
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        const auto __in_keys = std::get<0>(__in_rng.tuple());
        const auto __in_vals = std::get<1>(__in_rng.tuple());
        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        // Mark the first index as a new segment as well as an indexing corresponding to any key
        // that does not satisfy the binary predicate with the previous key. The first tuple mask element
        // is scanned over, and the third is a placeholder for exclusive_scan_by_segment to perform init
        // handling in the output write.
        const std::uint32_t __new_seg_mask = __id == 0 || !__binary_pred(__in_keys[__id - 1], __in_keys[__id]);
        return oneapi::dpl::__internal::make_tuple(
            oneapi::dpl::__internal::make_tuple(__new_seg_mask, _ValueType{__in_vals[__id]}), __new_seg_mask);
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

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__exclusive_sub_group_masked_scan(const __dpl_sycl::__sub_group& __sub_group, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    _LazyValueType __old_init;
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        if (__sub_group_local_id == 0)
            __old_init.__setup(__init_and_carry.__v);
        __init_and_carry.__v = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    }

    __value = sycl::shift_group_right(__sub_group, __value, 1);
    if constexpr (__init_present)
    {
        if (__sub_group_local_id == 0)
        {
            __value = __old_init.__v;
            __old_init.__destroy();
        }
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__inclusive_sub_group_masked_scan(const __dpl_sycl::__sub_group& __sub_group, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        __init_and_carry.__v = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _MaskOp,
          typename _InitBroadcastId, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__sub_group_masked_scan(const __dpl_sycl::__sub_group& __sub_group, _MaskOp __mask_fn,
                        _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                        _LazyValueType& __init_and_carry)
{
    if constexpr (__is_inclusive)
    {
        __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
    else
    {
        __exclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType>
void
__sub_group_scan(const __dpl_sycl::__sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                 _LazyValueType& __init_and_carry)
{
    auto __mask_fn = [](auto __sub_group_local_id, auto __offset) { return __sub_group_local_id >= __offset; };
    constexpr std::uint8_t __init_broadcast_id = __sub_group_size - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry);
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType, typename _SizeType>
void
__sub_group_scan_partial(const __dpl_sycl::__sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                         _LazyValueType& __init_and_carry, _SizeType __elements_to_process)
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    std::uint8_t __init_broadcast_id = __elements_to_process - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry);
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, bool __capture_output,
          std::uint16_t __max_inputs_per_item, typename _GenInput, typename _ScanInputTransform, typename _BinaryOp,
          typename _WriteOp, typename _LazyValueType, typename _InRng, typename _OutRng>
void
__scan_through_elements_helper(const __dpl_sycl::__sub_group& __sub_group, _GenInput __gen_input,
                               _ScanInputTransform __scan_input_transform, _BinaryOp __binary_op, _WriteOp __write_op,
                               _LazyValueType& __sub_group_carry, const _InRng& __in_rng, _OutRng& __out_rng,
                               std::size_t __start_id, std::size_t __n, std::uint32_t __iters_per_item,
                               std::size_t __subgroup_start_id, std::uint32_t __sub_group_id,
                               std::uint32_t __active_subgroups)
{
    using _GenInputType = std::invoke_result_t<_GenInput, _InRng, std::size_t, typename _GenInput::TempData&>;

    bool __is_full_block = (__iters_per_item == __max_inputs_per_item);
    bool __is_full_thread = __subgroup_start_id + __iters_per_item * __sub_group_size <= __n;
    using _TempData = typename _GenInput::TempData;
    _TempData __temp_data{};
    if (__is_full_thread)
    {

        _GenInputType __v = __gen_input(__in_rng, __start_id, __temp_data);
        __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(__sub_group, __scan_input_transform(__v),
                                                                           __binary_op, __sub_group_carry);
        if constexpr (__capture_output)
        {
            __write_op(__out_rng, __start_id, __v, __temp_data);
        }

        if (__is_full_block)
        {
            // For full block and full thread, we can unroll the loop
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __j = 1; __j < __max_inputs_per_item; __j++)
            {
                __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size, __temp_data);
                __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __write_op(__out_rng, __start_id + __j * __sub_group_size, __v, __temp_data);
                }
            }
        }
        else
        {
            // For full thread but not full block, we can't unroll the loop, but we
            // can proceed without special casing for partial subgroups.
            for (std::uint32_t __j = 1; __j < __iters_per_item; __j++)
            {
                __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size, __temp_data);
                __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __write_op(__out_rng, __start_id + __j * __sub_group_size, __v, __temp_data);
                }
            }
        }
    }
    else
    {
        // For partial thread, we need to handle the partial subgroup at the end of the range
        if (__sub_group_id < __active_subgroups)
        {
            std::uint32_t __iters =
                oneapi::dpl::__internal::__dpl_ceiling_div(__n - __subgroup_start_id, __sub_group_size);

            if (__iters == 1)
            {
                std::size_t __local_id = (__start_id < __n) ? __start_id : __n - 1;
                _GenInputType __v = __gen_input(__in_rng, __local_id, __temp_data);
                __sub_group_scan_partial<__sub_group_size, __is_inclusive, __init_present>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                    __n - __subgroup_start_id);
                if constexpr (__capture_output)
                {
                    if (__start_id < __n)
                        __write_op(__out_rng, __start_id, __v, __temp_data);
                }
            }
            else
            {
                _GenInputType __v = __gen_input(__in_rng, __start_id, __temp_data);
                __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __write_op(__out_rng, __start_id, __v, __temp_data);
                }

                for (std::uint32_t __j = 1; __j < __iters - 1; __j++)
                {
                    std::size_t __local_id = __start_id + __j * __sub_group_size;
                    __v = __gen_input(__in_rng, __local_id, __temp_data);
                    __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                        __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                    if constexpr (__capture_output)
                    {
                        __write_op(__out_rng, __local_id, __v, __temp_data);
                    }
                }

                std::size_t __offset = __start_id + (__iters - 1) * __sub_group_size;
                std::size_t __local_id = (__offset < __n) ? __offset : __n - 1;
                __v = __gen_input(__in_rng, __local_id, __temp_data);
                __sub_group_scan_partial<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                    __n - (__subgroup_start_id + (__iters - 1) * __sub_group_size));
                if constexpr (__capture_output)
                {
                    if (__offset < __n)
                        __write_op(__out_rng, __offset, __v, __temp_data);
                }
            }
        }
    }
}

constexpr inline std::uint8_t
__get_reduce_then_scan_default_sg_sz()
{
    return 32;
}

constexpr inline std::uint8_t
__get_reduce_then_scan_workaround_sg_sz()
{
    return 16;
}

// The default sub-group size for reduce-then-scan is 32, but we conditionally enable sub-group sizes of 16 on Intel
// devices to workaround a hardware bug. From the host side, return 32 to assert that this sub-group size is supported
// by an arbitrary device.
constexpr inline std::uint8_t
__get_reduce_then_scan_reqd_sg_sz_host()
{
    return __get_reduce_then_scan_default_sg_sz();
}

// To workaround a hardware bug on certain Intel iGPUs with older driver versions and -O0 device compilation, use a
// sub-group size of 16. Note this function may only be called on the device as _ONEDPL_DETECT_SPIRV_COMPILATION is only
// valid here.
constexpr inline std::uint8_t
__get_reduce_then_scan_actual_sg_sz_device()
{
    return
#if _ONEDPL_DETECT_COMPILER_OPTIMIZATIONS_ENABLED || !_ONEDPL_DETECT_SPIRV_COMPILATION
        __get_reduce_then_scan_default_sg_sz();
#else
        __get_reduce_then_scan_workaround_sg_sz();
#endif
}

struct __reduce_then_scan_sub_group_params
{
    __reduce_then_scan_sub_group_params(std::uint32_t __work_group_size, std::uint8_t __sub_group_size,
                                        std::uint32_t __max_num_work_groups, std::uint32_t __max_block_size,
                                        std::size_t __inputs_remaining)
        : __num_sub_groups_local(__work_group_size / __sub_group_size),
          __num_sub_groups_global(__num_sub_groups_local * __max_num_work_groups)
    {
        const std::uint32_t __max_inputs_per_subgroup = __max_block_size / __num_sub_groups_global;
        const std::uint32_t __evenly_divided_remaining_inputs =
            std::max(std::size_t{__sub_group_size},
                     oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining) / __num_sub_groups_global);
        __inputs_per_sub_group =
            __inputs_remaining >= __max_block_size ? __max_inputs_per_subgroup : __evenly_divided_remaining_inputs;
        __inputs_per_item = __inputs_per_sub_group / __sub_group_size;
    }

    std::uint32_t __num_sub_groups_local;
    std::uint32_t __num_sub_groups_global;
    std::uint32_t __inputs_per_sub_group;
    std::uint32_t __inputs_per_item;
};

template <typename... _Name>
class __reduce_then_scan_partition_kernel;

template <typename... _Name>
class __reduce_then_scan_reduce_kernel;

template <typename... _Name>
class __reduce_then_scan_scan_kernel;

template <std::uint16_t __max_inputs_per_item, bool __is_inclusive, bool __is_unique_pattern_v,
          typename _GenReduceInput, typename _ReduceOp, typename _InitType, typename _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter;

template <std::uint16_t __max_inputs_per_item, bool __is_inclusive, bool __is_unique_pattern_v,
          typename _GenReduceInput, typename _ReduceOp, typename _InitType, typename... _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter<__max_inputs_per_item, __is_inclusive, __is_unique_pattern_v,
                                                    _GenReduceInput, _ReduceOp, _InitType,
                                                    __internal::__optional_kernel_name<_KernelName...>>
{
    static constexpr std::uint8_t __sub_group_size = __get_reduce_then_scan_actual_sg_sz_device();
    // Step 1 - SubGroupReduce is expected to perform sub-group reductions to global memory
    // input buffer
    template <typename _InRng, typename _TmpStorageAcc>
    sycl::event
    operator()(sycl::queue& __q, const sycl::nd_range<1> __nd_range, _InRng&& __in_rng,
               _TmpStorageAcc& __scratch_container, const sycl::event& __prior_event,
               const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        using _InitValueType = typename _InitType::__value_type;
        return __q.submit([&, this](sycl::handler& __cgh) {
            __dpl_sycl::__local_accessor<_InitValueType> __sub_group_partials(__max_num_sub_groups_local, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng);
            auto __temp_acc = __scratch_container.template __get_scratch_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});
            __cgh.parallel_for<_KernelName...>(
                    __nd_range, [=, *this](sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(__sub_group_size)]] {
                // Compute work distribution fields dependent on sub-group size within the kernel. This is because we
                // can only rely on the value of __sub_group_size provided in the device compilation phase within the
                // kernel itself.
                __reduce_then_scan_sub_group_params __sub_group_params(
                    __work_group_size, __sub_group_size, __max_num_work_groups, __max_block_size, __inputs_remaining);

                _InitValueType* __temp_ptr = _TmpStorageAcc::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                std::size_t __group_id = __ndi.get_group(0);
                __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
                std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
                std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

                oneapi::dpl::__internal::__lazy_ctor_storage<_InitValueType> __sub_group_carry;
                std::size_t __group_start_id =
                    (__block_num * __max_block_size) + (__group_id * __sub_group_params.__inputs_per_sub_group *
                                                        __sub_group_params.__num_sub_groups_local);
                if constexpr (__is_unique_pattern_v)
                {
                    // for unique patterns, the first element is always copied to the output, so we need to skip it
                    __group_start_id += 1;
                }
                std::size_t __max_inputs_in_group =
                    __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local;
                std::uint32_t __inputs_in_group = std::min(__n - __group_start_id, __max_inputs_in_group);
                std::uint32_t __active_subgroups = oneapi::dpl::__internal::__dpl_ceiling_div(
                    __inputs_in_group, __sub_group_params.__inputs_per_sub_group);
                std::size_t __subgroup_start_id =
                    __group_start_id + (__sub_group_id * __sub_group_params.__inputs_per_sub_group);

                std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

                if (__sub_group_id < __active_subgroups)
                {
                    // adjust for lane-id
                    // compute sub-group local prefix on T0..63, K samples/T, send to accumulator kernel
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/false,
                                                   /*__capture_output=*/false, __max_inputs_per_item>(
                        __sub_group, __gen_reduce_input, oneapi::dpl::identity{}, __reduce_op, nullptr,
                        __sub_group_carry, __in_rng, /*unused*/ __in_rng, __start_id, __n,
                        __sub_group_params.__inputs_per_item, __subgroup_start_id, __sub_group_id, __active_subgroups);
                    if (__sub_group_local_id == 0)
                        __sub_group_partials[__sub_group_id] = __sub_group_carry.__v;
                    __sub_group_carry.__destroy();
                }
                __dpl_sycl::__group_barrier(__ndi);

                // compute sub-group local prefix sums on (T0..63) carries
                // and store to scratch space at the end of dst; next
                // accumulator kernel takes M thread carries from scratch
                // to compute a prefix sum on global carries
                if (__sub_group_id == 0)
                {
                    __start_id = (__group_id * __sub_group_params.__num_sub_groups_local);
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    if (__iters == 1)
                    {
                        // fill with unused dummy values to avoid overrunning input
                        std::uint32_t __load_id = std::min(std::uint32_t{__sub_group_local_id}, __active_subgroups - 1);
                        _InitValueType __v = __sub_group_partials[__load_id];
                        __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                            __sub_group, __v, __reduce_op, __sub_group_carry, __active_subgroups);
                        if (__sub_group_local_id < __active_subgroups)
                            __temp_ptr[__start_id + __sub_group_local_id] = __v;
                    }
                    else
                    {
                        std::uint32_t __reduction_scan_id = __sub_group_local_id;
                        // need to pull out first iteration tp avoid identity
                        _InitValueType __v = __sub_group_partials[__reduction_scan_id];
                        __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                            __sub_group, __v, __reduce_op, __sub_group_carry);
                        __temp_ptr[__start_id + __reduction_scan_id] = __v;
                        __reduction_scan_id += __sub_group_size;

                        for (std::uint32_t __i = 1; __i < __iters - 1; __i++)
                        {
                            __v = __sub_group_partials[__reduction_scan_id];
                            __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                                __sub_group, __v, __reduce_op, __sub_group_carry);
                            __temp_ptr[__start_id + __reduction_scan_id] = __v;
                            __reduction_scan_id += __sub_group_size;
                        }
                        // If we are past the input range, then the previous value of v is passed to the sub-group scan.
                        // It does not affect the result as our sub_group_scan will use a mask to only process in-range elements.

                        // fill with unused dummy values to avoid overrunning input
                        std::uint32_t __load_id =
                            std::min(__reduction_scan_id, __sub_group_params.__num_sub_groups_local - 1);

                        __v = __sub_group_partials[__load_id];
                        __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                            __sub_group, __v, __reduce_op, __sub_group_carry,
                            __active_subgroups - ((__iters - 1) * __sub_group_size));
                        if (__reduction_scan_id < __sub_group_params.__num_sub_groups_local)
                            __temp_ptr[__start_id + __reduction_scan_id] = __v;
                    }

                    __sub_group_carry.__destroy();
                }
            });
        });
    }

    // Constant parameters throughout all blocks
    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::size_t __n;

    const _GenReduceInput __gen_reduce_input;
    const _ReduceOp __reduce_op;
    _InitType __init;
};

template <std::uint16_t __max_inputs_per_item, bool __is_inclusive, bool __is_unique_pattern_v, typename _ReduceOp,
          typename _GenScanInput, typename _ScanInputTransform, typename _WriteOp, typename _InitType,
          typename _KernelName>
struct __parallel_reduce_then_scan_scan_submitter;

template <std::uint16_t __max_inputs_per_item, bool __is_inclusive, bool __is_unique_pattern_v, typename _ReduceOp,
          typename _GenScanInput, typename _ScanInputTransform, typename _WriteOp, typename _InitType,
          typename... _KernelName>
struct __parallel_reduce_then_scan_scan_submitter<__max_inputs_per_item, __is_inclusive, __is_unique_pattern_v,
                                                  _ReduceOp, _GenScanInput, _ScanInputTransform, _WriteOp, _InitType,
                                                  __internal::__optional_kernel_name<_KernelName...>>
{
    using _InitValueType = typename _InitType::__value_type;
    static constexpr std::uint8_t __sub_group_size = __get_reduce_then_scan_actual_sg_sz_device();

    _InitValueType
    __get_block_carry_in(const std::size_t __block_num, _InitValueType* __tmp_ptr,
                         const std::size_t __num_sub_groups_global) const
    {
        return __tmp_ptr[__num_sub_groups_global + (__block_num % 2)];
    }

    template <typename _ValueType>
    void
    __set_block_carry_out(const std::size_t __block_num, _InitValueType* __tmp_ptr, const _ValueType __block_carry_out,
                          const std::size_t __num_sub_groups_global) const
    {
        __tmp_ptr[__num_sub_groups_global + 1 - (__block_num % 2)] = __block_carry_out;
    }

    template <typename _InRng, typename _OutRng, typename _TmpStorageAcc>
    sycl::event
    operator()(sycl::queue& __q, const sycl::nd_range<1> __nd_range, _InRng&& __in_rng, _OutRng&& __out_rng,
               _TmpStorageAcc& __scratch_container, const sycl::event& __prior_event,
               const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        std::size_t __num_remaining = __n - __block_num * __max_block_size;
        // for unique patterns, the first element is always copied to the output, so we need to skip it
        if constexpr (__is_unique_pattern_v)
        {
            assert(__num_remaining > 0);
            __num_remaining -= 1;
        }
        std::uint32_t __inputs_in_block = std::min(__num_remaining, std::size_t{__max_block_size});
        return __q.submit([&, this](sycl::handler& __cgh) {
            // We need __num_sub_groups_local + 1 temporary SLM locations to store intermediate results:
            //   __num_sub_groups_local for each sub-group partial from the reduce kernel +
            //   1 element for the accumulated block-local carry-in from previous groups in the block
            __dpl_sycl::__local_accessor<_InitValueType> __sub_group_partials(__max_num_sub_groups_local + 1, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng, __out_rng);
            auto __temp_acc = __scratch_container.template __get_scratch_acc<sycl::access_mode::read_write>(__cgh);
            auto __res_acc =
                __scratch_container.template __get_result_acc<sycl::access_mode::write>(__cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<_KernelName...>(
                    __nd_range, [=, *this] (sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(__sub_group_size)]] {
                // Compute work distribution fields dependent on sub-group size within the kernel. This is because we
                // can only rely on the value of __sub_group_size provided in the device compilation phase within the
                // kernel itself.
                __reduce_then_scan_sub_group_params __sub_group_params(
                    __work_group_size, __sub_group_size, __max_num_work_groups, __max_block_size, __inputs_remaining);

                const std::uint32_t __active_groups = oneapi::dpl::__internal::__dpl_ceiling_div(
                    __inputs_in_block,
                    __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local);

                _InitValueType* __tmp_ptr = _TmpStorageAcc::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                _InitValueType* __res_ptr =
                    _TmpStorageAcc::__get_usm_or_buffer_accessor_ptr(__res_acc, __max_num_sub_groups_global + 2);
                std::uint32_t __group_id = __ndi.get_group(0);
                __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
                std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
                std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

                std::size_t __group_start_id =
                    (__block_num * __max_block_size) + (__group_id * __sub_group_params.__inputs_per_sub_group *
                                                        __sub_group_params.__num_sub_groups_local);
                if constexpr (__is_unique_pattern_v)
                {
                    // for unique patterns, the first element is always copied to the output, so we need to skip it
                    __group_start_id += 1;
                }

                std::size_t __max_inputs_in_group =
                    __sub_group_params.__inputs_per_sub_group * __sub_group_params.__num_sub_groups_local;
                std::uint32_t __inputs_in_group = std::min(__n - __group_start_id, __max_inputs_in_group);
                std::uint32_t __active_subgroups = oneapi::dpl::__internal::__dpl_ceiling_div(
                    __inputs_in_group, __sub_group_params.__inputs_per_sub_group);
                oneapi::dpl::__internal::__lazy_ctor_storage<_InitValueType> __carry_last;

                // propagate carry in from previous block
                oneapi::dpl::__internal::__lazy_ctor_storage<_InitValueType> __sub_group_carry;

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
                    std::size_t __subgroups_before_my_group = __group_id * __sub_group_params.__num_sub_groups_local;
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
                    std::uint32_t __offset = __sub_group_params.__num_sub_groups_local - 1;
                    // only need 32 carries for WGs0..WG32, 64 for WGs32..WGs64, etc.
                    if (__group_id > 0)
                    {
                        // only need the last element from each scan of num_sub_groups_local subgroup reductions
                        const std::size_t __elements_to_process =
                            __subgroups_before_my_group / __sub_group_params.__num_sub_groups_local;
                        const std::size_t __pre_carry_iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__elements_to_process, __sub_group_size);
                        if (__pre_carry_iters == 1)
                        {
                            // single partial scan
                            std::size_t __proposed_id =
                                __sub_group_params.__num_sub_groups_local * __sub_group_local_id + __offset;
                            std::size_t __remaining_elements = __elements_to_process;
                            std::size_t __reduction_id = (__proposed_id < __subgroups_before_my_group)
                                                             ? __proposed_id
                                                             : __subgroups_before_my_group - 1;
                            _InitValueType __value = __tmp_ptr[__reduction_id];
                            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true,
                                                     /*__init_present=*/false>(__sub_group, __value, __reduce_op,
                                                                               __carry_last, __remaining_elements);
                        }
                        else
                        {
                            // multiple iterations
                            // first 1 full
                            std::uint32_t __reduction_id =
                                __sub_group_params.__num_sub_groups_local * __sub_group_local_id + __offset;
                            std::uint32_t __reduction_id_increment =
                                __sub_group_params.__num_sub_groups_local * __sub_group_size;
                            _InitValueType __value = __tmp_ptr[__reduction_id];
                            __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                                __sub_group, __value, __reduce_op, __carry_last);
                            __reduction_id += __reduction_id_increment;
                            // then some number of full iterations
                            for (std::uint32_t __i = 1; __i < __pre_carry_iters - 1; __i++)
                            {
                                __value = __tmp_ptr[__reduction_id];
                                __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                                    __sub_group, __value, __reduce_op, __carry_last);
                                __reduction_id += __reduction_id_increment;
                            }

                            // final partial iteration

                            std::size_t __remaining_elements =
                                __elements_to_process - ((__pre_carry_iters - 1) * __sub_group_size);
                            // fill with unused dummy values to avoid overrunning input
                            std::size_t __final_reduction_id =
                                std::min(std::size_t{__reduction_id}, __subgroups_before_my_group - 1);
                            __value = __tmp_ptr[__final_reduction_id];
                            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true,
                                                     /*__init_present=*/true>(__sub_group, __value, __reduce_op,
                                                                              __carry_last, __remaining_elements);
                        }

                        // steps 3+4) load global carry in from neighbor work-group
                        //            and apply to local sub-group prefix carries
                        std::size_t __carry_offset = __sub_group_local_id;

                        std::uint8_t __iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);

                        std::uint8_t __i = 0;
                        for (; __i < __iters - 1; ++__i)
                        {
                            __sub_group_partials[__carry_offset] =
                                __reduce_op(__carry_last.__v, __sub_group_partials[__carry_offset]);
                            __carry_offset += __sub_group_size;
                        }
                        if (__i * __sub_group_size + __sub_group_local_id < __active_subgroups)
                        {
                            __sub_group_partials[__carry_offset] =
                                __reduce_op(__carry_last.__v, __sub_group_partials[__carry_offset]);
                            __carry_offset += __sub_group_size;
                        }
                        if (__sub_group_local_id == 0)
                            __sub_group_partials[__active_subgroups] = __carry_last.__v;
                        __carry_last.__destroy();
                    }
                }

                __dpl_sycl::__group_barrier(__ndi);

                // Get inter-work group and adjusted for intra-work group prefix
                bool __sub_group_carry_initialized = true;
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

                        if constexpr (std::is_same_v<_InitType,
                                                     oneapi::dpl::unseq_backend::__no_init_value<_InitValueType>>)
                        {
                            // This is the only case where we still don't have a carry in.  No init value, 0th block,
                            // group, and subgroup. This changes the final scan through elements below.
                            __sub_group_carry_initialized = false;
                        }
                        else
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
                            __get_block_carry_in(__block_num, __tmp_ptr, __sub_group_params.__num_sub_groups_global),
                            __value));
                    }
                    else if (__group_id > 0)
                    {
                        __sub_group_carry.__setup(__reduce_op(
                            __get_block_carry_in(__block_num, __tmp_ptr, __sub_group_params.__num_sub_groups_global),
                            __sub_group_partials[__active_subgroups]));
                    }
                    else
                    {
                        __sub_group_carry.__setup(
                            __get_block_carry_in(__block_num, __tmp_ptr, __sub_group_params.__num_sub_groups_global));
                    }
                }

                // step 5) apply global carries
                std::size_t __subgroup_start_id =
                    __group_start_id + (__sub_group_id * __sub_group_params.__inputs_per_sub_group);
                std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

                if (__sub_group_carry_initialized)
                {
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/true,
                                                   /*__capture_output=*/true, __max_inputs_per_item>(
                        __sub_group, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op,
                        __sub_group_carry, __in_rng, __out_rng, __start_id, __n, __sub_group_params.__inputs_per_item,
                        __subgroup_start_id, __sub_group_id, __active_subgroups);
                }
                else // first group first block, no subgroup carry
                {
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/false,
                                                   /*__capture_output=*/true, __max_inputs_per_item>(
                        __sub_group, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op,
                        __sub_group_carry, __in_rng, __out_rng, __start_id, __n, __sub_group_params.__inputs_per_item,
                        __subgroup_start_id, __sub_group_id, __active_subgroups);
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
                            __res_ptr[0] = __sub_group_carry.__v + 1;
                        }
                        else
                        {
                            __res_ptr[0] = __sub_group_carry.__v;
                        }
                    }
                    else
                    {
                        // capture the last carry out for the next block
                        __set_block_carry_out(__block_num, __tmp_ptr, __sub_group_carry.__v,
                                              __sub_group_params.__num_sub_groups_global);
                    }
                }
                __sub_group_carry.__destroy();
            });
        });
    }

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::uint32_t __max_num_sub_groups_global;
    const std::size_t __num_blocks;
    const std::size_t __n;

    const _ReduceOp __reduce_op;
    const _GenScanInput __gen_scan_input;
    const _ScanInputTransform __scan_input_transform;
    const _WriteOp __write_op;
    _InitType __init;
};

// Enable reduce-then-scan if the device uses the required sub-group size and is ran on a device
// with fast coordinated subgroup operations. We do not want to run this scan on CPU targets, as they are not
// performant with this algorithm.
inline bool
__is_gpu_with_reduce_then_scan_sg_sz(const sycl::queue& __q)
{
    return (__q.get_device().is_gpu() &&
            oneapi::dpl::__internal::__supports_sub_group_size(__q, __get_reduce_then_scan_reqd_sg_sz_host()));
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
template <std::uint32_t __bytes_per_work_item_iter, typename _CustomName, typename _InRng, typename _OutRng,
          typename _GenReduceInput, typename _ReduceOp, typename _GenScanInput, typename _ScanInputTransform,
          typename _WriteOp, typename _InitType, typename _Inclusive, typename _IsUniquePattern>
__future<sycl::event, __result_and_scratch_storage<typename _InitType::__value_type>>
__parallel_transform_reduce_then_scan(sycl::queue& __q, const std::size_t __n, _InRng&& __in_rng, _OutRng&& __out_rng,
                                      _GenReduceInput __gen_reduce_input, _ReduceOp __reduce_op,
                                      _GenScanInput __gen_scan_input, _ScanInputTransform __scan_input_transform,
                                      _WriteOp __write_op, _InitType __init, _Inclusive, _IsUniquePattern,
                                      sycl::event __prior_event = {})
{
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_reduce_kernel<_CustomName>>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_scan_kernel<_CustomName>>;
    using _ValueType = typename _InitType::__value_type;

    constexpr std::uint8_t __min_sub_group_size = __get_reduce_then_scan_workaround_sg_sz();
    constexpr std::uint8_t __max_sub_group_size = __get_reduce_then_scan_default_sg_sz();
    // Empirically determined maximum. May be less for non-full blocks.
    constexpr std::uint16_t __max_inputs_per_item =
        std::max(std::uint16_t{1}, std::uint16_t{512 / __bytes_per_work_item_iter});
    constexpr bool __inclusive = _Inclusive::value;
    constexpr bool __is_unique_pattern_v = _IsUniquePattern::value;

    const std::uint32_t __max_work_group_size = oneapi::dpl::__internal::__max_work_group_size(__q, 8192);
    // Round down to nearest multiple of the subgroup size
    const std::uint32_t __work_group_size = (__max_work_group_size / __max_sub_group_size) * __max_sub_group_size;

    // TODO: Investigate potentially basing this on some scale of the number of compute units. 128 work-groups has been
    // found to be reasonable number for most devices.
    constexpr std::uint32_t __num_work_groups = 128;
    // We may use a sub-group size of 16 or 32 depending on the compiler optimization level. Allocate sufficient
    // temporary storage to handle both cases.
    const std::uint32_t __max_num_sub_groups_local = __work_group_size / __min_sub_group_size;
    const std::uint32_t __max_num_sub_groups_global = __max_num_sub_groups_local * __num_work_groups;
    const std::uint32_t __max_inputs_per_work_group = __work_group_size * __max_inputs_per_item;
    const std::uint32_t __max_inputs_per_block = __max_inputs_per_work_group * __num_work_groups;
    std::size_t __inputs_remaining = __n;
    if constexpr (__is_unique_pattern_v)
    {
        // skip scan of zeroth element in unique patterns
        __inputs_remaining -= 1;
    }
    // reduce_then_scan kernel is not built to handle "empty" scans which includes `__n == 1` for unique patterns.
    // These trivial end cases should be handled at a higher level.
    assert(__inputs_remaining > 0);
    std::uint32_t __inputs_per_item =
        __inputs_remaining >= __max_inputs_per_block
            ? __max_inputs_per_item
            : oneapi::dpl::__internal::__dpl_ceiling_div(oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                                                         __num_work_groups * __work_group_size);
    const std::size_t __block_size = std::min(__inputs_remaining, std::size_t{__max_inputs_per_block});
    const std::size_t __num_blocks = __inputs_remaining / __block_size + (__inputs_remaining % __block_size != 0);

    // We need temporary storage for reductions of each sub-group (__num_sub_groups_global).
    // Additionally, we need two elements for the block carry-out to prevent a race condition
    // between reading and writing the block carry-out within a single kernel.
    __result_and_scratch_storage<_ValueType> __result_and_scratch{__q, __max_num_sub_groups_global + 2};

    // Reduce and scan step implementations
    using _ReduceSubmitter =
        __parallel_reduce_then_scan_reduce_submitter<__max_inputs_per_item, __inclusive, __is_unique_pattern_v,
                                                     _GenReduceInput, _ReduceOp, _InitType, _ReduceKernel>;
    using _ScanSubmitter =
        __parallel_reduce_then_scan_scan_submitter<__max_inputs_per_item, __inclusive, __is_unique_pattern_v, _ReduceOp,
                                                   _GenScanInput, _ScanInputTransform, _WriteOp, _InitType,
                                                   _ScanKernel>;
    _ReduceSubmitter __reduce_submitter{__num_work_groups,
                                        __work_group_size,
                                        __max_inputs_per_block,
                                        __max_num_sub_groups_local,
                                        __n,
                                        __gen_reduce_input,
                                        __reduce_op,
                                        __init};
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
                                    __init};

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
                                           __inputs_remaining, __b);
        // 2. Scan step - Compute intra-wg carries, determine sub-group carry-ins, and perform full input block scan.
        __prior_event = __scan_submitter(__q, __kernel_nd_range, __in_rng, __out_rng, __result_and_scratch,
                                         __prior_event, __inputs_remaining, __b);
        __inputs_remaining -= std::min(__inputs_remaining, __block_size);
        if (__b + 2 == __num_blocks)
        {
            __inputs_per_item = __inputs_remaining >= __max_inputs_per_block
                                    ? __max_inputs_per_item
                                    : oneapi::dpl::__internal::__dpl_ceiling_div(
                                          oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                                          __num_work_groups * __work_group_size);
        }
    }
    return __future{std::move(__prior_event), std::move(__result_and_scratch)};
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
