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

// This file contains SYCL specific macros and abstractions
// to support different versions of SYCL and to simplify its interfaces
//
// Include this header instead of sycl.hpp throughout the project

#ifndef _ONEDPL_UTILS_HETERO_H
#define _ONEDPL_UTILS_HETERO_H

#include "../utils.h"
#include <tuple>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _Predicate, typename _ValueType>
struct __create_mask_unique_copy
{
    _Predicate __predicate;

    template <typename _Idx, typename _Acc>
    _ValueType
    operator()(_Idx __idx, _Acc& __acc) const
    {
        using ::std::get;

        auto __predicate_result = 1;
        if (__idx != 0)
            __predicate_result = __predicate(get<0>(__acc[__idx]), get<0>(__acc[__idx + (-1)]));

        get<1>(__acc[__idx]) = __predicate_result;
        return _ValueType{__predicate_result};
    }
};

template <typename _Compare, typename _ReduceValueType>
struct __pattern_minmax_element_reduce_fn
{
    _Compare __comp;

    _ReduceValueType
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        using std::get;
        auto __chosen_for_min = __a;
        auto __chosen_for_max = __b;

        if (__comp(get<2>(__b), get<2>(__a)))
            __chosen_for_min = std::move(__b);
        if (__comp(get<3>(__b), get<3>(__a)))
            __chosen_for_max = std::move(__a);
        return _ReduceValueType{get<0>(__chosen_for_min), get<1>(__chosen_for_max), get<2>(__chosen_for_min),
                                get<3>(__chosen_for_max)};
    }
};

template <typename _ReduceValueType, typename _Compare>
struct __pattern_min_element_reduce_fn
{
    _Compare __comp;

    _ReduceValueType
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        using std::get;
        // TODO: Consider removing the non-commutative operator for SPIR-V targets when we see improved performance with the
        // non-sequential load path in transform_reduce.
        if constexpr (oneapi::dpl::__internal::__is_spirv_target_v)
        {
            // This operator doesn't track the lowest found index in case of equal min. or max. values. Thus, this operator is
            // not commutative.
            if (__comp(get<1>(__b), get<1>(__a)))
            {
                return __b;
            }
            return __a;
        }
        else
        {
            // This operator keeps track of the lowest found index in case of equal min. or max. values. Thus, this operator is
            // commutative.
            bool _is_a_lt_b = __comp(get<1>(__a), get<1>(__b));
            bool _is_b_lt_a = __comp(get<1>(__b), get<1>(__a));

            if (_is_b_lt_a || (!_is_a_lt_b && get<0>(__b) < get<0>(__a)))
            {
                return __b;
            }
            return __a;
        }
    }
};

template <typename _ReduceValueType>
struct __pattern_minmax_element_transform_fn
{
    template <typename _TGroupIdx, typename _TAcc>
    _ReduceValueType
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return _ReduceValueType{__gidx, __gidx, __acc[__gidx], __acc[__gidx]};
    }
};

template <typename _Predicate>
struct __pattern_count_transform_fn
{
    _Predicate __predicate;

    // int is being implicitly casted to difference_type
    // otherwise we can only pass the difference_type as a functor template parameter
    template <typename _TGroupIdx, typename _TAcc>
    int
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return (__predicate(__acc[__gidx]) ? 1 : 0);
    }
};

template <typename _ReduceValueType>
struct __pattern_min_element_transform_fn
{
    template <typename _TGroupIdx, typename _TAcc>
    _ReduceValueType
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return _ReduceValueType{__gidx, __acc[__gidx]};
    };
};

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UTILS_HETERO_H
