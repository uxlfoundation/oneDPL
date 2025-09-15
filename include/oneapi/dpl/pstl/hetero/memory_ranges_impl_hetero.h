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

#ifndef _ONEDPL_MEMORY_RANGES_IMPL_HETERO_H
#define _ONEDPL_MEMORY_RANGES_IMPL_HETERO_H

#if _ONEDPL_CPP20_RANGES_PRESENT

#include <ranges>
#include <utility>
#include <cassert>
#include <functional>
#include <type_traits>

#include "algorithm_ranges_impl_hetero.h" // for __pattern_walk_n
#include "dpcpp/execution_sycl_defs.h"    // for __hetero_tag

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_default_construct
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_default_construct(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    using _ValueType = std::ranges::range_value_t<_R>;

    auto __last = std::ranges::begin(__r) + std::ranges::size(__r);

    if constexpr (!std::is_trivially_default_constructible_v<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_default_construct<std::decay_t<_ExecutionPolicy>>{},
            std::forward<_R>(__r));
    }

    return __last;
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_value_construct
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_value_construct(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    using _ValueType = std::ranges::range_value_t<_R>;

    auto __last = std::ranges::begin(__r) + std::ranges::size(__r);

    if constexpr (oneapi::dpl::__internal::__trivial_uninitialized_value_construct<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_fill<__hetero_tag<_BackendTag>, _ValueType>{_ValueType()},
            std::forward<_R>(__r));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_value_construct<std::decay_t<_ExecutionPolicy>>{},
            std::forward<_R>(__r));
    }

    return __last;
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_copy
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r,
                             _OutRange&& __out_r)
{
    using _OutValueType = std::ranges::range_value_t<_OutRange>;
    using _OutRefType = std::ranges::range_reference_t<_OutRange>;
    using _InRefType = std::ranges::range_reference_t<_InRange>;

    auto __first1 = std::ranges::begin(__in_r);
    auto __first2 = std::ranges::begin(__out_r);

    const auto __n = oneapi::dpl::__ranges::__min_size_calc{}(__in_r, __out_r);
    if (__n == 0)
        return {__first1, __first2};

    auto __last1 = __first1 + __n;
    auto __last2 = __first2 + __n;

    if constexpr (oneapi::dpl::__internal::__trivial_uninitialized_copy<_OutValueType, _OutRefType, _InRefType>)
    {
        // subrange is used instead of take_view/drop_view because the latter throw exceptions in libstdc++10
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            std::ranges::subrange(__first1, __last1), std::ranges::subrange(__first2, __last2));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_copy<std::decay_t<_ExecutionPolicy>>{},
            std::ranges::subrange(__first1, __last1), std::ranges::subrange(__first2, __last2));
    }

    return {__last1, __last2};
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_move
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_move_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_move(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r,
                             _OutRange&& __out_r)
{
    using _OutValueType = std::ranges::range_value_t<_OutRange>;
    using _OutRefType = std::ranges::range_reference_t<_OutRange>;
    using _InRefType = std::ranges::range_reference_t<_InRange>;

    auto __first1 = std::ranges::begin(__in_r);
    auto __first2 = std::ranges::begin(__out_r);

    const auto __n = oneapi::dpl::__ranges::__min_size_calc{}(__in_r, __out_r);
    if (__n == 0)
        return {__first1, __first2};

    auto __last1 = __first1 + __n;
    auto __last2 = __first2 + __n;

    if constexpr (oneapi::dpl::__internal::__trivial_uninitialized_move<_OutValueType, _OutRefType, _InRefType>)
    {
        // subrange is used instead of take_view/drop_view because the latter throw exceptions in libstdc++10
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            std::ranges::subrange(__first1, __last1), std::ranges::subrange(__first2, __last2));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_move<std::decay_t<_ExecutionPolicy>>{},
            std::ranges::subrange(__first1, __last1), std::ranges::subrange(__first2, __last2));
    }

    return {__last1, __last2};
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_uninitialized_fill
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _T>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_fill(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, const _T& __value)
{
    using _ValueType = std::ranges::range_value_t<_R>;

    auto __first = std::ranges::begin(__r);
    auto __last = __first + std::ranges::size(__r);

    if constexpr (oneapi::dpl::__internal::__trivial_uninitialized_fill<_ValueType, _T>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_fill<__hetero_tag<_BackendTag>, _ValueType>{_ValueType(__value)},
            std::forward<_R>(__r));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_fill<_T, std::decay_t<_ExecutionPolicy>>{__value},
            std::forward<_R>(__r));
    }

    return __last;
}

//---------------------------------------------------------------------------------------------------------------------
// destroy
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_destroy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    using _ValueType = std::ranges::range_value_t<_R>;

    auto __last = std::ranges::begin(__r) + std::ranges::size(__r);

    if constexpr (!std::is_trivially_destructible_v<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_destroy<std::decay_t<_ExecutionPolicy>>{}, std::forward<_R>(__r));
    }

    return __last;
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_CPP20_RANGES_PRESENT
#endif // _ONEDPL_MEMORY_RANGES_IMPL_HETERO_H
