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

#    include <ranges>
#    include <utility>
#    include <cassert>
#    include <functional>
#    include <type_traits>

#    include "algorithm_ranges_impl_hetero.h" // for __pattern_walk_n
#    include "dpcpp/utils_ranges_sycl.h" // for oneapi::dpl::__internal::__ranges::views::all, etc.
#    include "dpcpp/execution_sycl_defs.h" // for __hetero_tag

namespace oneapi::dpl::__internal::__ranges
{

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_default_construct
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_default_construct(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    using _ValueType = typename std::ranges::range_value_t<_R>;

    if constexpr (!std::is_trivially_default_constructible_v<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_default_construct<std::decay_t<_ExecutionPolicy>>{},
            oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));
    }
    return std::ranges::borrowed_iterator_t<_R>{std::ranges::end(__r)};
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_value_construct
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_value_construct(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    using _ValueType = typename std::ranges::range_value_t<_R>;

    if constexpr (std::is_trivially_default_constructible_v<_ValueType> &&
                  std::is_trivially_copy_assignable_v<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_fill<__hetero_tag<_BackendTag>, _ValueType>{_ValueType()},
            oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_value_construct<std::decay_t<_ExecutionPolicy>>{},
            oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));
    }

    return std::ranges::borrowed_iterator_t<_R>{std::ranges::end(__r)};
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_copy
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    using _OutValueType = std::ranges::range_value_t<_OutRange>;
    using _OutRefType = std::ranges::range_reference_t<_OutRange>;
    using _InRefType = std::ranges::range_reference_t<_InRange>;

    assert(std::ranges::size(__in_r) == std::ranges::size(__out_r));
    const auto __n = std::ranges::size(__in_r);

    if constexpr (std::is_trivially_constructible_v<_OutValueType, _InRefType> && // required operation is trivial
                  std::is_trivially_default_constructible_v<_OutValueType> &&     // actual operations are trivial
                  std::is_trivially_assignable_v<_OutRefType, _InRefType>)
    {
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
        oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
        oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_copy<std::decay_t<_ExecutionPolicy>>{},
            oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
            oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
    }

    return {std::ranges::begin(__in_r) + __n, std::ranges::begin(__out_r) + __n};
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

    assert(std::ranges::size(__in_r) == std::ranges::size(__out_r));
    const auto __n = std::ranges::size(__in_r);

    if constexpr (std::is_trivially_constructible_v<_OutValueType, std::remove_reference_t<_InRefType>&&> &&
                  std::is_trivially_default_constructible_v<_OutValueType> &&
                  std::is_trivially_assignable_v<_OutRefType, _InRefType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
            oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_move<std::decay_t<_ExecutionPolicy>>{},
            oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
            oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
    }

    return {std::ranges::begin(__in_r) + __n, std::ranges::begin(__out_r) + __n};
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_uninitialized_fill
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _T>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_fill(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, const _T& __value)
{
    using _ValueType = std::ranges::range_value_t<_R>;

    const auto __first = std::ranges::begin(__r);
    const auto __last = __first + std::ranges::size(__r);

    if constexpr (std::is_trivially_constructible_v<_ValueType, _T> && // required operation is trivial
                  std::is_trivially_default_constructible_v<_ValueType> && // actual operations are trivial
                  std::is_trivially_copy_assignable_v<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__brick_fill<__hetero_tag<_BackendTag>, _ValueType>{_ValueType(__value)},
            oneapi::dpl::__ranges::views::all_write(std::forward<_R>(__r)));
    }
    else
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_uninitialized_fill<_T, std::decay_t<_ExecutionPolicy>>{__value},
            oneapi::dpl::__ranges::views::all_write(std::forward<_R>(__r)));
    }

    return std::ranges::borrowed_iterator_t<_R>{__last};
}

//---------------------------------------------------------------------------------------------------------------------
// destroy
//---------------------------------------------------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_destroy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    using _ValueType = typename std::ranges::range_value_t<_R>;

    if constexpr (!std::is_trivially_destructible_v<_ValueType>)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__op_destroy<std::decay_t<_ExecutionPolicy>>{},
            oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));
    }

    return std::ranges::borrowed_iterator_t<_R>{std::ranges::end(__r)};
}

} // namespace oneapi::dpl::__internal::__ranges

#endif // _ONEDPL_CPP20_RANGES_PRESENT
#endif // _ONEDPL_MEMORY_RANGES_IMPL_HETERO_H