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

#ifndef _ONEDPL_GLUE_MEMORY_RANGES_IMPL_H
#define _ONEDPL_GLUE_MEMORY_RANGES_IMPL_H

#if _ONEDPL_CPP20_RANGES_PRESENT

#include <utility>
#include <ranges>
#include <functional>
#include <type_traits>
#include <iterator>

#include <concepts> // for std::same_as, std::constructible_from, std::destructible, std::default_initializable

#include "execution_defs.h"
#include "oneapi/dpl/pstl/ranges_defs.h"

#include "memory_ranges_impl.h"

namespace oneapi
{
namespace dpl
{
namespace ranges
{

namespace __internal
{

template<typename _I>
concept __nothrow_random_access_iterator =
    std::random_access_iterator<_I> && std::is_lvalue_reference_v<std::iter_reference_t<_I>> &&
    std::same_as<std::remove_cvref_t<std::iter_reference_t<_I>>, std::iter_value_t<_I>>;

template<typename _S, typename _I>
concept __nothrow_sentinel_for = std::sentinel_for<_S, _I>;

template<typename _R>
concept __nothrow_random_access_range =
    std::ranges::range<_R> && __nothrow_random_access_iterator<std::ranges::iterator_t<_R>> &&
    __nothrow_sentinel_for<std::ranges::sentinel_t<_R>, std::ranges::iterator_t<_R>>;
}

namespace __internal
{

struct __uninitialized_default_construct_fn
{
    template<typename _ExecutionPolicy, oneapi::dpl::ranges::__internal::__nothrow_random_access_range _R>
    requires std::default_initializable<std::ranges::range_value_t<_R>>
        && oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_R>

    std::ranges::borrowed_iterator_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        return oneapi::dpl::__internal::__ranges::__pattern_uninitialized_default_construct(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r));
    }
}; //__uninitialized_default_construct_fn
}  //__internal

inline constexpr __internal::__uninitialized_default_construct_fn uninitialized_default_construct;

namespace __internal
{
struct __uninitialized_value_construct_fn
{
    template<typename _ExecutionPolicy, oneapi::dpl::ranges::__internal::__nothrow_random_access_range _R>
    requires std::default_initializable<std::ranges::range_value_t<_R>>
        && oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_R>

    std::ranges::borrowed_iterator_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        return oneapi::dpl::__internal::__ranges::__pattern_uninitialized_value_construct(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r));
    }
}; //__uninitialized_value_construct_fn
}  //__internal

inline constexpr __internal::__uninitialized_value_construct_fn uninitialized_value_construct;

namespace __internal
{

struct __uninitialized_copy_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _InRange,
             oneapi::dpl::ranges::__internal::__nothrow_random_access_range _OutRange>
    requires std::constructible_from<std::ranges::range_value_t<_OutRange>, std::ranges::range_reference_t<_InRange>>
        && oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_InRange> && std::ranges::sized_range<_OutRange>

    std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<_InRange>,
                                           std::ranges::borrowed_iterator_t<_OutRange>>
    operator()(_ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        using _Size = std::common_type_t<std::ranges::range_size_t<_InRange>, std::ranges::range_size_t<_OutRange>>;
        const _Size __size = std::ranges::min((_Size)std::ranges::size(__in_r), (_Size)std::ranges::size(__out_r));

        oneapi::dpl::__internal::__ranges::__pattern_uninitialized_copy(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::ranges::take_view(__in_r, __size),
            std::ranges::take_view(__out_r, __size));

        return {std::ranges::begin(__in_r) + __size, std::ranges::begin(__out_r) +  __size};
    }
}; //__uninitialized_copy_fn
}  //__internal

inline constexpr __internal::__uninitialized_copy_fn uninitialized_copy;

namespace __internal
{

struct __uninitialized_move_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _InRange,
             oneapi::dpl::ranges::__internal::__nothrow_random_access_range _OutRange>
    requires std::constructible_from<std::ranges::range_value_t<_OutRange>,
                                     std::ranges::range_rvalue_reference_t<_InRange>>
        && oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_InRange> && std::ranges::sized_range<_OutRange>

    std::ranges::uninitialized_move_result<std::ranges::borrowed_iterator_t<_InRange>,
                                           std::ranges::borrowed_iterator_t<_OutRange>>
    operator()(_ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        using _Size = std::common_type_t<std::ranges::range_size_t<_InRange>, std::ranges::range_size_t<_OutRange>>;
        const _Size __size = std::ranges::min((_Size)std::ranges::size(__in_r), (_Size)std::ranges::size(__out_r));

        oneapi::dpl::__internal::__ranges::__pattern_uninitialized_move(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::ranges::take_view(__in_r, __size),
            std::ranges::take_view(__out_r, __size));

        return {std::ranges::begin(__in_r) + __size, std::ranges::begin(__out_r) +  __size};
    }
}; //__uninitialized_move_fn
}  //__internal

inline constexpr __internal::__uninitialized_move_fn uninitialized_move;

namespace __internal
{

struct __uninitialized_fill_fn
{
    template<typename _ExecutionPolicy, oneapi::dpl::ranges::__internal::__nothrow_random_access_range _R, typename _T>
    requires std::constructible_from<std::ranges::range_value_t<_R>, const _T&>
        && oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_R>

    std::ranges::borrowed_iterator_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r, const _T& __value) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        return oneapi::dpl::__internal::__ranges::__pattern_uninitialized_fill(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __value);
    }
}; //__uninitialized_fill_fn
}  //__internal

inline constexpr __internal::__uninitialized_fill_fn uninitialized_fill;

namespace __internal
{

struct __destroy_fn
{
    template<typename _ExecutionPolicy, oneapi::dpl::ranges::__internal::__nothrow_random_access_range _R>
    requires std::destructible<std::ranges::range_value_t<_R>>
        && oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_R>

    std::ranges::borrowed_iterator_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        return oneapi::dpl::__internal::__ranges::__pattern_destroy(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r));
    }
}; //__destroy_fn
}  //__internal

inline constexpr __internal::__destroy_fn destroy;

} // ranges
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_CPP20_RANGES_PRESENT
#endif // _ONEDPL_GLUE_MEMORY_RANGES_IMPL_H
