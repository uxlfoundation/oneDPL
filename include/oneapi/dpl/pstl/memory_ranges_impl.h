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

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_H

#if _ONEDPL_CPP20_RANGES_PRESENT

#include <ranges>
#include <utility>
#include <cassert>
#include <functional>
#include <type_traits>

#include "execution_impl.h"

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

template <typename _Tag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_default_construct(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    using value_type = std::ranges::range_value_t<_R>;

    const auto __first = std::ranges::begin(__r);
    const auto __last = __first + std::ranges::size(__r);

    oneapi::dpl::__internal::__pattern_walk1(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
        [](auto& __obj) { ::new (std::addressof(__obj)) value_type;});

    return {__last};
}

template <typename _ExecutionPolicy, typename _R>
std::borrowed_iterator_t<_R>
__pattern_uninitialized_default_construct(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r)
{
    return std::ranges::uninitialized_default_construct(std::forward<_R>(__r));
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_value_construct
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_value_construct(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    using value_type = std::ranges::range_value_t<_R>;

    const auto __first = std::ranges::begin(__r);
    const auto __last = __first + std::ranges::size(__r);

    oneapi::dpl::__internal::__pattern_walk1(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
        [](auto& __obj) { ::new (static_cast<void*>(std::addressof(__obj))) value_type();});

    return {__last};
}

template <typename _ExecutionPolicy, typename _R>
std::borrowed_iterator_t<_R>
__pattern_uninitialized_value_construct(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r)
{
    return std::ranges::uninitialized_value_construct(std::forward<_R>(__r));
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_uninitialized_copy
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_copy(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    assert(std::ranges::size(__in_r) <= std::ranges::size(__out_r));

    using value_type = std::ranges::range_value_t<_OutRange>;

    const auto __first1 = std::ranges::begin(__in_r);
    const auto __first2 = std::ranges::begin(__out_r);
    const auto __size = __first + std::ranges::size(__in_r);

    const auto __last1 = __first1 + __size;
    const auto __last2 = __first2 + __size;

    oneapi::dpl::__internal::__pattern_walk2(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
        __first2, [](auto&& __src, auto& __target) {
            ::new (static_cast<void*>(std::addressof(__target))) value_type(std::forward<decltype(__src)>(__src));
            });

    return {__last1, __last2};
}

template<typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_copy(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _InRange&& __in_r, _OutRange&& __out_r)
{
    return std::ranges::uninitialized_copy(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r));
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H
