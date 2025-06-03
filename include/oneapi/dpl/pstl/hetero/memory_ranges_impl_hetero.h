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

#include "../parallel_backend.h"
#include "../utils_ranges.h"
#include "../memory_impl.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#endif

#include <ranges>
#include <utility>
#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_default_construct(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    const auto __first = std::ranges::begin(__r);

    oneapi::dpl::__internal::__op_uninitialized_default_construct<_ExecutionPolicy> __f;

    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag,
        std::forward<_ExecutionPolicy>(__exec), __f, oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));

    return {__first + __res};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
std::ranges::borrowed_iterator_t<_R>
__pattern_uninitialized_value_construct(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    const auto __first = std::ranges::begin(__r);

    oneapi::dpl::__internal::__op_uninitialized_value_construct<_ExecutionPolicy> __f;

    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag,
        std::forward<_ExecutionPolicy>(__exec), __f, oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));

    return {__first + __res};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_copy_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    assert(std::ranges::size(__in_r) == std::ranges::size(__out_r));

    const auto __first1 = std::ranges::begin(__in_r);
    const auto __first2 = std::ranges::begin(__out_r);

    oneapi::dpl::__internal::__op_uninitialized_copy<_ExecutionPolicy> __f;

    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag,
        std::forward<_ExecutionPolicy>(__exec), __f, oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
        oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));

    return {__first1 + __res, __first2 + __res};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
std::ranges::uninitialized_move_result<std::ranges::borrowed_iterator_t<_InRange>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_uninitialized_move(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    assert(std::ranges::size(__in_r) == std::ranges::size(__out_r));

    const auto __first1 = std::ranges::begin(__in_r);
    const auto __first2 = std::ranges::begin(__out_r);

    oneapi::dpl::__internal::__op_uninitialized_move<_ExecutionPolicy> __f;

    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag,
        std::forward<_ExecutionPolicy>(__exec), __f, oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
        oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));

    return {__first1 + __res, __first2 + __res};
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_MEMORY_RANGES_IMPL_HETERO_H
