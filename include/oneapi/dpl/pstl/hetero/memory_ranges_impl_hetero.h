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
__pattern_uninitialized_default_construct(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _R&& __r)
{
    const auto __first = std::ranges::begin(__r);
    const auto __size = std::ranges::size(__r);

    oneapi::dpl::__internal::__op_uninitialized_default_construct<_ExecutionPolicy> __f;

    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag,
        std::forward<_ExecutionPolicy>(__exec), __f, oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));

    return {__first + __size};
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_MEMORY_RANGES_IMPL_HETERO_H
