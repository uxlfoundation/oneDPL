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

#ifndef _ONEDPL_KT_WG_SCAN_H
#define _ONEDPL_KT_WG_SCAN_H

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>

#include "../../../../pstl/utils.h"
#include "../sub_group/sub_group_scan.h"
#include "../../../../pstl/hetero/dpcpp/unseq_backend_sycl.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

struct __no_init_callback
{
    template <typename... _Args>
    void operator()(_Args&&...)
    {
    }
};

template <typename _InitCallbackFn>
struct __init_callback_fn
{
    template <typename _Subgroup, typename _T> 
    auto operator()(const _Subgroup& __sub_group, _T __wg_carry)
    {
        return __init_callback(__sub_group, __wg_carry);
    }
    _InitCallbackFn __init_callback;
};

// TODO: consider adding init callback
template <int __sub_group_size, int __iters_per_item, typename _InputType, typename _NdItem, typename _SlmAcc,
          typename _ProcessInitCallback, typename _BinaryOperation>
auto
__work_group_scan_impl(const _NdItem& __item, _SlmAcc __local_acc, _InputType __input[__iters_per_item],
                       _BinaryOperation __binary_op, _ProcessInitCallback __process_init_callback, uint32_t __items_in_scan)
{
    constexpr bool __b_init_callback = !std::is_same_v<__no_init_callback, _ProcessInitCallback>;

    auto __sub_group = __item.get_sub_group();
    auto __sub_group_carry = __sub_group_scan<__sub_group_size, __iters_per_item>(__sub_group, __input, __binary_op);
    const std::uint8_t __sub_group_group_id = __sub_group.get_group_linear_id();
    const std::uint8_t __active_sub_groups =
        oneapi::dpl::__internal::__dpl_ceiling_div(__items_in_scan, __sub_group_size * __iters_per_item);
    // When there is no init callback, the compiler can just optimize out this variable.
    [[maybe_unused]] _InputType __wg_init;
    if (__sub_group.get_local_linear_id() == __sub_group_size - 1)
    {
        __local_acc[__sub_group.get_group_linear_id()] = __sub_group_carry;
    }
    sycl::group_barrier(__item.get_group());
    if (__sub_group_group_id == 0)
    {
        const auto __num_iters = oneapi::dpl::__internal::__dpl_ceiling_div(__active_sub_groups, __sub_group_size);
        _InputType __wg_carry{};
        auto __idx = __sub_group.get_local_linear_id();
        auto __val = __local_acc[__idx];
        if (__num_iters == 1)
        {
            __sub_group_scan_partial<__sub_group_size, true, false>(__sub_group, __val, __binary_op, __wg_carry,
                                                                    __active_sub_groups);
            __local_acc[__idx] = __val;
        }
        else
        {
            __sub_group_scan<__sub_group_size, true, false>(__sub_group, __val, __binary_op, __wg_carry);
            __local_acc[__idx] = __val;
            __idx += __sub_group_size;
            for (int __i = 1; __i < __num_iters - 1; ++__i)
            {
                __val = __local_acc[__idx];
                __sub_group_scan<__sub_group_size, true, true>(__sub_group, __val, __binary_op, __wg_carry);
                __local_acc[__idx] = __val;
                __idx += __sub_group_size;
            }
            __val = __local_acc[__idx];
            __sub_group_scan_partial<__sub_group_size, true, true>(__sub_group, __val, __binary_op, __wg_carry,
                                                                   __active_sub_groups -
                                                                       (__num_iters - 1) * __sub_group_size);
            __local_acc[__idx] = __val;
        }
        if constexpr (__b_init_callback)
            __wg_init = __process_init_callback(__sub_group, __wg_carry);
    }
    // TODO: cleaner logic
    sycl::group_barrier(__item.get_group());
    if constexpr (__b_init_callback)
    {
        __wg_init = sycl::group_broadcast(__item.get_group(), __wg_init);
        if (__sub_group_group_id < __active_sub_groups)
        {
            _InputType __sub_group_carry_in = (__sub_group_group_id == 0) ? __wg_init
                : __binary_op(__wg_init, sycl::group_broadcast(__sub_group, __local_acc[__sub_group_group_id - 1]));
            for (int __i = 0; __i < __iters_per_item; ++__i)
                __input[__i] = __binary_op(__sub_group_carry_in, __input[__i]);
        }
    }
    else
    {
        if (__sub_group_group_id > 0 && __sub_group_group_id < __active_sub_groups)
        {
            _InputType __sub_group_carry_in = sycl::group_broadcast(__sub_group, __local_acc[__sub_group_group_id - 1]);
            for (int __i = 0; __i < __iters_per_item; ++__i)
                __input[__i] = __binary_op(__sub_group_carry_in, __input[__i]);
        }
    }
    return __local_acc[__active_sub_groups - 1];
}

template <int __sub_group_size, int __iters_per_item, typename _InputType, typename _NdItem, typename _SlmAcc,
          typename _InitCallback, typename _BinaryOperation>
auto
__work_group_scan(const _NdItem& __item, _SlmAcc __local_acc, _InputType __input[__iters_per_item],
                  _BinaryOperation __binary_op, _InitCallback __init_callback, uint32_t __items_in_scan)
{
    return __work_group_scan_impl<__sub_group_size, __iters_per_item>(__item, __local_acc, __input, __binary_op,
                                                                      __init_callback_fn<_InitCallback>{__init_callback}, __items_in_scan);

}

template <int __sub_group_size, int __iters_per_item, typename _InputType, typename _NdItem, typename _SlmAcc,
          typename _BinaryOperation>
auto
__work_group_scan(const _NdItem& __item, _SlmAcc __local_acc, _InputType __input[__iters_per_item],
                  _BinaryOperation __binary_op, uint32_t __items_in_scan)
{
    return __work_group_scan_impl<__sub_group_size, __iters_per_item>(__item, __local_acc, __input, __binary_op,
                                                                     __no_init_callback{}, __items_in_scan);

}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt
#endif
