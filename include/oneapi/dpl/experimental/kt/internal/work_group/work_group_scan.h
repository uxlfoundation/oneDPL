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

#include <cstdint>
#include <type_traits>

#include "../../../../pstl/utils.h"
#include "../../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../sub_group/sub_group_scan.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

struct __no_init_callback
{
};

template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _NdItem,
          typename _SlmAcc, typename _ProcessInitCallback, typename _BinaryOperation>
_InputType
__work_group_scan_impl(const _NdItem& __item, _SlmAcc __local_acc, _InputType __input[__iters_per_item],
                       _BinaryOperation __binary_op, _ProcessInitCallback __process_init_callback,
                       std::uint32_t __items_in_scan)
{
    constexpr bool __b_init_callback = !std::is_same_v<__no_init_callback, _ProcessInitCallback>;

    sycl::sub_group __sub_group = __item.get_sub_group();
    _InputType __sub_group_carry =
        __sub_group_scan<__sub_group_size, __iters_per_item>(__sub_group, __input, __binary_op);
    const std::uint8_t __sub_group_group_id = __sub_group.get_group_linear_id();
    const std::uint8_t __active_sub_groups =
        oneapi::dpl::__internal::__dpl_ceiling_div(__items_in_scan, __sub_group_size * __iters_per_item);
    // When there is no init callback, the compiler can just optimize out this variable.
    [[maybe_unused]] _InputType __wg_init;
    if (__sub_group.get_local_linear_id() == __sub_group_size - 1)
    {
        __local_acc[__sub_group.get_group_linear_id()] = __sub_group_carry;
    }
    __dpl_sycl::__group_barrier(__item);
    if (__sub_group_group_id == 0)
    {
        const std::uint8_t __num_iters =
            oneapi::dpl::__internal::__dpl_ceiling_div(__active_sub_groups, __sub_group_size);
        _InputType __wg_carry{};
        std::uint8_t __idx = __sub_group.get_local_linear_id();
        _InputType __val = __local_acc[__idx];
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
            for (std::uint8_t __i = 1; __i < __num_iters - 1; ++__i)
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
    __dpl_sycl::__group_barrier(__item);
    if constexpr (__b_init_callback)
    {
        __wg_init = __dpl_sycl::__group_broadcast(__item.get_group(), __wg_init);
        if (__sub_group_group_id < __active_sub_groups)
        {
            _InputType __sub_group_carry_in =
                (__sub_group_group_id == 0)
                    ? __wg_init
                    : __binary_op(__wg_init,
                                  __dpl_sycl::__group_broadcast(__sub_group, __local_acc[__sub_group_group_id - 1]));
            for (std::uint16_t __i = 0; __i < __iters_per_item; ++__i)
                __input[__i] = __binary_op(__sub_group_carry_in, __input[__i]);
        }
    }
    else
    {
        if (__sub_group_group_id > 0 && __sub_group_group_id < __active_sub_groups)
        {
            _InputType __sub_group_carry_in =
                __dpl_sycl::__group_broadcast(__sub_group, __local_acc[__sub_group_group_id - 1]);
            for (std::uint16_t __i = 0; __i < __iters_per_item; ++__i)
                __input[__i] = __binary_op(__sub_group_carry_in, __input[__i]);
        }
    }
    return __local_acc[__active_sub_groups - 1];
}

template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _NdItem,
          typename _SlmAcc, typename _InitCallback, typename _BinaryOperation>
_InputType
__work_group_scan(const _NdItem& __item, _SlmAcc __local_acc, _InputType __input[__iters_per_item],
                  _BinaryOperation __binary_op, _InitCallback __init_callback, std::uint32_t __items_in_scan)
{
    return __work_group_scan_impl<__sub_group_size, __iters_per_item>(__item, __local_acc, __input, __binary_op,
                                                                      __init_callback, __items_in_scan);
}

template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _NdItem,
          typename _SlmAcc, typename _BinaryOperation>
_InputType
__work_group_scan(const _NdItem& __item, _SlmAcc __local_acc, _InputType __input[__iters_per_item],
                  _BinaryOperation __binary_op, std::uint32_t __items_in_scan)
{
    return __work_group_scan_impl<__sub_group_size, __iters_per_item>(__item, __local_acc, __input, __binary_op,
                                                                      __no_init_callback{}, __items_in_scan);
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
