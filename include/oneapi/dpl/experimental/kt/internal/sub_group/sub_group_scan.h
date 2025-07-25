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

#ifndef _ONEDPL_KT_SUB_GROUP_SCAN_H
#define _ONEDPL_KT_SUB_GROUP_SCAN_H

#include <cstdint>

#include "../../../../pstl/utils.h"
#include "../../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../../pstl/hetero/dpcpp/unseq_backend_sycl.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

// This implementation models what is defined in pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h with the
// default constructibility requirement removed for simplification for the types supported in the KT.
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
        __value = __binary_op(__init_and_carry, __value);
        __init_and_carry = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _MaskOp,
          typename _InitBroadcastId, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__sub_group_masked_scan(const sycl::sub_group& __sub_group, _MaskOp __mask_fn, _InitBroadcastId __init_broadcast_id,
                        _ValueType& __value, _BinaryOp __binary_op, _LazyValueType& __init_and_carry)
{
    static_assert(__is_inclusive, "__sub_group_masked_scan is only currently supported for inclusive scans.");
    __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                        __value, __binary_op, __init_and_carry);
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType>
void
__sub_group_scan(const sycl::sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
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
__sub_group_scan_partial(const sycl::sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                         _LazyValueType& __init_and_carry, _SizeType __elements_to_process)
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    std::uint8_t __init_broadcast_id = __elements_to_process - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry);
}

//
// An optimized scan in a sycl::sub_group performed in local registers.
// Input is accepted in the form of an array in sub-group strided order. Formally, for some index i in __input,
// __input[i] must correspond to position
//      (i * sg_sz + sg_lid)
// in the desired sub-group scan where sg_sz is the size of the sub-group and sg_lid is the local offset of an item in
// the sub-group. This layout is to align with optimal loads from global memory without extra data movement.
// The scan results are updated in __input.
//
template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _SubGroup,
          typename _BinaryOperation>
_InputType
__sub_group_scan(const _SubGroup& __sub_group, _InputType __input[__iters_per_item], _BinaryOperation __binary_op,
                 uint32_t __items_in_scan)
{
    const bool __is_full = __items_in_scan == __sub_group_size * __iters_per_item;
    _InputType __carry{};
    if (__is_full)
    {
        __sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ false>(__sub_group, __input[0],
                                                                                              __binary_op, __carry);
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 1; __i < __iters_per_item; ++__i)
        {
            __sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ true>(
                __sub_group, __input[__i], __binary_op, __carry);
        }
    }
    else
    {
        const std::uint16_t __limited_iters_per_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__items_in_scan, __sub_group_size);
        std::uint16_t __i = 0;
        if (__limited_iters_per_item == 1)
        {
            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ false>(
                __sub_group, __input[__i], __binary_op, __carry,
                __items_in_scan - __i * __iters_per_item * __sub_group_size);
        }
        else
        {
            __sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ false>(
                __sub_group, __input[__i++], __binary_op, __carry);
            for (; __i < __limited_iters_per_item - 1; ++__i)
            {
                __sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ true>(
                    __sub_group, __input[__i], __binary_op, __carry);
            }
            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ true>(
                __sub_group, __input[__i], __binary_op, __carry,
                __items_in_scan - __i * __iters_per_item * __sub_group_size);
        }
    }
    return __carry;
}

template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _SubGroup,
          typename _BinaryOperation>
_InputType
__sub_group_scan(const _SubGroup& __sub_group, _InputType __input[__iters_per_item], _BinaryOperation __binary_op)
{
    return __sub_group_scan<__sub_group_size, __iters_per_item>(__sub_group, __input, __binary_op,
                                                                __sub_group_size * __iters_per_item);
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
