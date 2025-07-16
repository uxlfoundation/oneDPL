
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

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>


#include "../utils.h"
#include "../../../../pstl/utils.h"
#include "../../../../pstl/hetero/dpcpp/unseq_backend_sycl.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

// TODO: do we want our own implementations here or just use onedpl sycl backend?
template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__inclusive_sub_group_masked_scan(const sycl::sub_group& __sub_group, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
#pragma unroll
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    //if constexpr (__init_present)
    //{
    __value = __binary_op(__init_and_carry, __value);
    __init_and_carry = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    //}
    //else
    //{
    //    __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    //}
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _MaskOp,
          typename _InitBroadcastId, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__sub_group_masked_scan(const sycl::sub_group& __sub_group, _MaskOp __mask_fn, _InitBroadcastId __init_broadcast_id,
                        _ValueType& __value, _BinaryOp __binary_op, _LazyValueType& __init_and_carry)
{
    if constexpr (__is_inclusive)
    {
        __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
    //else
    //{
    //    __exclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
    //                                                                        __value, __binary_op, __init_and_carry);
    //}
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

//template <int sg_size, int iters_per_item, typename ArrayOrder, typename BinaryOperation>
//auto
//work_group_scan(ArrayOrder, input[iters_per_item], output[iters_per_item], BinaryOperation binary_op, int num_remaining);
template <int sub_group_size, int iters_per_item, typename InputType, /*typename OutputType,*/
          typename SubGroup, typename ArrayOrder, typename BinaryOperation>
auto
sub_group_scan(ArrayOrder, const SubGroup& sub_group, InputType input[iters_per_item],
               /*OutputType output[iters_per_item],*/ BinaryOperation binary_op, uint32_t items_in_scan)
{
    const bool is_full = items_in_scan == sub_group_size * iters_per_item;
    if constexpr (std::is_same_v<ArrayOrder, item_array_order::sub_group_stride>)
    {
        InputType carry = oneapi::dpl::unseq_backend::__known_identity<BinaryOperation, InputType>;
        if (is_full)
        {
#pragma unroll
            for (int i = 0; i < iters_per_item; ++i)
            {
                __sub_group_scan<sub_group_size, true, true>(sub_group, input[i], binary_op, carry);
            }
        }
        else
        {
            const auto limited_iters_per_item =
                oneapi::dpl::__internal::__dpl_ceiling_div(items_in_scan, sub_group_size);
            int i = 0;
#pragma unroll
            for (; i < limited_iters_per_item - 1; ++i)
            {
                __sub_group_scan<sub_group_size, true, true>(sub_group, input[i], binary_op, carry);
            }
            __sub_group_scan_partial<sub_group_size, true, true>(sub_group, input[i], binary_op, carry,
                                                                 items_in_scan - i * iters_per_item * sub_group_size);
        }
        return carry;
    }
    else
    {
        static_assert(false, "Current strategy unsupported");
    }
}

template <int sub_group_size, int iters_per_item, typename InputType, /*typename OutputType,*/
          typename SubGroup, typename ArrayOrder, typename BinaryOperation>
auto
sub_group_scan(ArrayOrder, const SubGroup& sub_group, InputType input[iters_per_item],
               /*OutputType output[iters_per_item],*/ BinaryOperation binary_op)
{
    return sub_group_scan<sub_group_size, iters_per_item>(ArrayOrder{}, sub_group, input, binary_op,
                                                          sub_group_size * iters_per_item);
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
