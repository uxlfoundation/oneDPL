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
#include <type_traits>

#include "../../../../pstl/utils.h"
#include "../../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../../pstl/hetero/dpcpp/unseq_backend_sycl.h"
#include "../../../../pstl/hetero/dpcpp/parallel_backend_sycl.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

template <typename _T>
struct __is_lazy_ctor_storage : std::false_type
{
};

template <typename _T>
struct __is_lazy_ctor_storage<oneapi::dpl::__internal::__lazy_ctor_storage<_T>> : std::true_type
{
};

template <typename _T>
struct __scan_input
{
    using type = _T;
};

template <typename _T>
struct __scan_input<oneapi::dpl::__internal::__lazy_ctor_storage<_T>>
{
    using type = _T;
};

template <typename _T>
using __scan_input_t = typename __scan_input<std::decay_t<_T>>::type;

template <typename _T>
decltype(auto)
__extract_scan_input(_T&& __value)
{
    if constexpr (__is_lazy_ctor_storage<std::decay_t<_T>>::value)
        return (__value.__v);
    else
        return (__value);
}

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType>
void
__exclusive_sub_group_masked_scan(const sycl::nd_item<1>& __ndi, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  oneapi::dpl::__internal::__lazy_ctor_storage<_ValueType>& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __ndi.get_sub_group().get_local_linear_id();
    for (std::uint8_t __shift = 1; __shift < __sub_group_size; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__ndi.get_sub_group(), __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    oneapi::dpl::__internal::__lazy_ctor_storage<_ValueType> __old_init;
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        if (__sub_group_local_id == 0)
            __old_init.__setup(__init_and_carry.__v);
        __init_and_carry.__v = sycl::group_broadcast(__ndi.get_sub_group(), __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__ndi.get_sub_group(), __value, __init_broadcast_id));
    }

    __value = sycl::shift_group_right(__ndi.get_sub_group(), __value, 1);
    if constexpr (__init_present)
    {
        if (__sub_group_local_id == 0)
        {
            __value = __old_init.__v;
            __old_init.__destroy();
        }
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType>
void
__inclusive_sub_group_masked_scan(const sycl::nd_item<1>& __ndi, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  oneapi::dpl::__internal::__lazy_ctor_storage<_ValueType>& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __ndi.get_sub_group().get_local_linear_id();
    for (std::uint8_t __shift = 1; __shift < __sub_group_size; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__ndi.get_sub_group(), __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        __init_and_carry.__v = sycl::group_broadcast(__ndi.get_sub_group(), __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__ndi.get_sub_group(), __value, __init_broadcast_id));
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _MaskOp,
          typename _InitBroadcastId, typename _BinaryOp, typename _ValueType>
void
__sub_group_masked_scan(const sycl::nd_item<1>& __ndi, _MaskOp __mask_fn, _InitBroadcastId __init_broadcast_id,
                        _ValueType& __value, _BinaryOp __binary_op,
                        oneapi::dpl::__internal::__lazy_ctor_storage<_ValueType>& __init_and_carry)
{
    if constexpr (__is_inclusive)
    {
        __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__ndi, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
    else
    {
        __exclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__ndi, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType>
void
__single_sub_group_scan(const sycl::nd_item<1>& __ndi, _ValueType& __value, _BinaryOp __binary_op,
                        oneapi::dpl::__internal::__lazy_ctor_storage<_ValueType>& __init_and_carry)
{
    auto __mask_fn = [](auto __sub_group_local_id, auto __offset) { return __sub_group_local_id >= __offset; };
    std::uint8_t __init_broadcast_id = __sub_group_size - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(__ndi, __mask_fn, __init_broadcast_id,
                                                                              __value, __binary_op, __init_and_carry);
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType>
void
__single_sub_group_scan_partial(const sycl::nd_item<1>& __ndi, _ValueType& __value, _BinaryOp __binary_op,
                                oneapi::dpl::__internal::__lazy_ctor_storage<_ValueType>& __init_and_carry,
                                std::uint32_t __elements_to_process)
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    std::uint8_t __init_broadcast_id = __elements_to_process - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(__ndi, __mask_fn, __init_broadcast_id,
                                                                              __value, __binary_op, __init_and_carry);
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
template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputTypeWrapped,
          typename _BinaryOperation>
auto
__sub_group_scan(const sycl::nd_item<1>& __ndi, _InputTypeWrapped __input[__iters_per_item],
                 _BinaryOperation __binary_op, std::uint32_t __items_in_scan)
{
    using _InputType = __scan_input_t<_InputTypeWrapped>;
    const bool __is_full = __items_in_scan == __sub_group_size * __iters_per_item;
    oneapi::dpl::__internal::__lazy_ctor_storage<_InputType> __carry;
    oneapi::dpl::__internal::__scoped_destroyer<_InputType> __destroy_when_leaving_scope{__carry};
    if (__is_full)
    {
        __single_sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ false>(
            __ndi, __extract_scan_input(__input[0]), __binary_op, __carry);
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 1; __i < __iters_per_item; ++__i)
        {
            __single_sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ true>(
                __ndi, __extract_scan_input(__input[__i]), __binary_op, __carry);
        }
    }
    else
    {
        const std::uint16_t __limited_iters_per_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__items_in_scan, __sub_group_size);
        std::uint16_t __i = 0;
        if (__limited_iters_per_item == 1)
        {
            __single_sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ false>(
                __ndi, __extract_scan_input(__input[__i]), __binary_op, __carry,
                __items_in_scan - __i * __sub_group_size);
        }
        else if (__limited_iters_per_item > 1)
        {
            __single_sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ false>(
                __ndi, __extract_scan_input(__input[__i++]), __binary_op, __carry);
            for (; __i < __limited_iters_per_item - 1; ++__i)
            {
                __single_sub_group_scan<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ true>(
                    __ndi, __extract_scan_input(__input[__i]), __binary_op, __carry);
            }
            __single_sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true, /*__init_present*/ true>(
                __ndi, __extract_scan_input(__input[__i]), __binary_op, __carry,
                __items_in_scan - __i * __sub_group_size);
        }
    }
    return __carry.__v;
}

template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _BinaryOperation>
_InputType
__sub_group_scan(const sycl::nd_item<1>& __ndi, _InputType __input[__iters_per_item], _BinaryOperation __binary_op)
{
    return __sub_group_scan<__sub_group_size, __iters_per_item>(__ndi, __input, __binary_op,
                                                                __sub_group_size * __iters_per_item);
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
