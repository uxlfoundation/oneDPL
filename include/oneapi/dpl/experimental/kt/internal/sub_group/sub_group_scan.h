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
#include "../../../../pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

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
__sub_group_scan(const _SubGroup& __sub_group,
                 oneapi::dpl::__internal::__lazy_ctor_storage<_InputType> __input[__iters_per_item],
                 _BinaryOperation __binary_op, std::uint32_t __items_in_scan)
{
    const bool __is_full = __items_in_scan == __sub_group_size * __iters_per_item;
    oneapi::dpl::__internal::__lazy_ctor_storage<_InputType> __carry;
    oneapi::dpl::__internal::__scoped_destroyer<_InputType> __destroy_when_leaving_scope{__carry};
    if (__is_full)
    {
        oneapi::dpl::__par_backend_hetero::__sub_group_scan<__sub_group_size, /*__is_inclusive*/ true,
                                                            /*__init_present*/ false>(__sub_group, __input[0].__v,
                                                                                      __binary_op, __carry);
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 1; __i < __iters_per_item; ++__i)
        {
            oneapi::dpl::__par_backend_hetero::__sub_group_scan<__sub_group_size, /*__is_inclusive*/ true,
                                                                /*__init_present*/ true>(__sub_group, __input[__i].__v,
                                                                                         __binary_op, __carry);
        }
    }
    else
    {
        const std::uint16_t __limited_iters_per_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__items_in_scan, __sub_group_size);
        std::uint16_t __i = 0;
        if (__limited_iters_per_item == 1)
        {
            oneapi::dpl::__par_backend_hetero::__sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true,
                                                                        /*__init_present*/ false>(
                __sub_group, __input[__i].__v, __binary_op, __carry, __items_in_scan - __i * __sub_group_size);
        }
        else if (__limited_iters_per_item > 1)
        {
            oneapi::dpl::__par_backend_hetero::__sub_group_scan<__sub_group_size, /*__is_inclusive*/ true,
                                                                /*__init_present*/ false>(
                __sub_group, __input[__i++].__v, __binary_op, __carry);
            for (; __i < __limited_iters_per_item - 1; ++__i)
            {
                oneapi::dpl::__par_backend_hetero::__sub_group_scan<__sub_group_size, /*__is_inclusive*/ true,
                                                                    /*__init_present*/ true>(
                    __sub_group, __input[__i].__v, __binary_op, __carry);
            }
            oneapi::dpl::__par_backend_hetero::__sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true,
                                                                        /*__init_present*/ true>(
                __sub_group, __input[__i].__v, __binary_op, __carry, __items_in_scan - __i * __sub_group_size);
        }
    }
    return __carry.__v;
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
