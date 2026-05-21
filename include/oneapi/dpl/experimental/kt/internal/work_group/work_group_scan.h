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
#include "../../../../pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h"
#include "../sub_group/sub_group_scan.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _NdItem,
          typename _SlmAcc, typename _BinaryOperation, typename _InitCallback>
void
__work_group_scan_impl(const _NdItem& __item, _SlmAcc __local_acc,
                       oneapi::dpl::__internal::__lazy_ctor_storage<_InputType> __input[__iters_per_item],
                       _BinaryOperation __binary_op, _InitCallback __init_callback, std::uint32_t __items_in_scan)
{
    sycl::sub_group __sub_group = __item.get_sub_group();
    const std::uint8_t __sub_group_group_id = __sub_group.get_group_linear_id();
    const std::uint8_t __active_sub_groups =
        oneapi::dpl::__internal::__dpl_ceiling_div(__items_in_scan, __sub_group_size * __iters_per_item);

    const std::uint32_t __items_in_sub_group_scan = __sub_group_size * __iters_per_item;
    // Perform scan at sub-group level. For non active sub-groups, we pad these with the last element and ultimately
    // discard the scan value. Due to observed performance regressions when changing this behavior, non-full sub-groups
    // still perform scans over their full set of values. The returned result is defined as each array element is
    // initialized by the last input, but not part of the actual scan and is ignored. The __sub_group_carry in
    // this case does not affect the scan as it only occurs with the last sub-group.
    //
    // TODO: we should analyze why limiting the sub-group scan causes performance regressions.
    _InputType __sub_group_carry = __sub_group_scan<__sub_group_size, __iters_per_item>(
        __sub_group, __input, __binary_op, __items_in_sub_group_scan);
    [[maybe_unused]] _InputType __wg_init = __input[0].__v;
    if (__sub_group.get_local_linear_id() == __sub_group_size - 1)
    {
        __local_acc[__sub_group.get_group_linear_id()] = __sub_group_carry;
    }
    __dpl_sycl::__group_barrier(__item);
    // Scan over sub-group level reductions to compute incoming prefixes for a sub-group. Guard against
    // applying prefixes of non active sub-groups as there is no guarantee it is an identity.
    if (__sub_group_group_id == 0)
    {
        const std::uint8_t __num_iters =
            oneapi::dpl::__internal::__dpl_ceiling_div(__active_sub_groups, __sub_group_size);
        oneapi::dpl::__internal::__lazy_ctor_storage<_InputType> __wg_carry;
        std::uint8_t __idx = __sub_group.get_local_linear_id();
        _InputType __val = __local_acc[__idx];
        if (__num_iters == 1)
        {
            oneapi::dpl::__par_backend_hetero::__sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true,
                                                                        /*__init_present*/ false>(
                __sub_group, __val, __binary_op, __wg_carry, __active_sub_groups);
            __local_acc[__idx] = __val;
        }
        else
        {
            oneapi::dpl::__par_backend_hetero::__sub_group_scan<__sub_group_size, /*__is_inclusive*/ true,
                                                                /*__init_present*/ false>(__sub_group, __val,
                                                                                          __binary_op, __wg_carry);
            __local_acc[__idx] = __val;
            __idx += __sub_group_size;
            for (std::uint8_t __i = 1; __i < __num_iters - 1; ++__i)
            {
                __val = __local_acc[__idx];
                oneapi::dpl::__par_backend_hetero::__sub_group_scan<__sub_group_size, /*__is_inclusive*/ true,
                                                                    /*__init_present*/ true>(__sub_group, __val,
                                                                                             __binary_op, __wg_carry);
                __local_acc[__idx] = __val;
                __idx += __sub_group_size;
            }
            __val = __local_acc[__idx];
            oneapi::dpl::__par_backend_hetero::__sub_group_scan_partial<__sub_group_size, /*__is_inclusive*/ true,
                                                                        /*__init_present*/ true>(
                __sub_group, __val, __binary_op, __wg_carry,
                __active_sub_groups - (__num_iters - 1) * __sub_group_size);
            __local_acc[__idx] = __val;
        }
        // Init callback, most common case is expected to be a decoupled lookback to achieve a global scan between
        // work-groups.
        __init_callback(__wg_init, __sub_group, __wg_carry.__v);
        __wg_carry.__destroy();
    }
    __dpl_sycl::__group_barrier(__item);
    // Determine incoming prefix from previous sub-groups and / or work-groups, and update results in __input
    if constexpr (_InitCallback::__apply_prefix)
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
                __input[__i].__v = __binary_op(__sub_group_carry_in, __input[__i].__v);
        }
    }
    else
    {
        if (__sub_group_group_id > 0 && __sub_group_group_id < __active_sub_groups)
        {
            _InputType __sub_group_carry_in =
                __dpl_sycl::__group_broadcast(__sub_group, __local_acc[__sub_group_group_id - 1]);
            for (std::uint16_t __i = 0; __i < __iters_per_item; ++__i)
                __input[__i].__v = __binary_op(__sub_group_carry_in, __input[__i].__v);
        }
    }
}

//
// An optimized work-group scan that is made up of smaller register based sub-group scans.
// The SLM requirement for this implementation is sizeof(_InputType) * ceil(work_group_size / sub_group_size) as
// a single element per sub-group of _InputType is required. The results of the scan are updated in __input.
// Furthermore, a callback functor, _InitCallback, is accepted that prefixes the result of the work-group scan if its
// static __apply_prefix member is set to true.
//
// Input is accepted in the form of an array in sub-group strided order with sub-groups processing contiguous blocks
// in an input. Formally, for some index i in __input, __input[i] must correspond to position
//      (i * sg_sz + sg_lid) + (sg_sz * sg_gid * iters_per_item)
// in the desired work-group scan where sg_sz is the size of the sub-group, sg_lid is the local offset of an item in
// the sub-group, and sg_gid is the group number of the sub-group in the containing work-group. This layout is to align
// with optimal loads from global memory without extra data movement.
//
// Suppose, a work scan over 0, 1, 2, 3, ... 31 is to be performed with a __sub_group_size of 4, 4 __iters_per_item,
// and 2 total sub-groups for a total of 8 work-items in the work-group. To perform this operation, the elements must
// be held in the following order:
//
// sub_group 0: work_group_id 0:  0,  4,  8,  12
//              work_group_id 1:  1,  5,  9,  13
//              work_group_id 2:  2,  6,  10, 14
//              work_group_id 3:  3,  7,  11, 15
//
// sub_group 1: work_group_id 4:  16, 20, 24, 28
//              work_group_id 5:  17, 21, 25, 29
//              work_group_id 6:  18, 22, 26, 30
//              work_group_id 7:  19, 23, 27, 31
//
template <std::uint8_t __sub_group_size, std::uint16_t __iters_per_item, typename _InputType, typename _NdItem,
          typename _SlmAcc, typename _BinaryOperation, typename _InitCallback>
void
__work_group_scan(const _NdItem& __item, _SlmAcc __local_acc,
                  oneapi::dpl::__internal::__lazy_ctor_storage<_InputType> __input[__iters_per_item],
                  _BinaryOperation __binary_op, _InitCallback __init_callback, std::uint32_t __items_in_scan)
{
    __work_group_scan_impl<__sub_group_size, __iters_per_item>(__item, __local_acc, __input, __binary_op,
                                                               __init_callback, __items_in_scan);
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
