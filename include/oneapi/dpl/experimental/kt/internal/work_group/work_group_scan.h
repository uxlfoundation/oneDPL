
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
#include "../utils.h"
#include "../sub_group/sub_group_scan.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

// TODO: consider adding init callback
template <int sub_group_size, int iters_per_item, typename InputType, /*typename OutputType,*/
          typename NdItem, typename SlmAcc, typename ArrayOrder, typename BinaryOperation>
auto
work_group_scan(ArrayOrder, const NdItem& item, SlmAcc local_acc, InputType input[iters_per_item],
                /*OutputType output[iters_per_item],*/ BinaryOperation binary_op, uint32_t items_in_scan)
{
    // This is the only currently supported strategy. However, we may wish to have future ones.
    if constexpr (std::is_same_v<ArrayOrder, item_array_order::sub_group_stride>)
    {
        auto sub_group = item.get_sub_group();
        auto sub_group_carry =
            sub_group_scan<sub_group_size, iters_per_item>(ArrayOrder{}, sub_group, input, /*output,*/ binary_op);
        const std::uint8_t sub_group_group_id = sub_group.get_group_linear_id();
        const std::uint8_t active_sub_groups =
            oneapi::dpl::__internal::__dpl_ceiling_div(items_in_scan, sub_group_size * iters_per_item);
        if (sub_group.get_local_linear_id() == sub_group_size - 1)
        {
            local_acc[sub_group.get_group_linear_id()] = sub_group_carry;
        }
        sycl::group_barrier(item.get_group());
        if (sub_group_group_id == 0)
        {
            const auto num_iters = oneapi::dpl::__internal::__dpl_ceiling_div(active_sub_groups, sub_group_size);
            InputType wg_carry = 0;
            auto idx = sub_group.get_local_linear_id();
            for (int i = 0; i < num_iters - 1; ++i)
            {
                auto val = local_acc[idx];
                __sub_group_scan<sub_group_size, true, true>(sub_group, val, binary_op, wg_carry);
                local_acc[idx] = val;
                idx += sub_group_size;
            }
            // masked last iteration for partial case
            const auto num_sub_groups = sub_group.get_group_linear_range();
            if (active_sub_groups == num_sub_groups)
            {
                auto val = local_acc[idx];
                __sub_group_scan<sub_group_size, true, true>(sub_group, val, binary_op, wg_carry);
                local_acc[idx] = val;
            }
            else
            {
                auto val = local_acc[idx];
                __sub_group_scan_partial<sub_group_size, true, true>(sub_group, val, binary_op, wg_carry,
                                                                     active_sub_groups % sub_group_size);
                local_acc[idx] = val;
            }
        }
        sycl::group_barrier(item.get_group());
        if (sub_group_group_id > 0)
        {
            if (sub_group_group_id < active_sub_groups)
            {
                const auto carry_in = sycl::group_broadcast(sub_group, local_acc[sub_group_group_id - 1]);
                for (int i = 0; i < iters_per_item; ++i)
                    input[i] = binary_op(carry_in, input[i]);
            }
        }
        // Should we use a group broadcast here?
        return local_acc[active_sub_groups - 1];
    }
    else
    {
        static_assert(false, "Current strategy unsupported");
    }
}

template <int sub_group_size, int iters_per_item, typename InputType, /*typename OutputType,*/
          typename NdItem, typename SlmAcc, typename ArrayOrder, typename BinaryOperation>
auto
work_group_scan(ArrayOrder, const NdItem& item, SlmAcc local_acc, InputType input[iters_per_item],
                /*OutputType output[iters_per_item],*/ BinaryOperation binary_op)
{
    return work_group_scan<sub_group_size, iters_per_item>(ArrayOrder{}, item, local_acc, input, binary_op,
                                                           item.get_local_range()[0] * iters_per_item);
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt
#endif
