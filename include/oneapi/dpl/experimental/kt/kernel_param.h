// -*- C++ -*-
//===-- kernel_param.h ------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_KERNEL_PARAM_H
#define _ONEDPL_KT_KERNEL_PARAM_H

#include <cstdint>

#include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"

namespace oneapi::dpl::experimental::kt
{

template <std::uint16_t __data_per_work_item, std::uint16_t __work_group_size,
          typename _KernelName = oneapi::dpl::execution::DefaultKernelName>
struct kernel_param
{
    inline static constexpr std::uint16_t data_per_workitem = __data_per_work_item;
    inline static constexpr std::uint16_t workgroup_size = __work_group_size;
    using kernel_name = _KernelName;
};

} // namespace oneapi::dpl::experimental::kt

#endif // _ONEDPL_KT_KERNEL_PARAM_H
