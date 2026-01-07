// -*- C++ -*-
//===-- sycl_radix_sort_utils.h ------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H

#include <limits>
#include <cstdint>
#include <type_traits>
#include <bit>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "oneapi/dpl/pstl/onedpl_config.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

struct __esimd_tag{};
struct __sycl_tag{};


//-----------------------------------------------------------------------------
// Parameter validation
//-----------------------------------------------------------------------------
template <std::uint8_t __radix_bits, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size>
constexpr void
__check_onesweep_params()
{
    static_assert(__radix_bits == 8);
    static_assert(__data_per_workitem % 32 == 0);
    static_assert(__workgroup_size == 32 || __workgroup_size == 64);
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
