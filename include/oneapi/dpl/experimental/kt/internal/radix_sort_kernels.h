// -*- C++ -*-
//===-- radix_sort_kernels.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_RADIX_SORT_KERNELS_H

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <typename _KtTag, bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KeyT, typename _RngPack1, typename _RngPack2>
struct __one_wg_kernel;

template <typename _KtTag, bool __is_ascending, std::uint8_t __radix_bits, std::uint32_t __hist_work_group_count,
          std::uint16_t __hist_work_group_size, typename _KeysRng>
struct __global_histogram;

template <typename _KtTag, bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel;

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_RADIX_SORT_KERNELS_H
