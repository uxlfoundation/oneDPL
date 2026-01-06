// -*- C++ -*-
//===-- sycl_radix_sort_kernels.h ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H

#include <cstdint>
#include <type_traits>

#include "oneapi/dpl/pstl/onedpl_config.h"
#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/utils.h"

#include "sycl_radix_sort_utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__sycl::__impl
{

//-----------------------------------------------------------------------------
// Histogram kernel: Compute global histograms for all stages
//-----------------------------------------------------------------------------
// TODO: Implement with SYCL sub-group operations and group-local memory
//
// Algorithm:
// 1. Initialize group-local histogram in SLM
// 2. Each work-item processes chunk of data
// 3. For each key, extract bins for all stages and increment local histogram
// 4. Reduce local histograms to global histogram via atomics
//
// Key SYCL features to use:
// - sycl::local_accessor for group-local histogram
// - sycl::atomic_ref with memory_scope::work_group for local updates
// - sycl::atomic_ref with memory_scope::device for global updates

//-----------------------------------------------------------------------------
// Onesweep kernel: Main sweep/reorder kernel with lookback
//-----------------------------------------------------------------------------
// TODO: Implement with SYCL work-group coordination
//
// Algorithm (per stage):
// 1. Load keys (and values if key-value sort)
// 2. Local ranking within work-item
//    - Extract bins for current stage
//    - Use atomics on SLM counters to assign local ranks
// 3. Global ranking
//    - Scan work-item histograms across work-group
//    - Add global histogram offsets (from scan phase)
//    - Lookback: add previous work-groups' contributions
// 4. Reorder via SLM
//    - Write keys to SLM at local ranks
//    - Coalesced read from SLM and write to global memory
//
// Key SYCL features to use:
// - sycl::local_accessor for SLM (histograms and reorder buffer)
// - sycl::sub_group operations for ranking within work-item
// - sycl::group_barrier for synchronization
// - sycl::atomic_ref for lookback coordination

} // namespace oneapi::dpl::experimental::kt::gpu::__sycl::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
