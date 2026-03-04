// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_UTILS_H
#define _ONEDPL_KT_UTILS_H

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/utils.h"

#include <cstdint>
#include <algorithm>
#include <iterator>

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

// The number of groups that should be launched in a cooperative kernel.
// Returns the min of the max groups supported by the HW and the tile count
template <typename _Kernel>
std::uint32_t
__get_num_cooperative_groups(const _Kernel& __kernel, sycl::queue& __q, std::uint32_t __work_group_size,
                             std::uint32_t __tile_count, std::uint32_t __slm_size_bytes)
{
    std::uint32_t __max_num_cooperative_groups = 1;
#if defined(SYCL_EXT_ONEAPI_FORWARD_PROGRESS) && defined(SYCL_EXT_ONEAPI_ROOT_GROUP)
    std::uint32_t __max_work_group_kernel_query =
        __kernel.template ext_oneapi_get_info<syclex::info::kernel_queue_specific::max_num_work_groups>(
            __q, __work_group_size, __slm_size_bytes);

    // There is a bug produced on BMG where zeKernelSuggestMaxCooperativeGroupCount suggests too large of a
    // work-group count when we are beyond half SLM capacity, causing a hang. To fix this, we can manually compute
    // the safe number of groups to launch and take the min with the root group query for any kernel specific
    // restrictions that may limit the number of groups
    constexpr std::uint32_t __xve_per_xe = 8;
    constexpr std::uint32_t __lanes_per_xe = 2048;
    const std::uint32_t __max_groups_per_xe = __lanes_per_xe / __work_group_size;

    const std::uint32_t __max_slm_xe = __q.get_device().get_info<sycl::info::device::local_mem_size>();
    const std::uint32_t __xes_on_device =
        __q.get_device().get_info<sycl::info::device::max_compute_units>() / __xve_per_xe;

    // The HW reserves SLM for a work group on a limited number of granularities. We must account for this to avoid
    // launching too many groups.
    constexpr std::uint32_t __kib = 1 << 10;
    constexpr std::uint32_t __slm_granularity_table[] = {0,          1 * __kib,  2 * __kib,  4 * __kib,
                                                         8 * __kib,  16 * __kib, 24 * __kib, 32 * __kib,
                                                         48 * __kib, 64 * __kib, 96 * __kib, 128 * __kib};
    constexpr std::uint32_t __slm_granularity_table_size = sizeof(__slm_granularity_table) / sizeof(std::uint32_t);
    const std::uint32_t* __slm_granularity_it = std::lower_bound(
        __slm_granularity_table, __slm_granularity_table + __slm_granularity_table_size, __slm_size_bytes);
    assert(__slm_granularity_it != std::cend(__slm_granularity_table));
    const std::uint32_t __true_slm_size_bytes = *__slm_granularity_it;

    const std::uint32_t __groups_per_xe_slm_adj = std::min(__max_groups_per_xe, __max_slm_xe / __true_slm_size_bytes);
    const std::uint32_t __concurrent_groups_est = __groups_per_xe_slm_adj * __xes_on_device;
    __max_num_cooperative_groups = std::min({__max_work_group_kernel_query, __tile_count, __concurrent_groups_est});
#else
    static_assert(oneapi::dpl::__internal::__always_false_v<_Kernel>,
                  "SYCL_EXT_ONEAPI_FORWARD_PROGRESS and SYCL_EXT_ONEAPI_ROOT_GROUP must be defined to call "
                  "__get_max_num_cooperative_groups");
#endif
    return __max_num_cooperative_groups;
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif
