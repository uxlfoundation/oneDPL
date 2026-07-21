// -*- C++ -*-
//===-- parallel_backend_sycl_histogram.h ---------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../histogram_binhash_utils.h"
#include "../../utils.h"

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename _Range>
struct __custom_boundary_range_binhash
{
    _Range __boundaries;
    __custom_boundary_range_binhash(_Range __boundaries_) : __boundaries(__boundaries_) {}

    template <typename _T2>
    oneapi::dpl::__internal::__bin_idx_t
    get_bin(_T2 __value) const
    {
        return oneapi::dpl::__internal::__custom_boundary_get_bin_helper(
            __boundaries, __boundaries.size(), __value, __boundaries[0], __boundaries[__boundaries.size() - 1]);
    }
};

// Baseline wrapper which provides no acceleration via SLM memory, but still
// allows generic calls to a wrapped binhash structure from within the kernels
template <typename _BinHash, typename _ExtraMemAccessor>
struct __binhash_SLM_wrapper
{
    _BinHash __bin_hash;
    __binhash_SLM_wrapper(_BinHash __bin_hash_, _ExtraMemAccessor /*__slm_mem_*/,
                          const sycl::nd_item<1>& /*__self_item*/)
        : __bin_hash(__bin_hash_)
    {
    }

    template <typename _T>
    oneapi::dpl::__internal::__bin_idx_t
    get_bin(_T __value) const
    {
        return __bin_hash.get_bin(__value);
    }
};

// Specialization for custom range binhash function which stores boundary data
// into SLM for quick repeated usage
template <typename _Range, typename _ExtraMemAccessor>
struct __binhash_SLM_wrapper<__custom_boundary_range_binhash<_Range>, _ExtraMemAccessor>
{
    using _bin_hash_type = typename oneapi::dpl::__par_backend_hetero::__custom_boundary_range_binhash<_Range>;

    _ExtraMemAccessor __slm_mem;
    __binhash_SLM_wrapper(_bin_hash_type __bin_hash, _ExtraMemAccessor __slm_mem_, const sycl::nd_item<1>& __self_item)
        : __slm_mem(__slm_mem_)
    {
        //initialize __slm_memory
        ::std::uint32_t __gSize = __self_item.get_local_range()[0];
        ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
        auto __size = __bin_hash.__boundaries.size();
        ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__size, __gSize);
        ::std::uint8_t __k = 0;
        for (; __k < __factor - 1; ++__k)
        {
            __slm_mem[__gSize * __k + __self_lidx] = __bin_hash.__boundaries[__gSize * __k + __self_lidx];
        }
        // residual
        if (__gSize * __k + __self_lidx < __size)
        {
            __slm_mem[__gSize * __k + __self_lidx] = __bin_hash.__boundaries[__gSize * __k + __self_lidx];
        }
    }

    template <typename _T>
    oneapi::dpl::__internal::__bin_idx_t
    get_bin(_T __value) const
    {
        auto __size = __slm_mem.size();
        return oneapi::dpl::__internal::__custom_boundary_get_bin_helper(__slm_mem, __size, __value, __slm_mem[0],
                                                                         __slm_mem[__size - 1]);
    }
};

template <typename _BinHash, typename _ExtraMemAccessor>
auto
__make_SLM_binhash(_BinHash __bin_hash, _ExtraMemAccessor __slm_mem, const sycl::nd_item<1>& __self_item)
{
    return __binhash_SLM_wrapper(__bin_hash, __slm_mem, __self_item);
}

template <typename... _Name>
class __histo_kernel_local_atomics;

template <typename... _Name>
class __histo_kernel_private_glocal_atomics;

template <typename _HistAccessor, typename _OffsetT, typename _Size>
void
__clear_wglocal_histograms(const _HistAccessor& __local_histogram, const _OffsetT& __offset, _Size __num_bins,
                           const sycl::nd_item<1>& __self_item)
{
    using _BinUint_t =
        ::std::conditional_t<(sizeof(_Size) >= sizeof(::std::uint32_t)), ::std::uint64_t, ::std::uint32_t>;
    _BinUint_t __gSize = __self_item.get_local_range()[0];
    ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
    ::std::uint8_t __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, __gSize);
    ::std::uint8_t __k = 0;

    for (; __k < __factor - 1; ++__k)
    {
        __local_histogram[__offset + __gSize * __k + __self_lidx] = 0;
    }
    // residual
    if (__gSize * __k + __self_lidx < __num_bins)
    {
        __local_histogram[__offset + __gSize * __k + __self_lidx] = 0;
    }
    sycl::group_barrier(__self_item.get_group());
}

// Atomically increment the bin for __x in a histogram with generalized addressing:
//   address = __c * __stride + __offset
// where __c is the bin index. Stride=1 with offset=base gives the contiguous per-WG
// layout used by the global-atomics path; stride=num_copies with offset=copy_slot gives
// the blocked-by-bin replicated layout used by the SLM path.
template <sycl::access::address_space _AddressSpace, typename _ValueType, typename _HistAccessor, typename _BinFunc>
void
__accum_local_atomics_iter(const _ValueType& __x, const _HistAccessor& __wg_local_histogram, std::size_t __offset,
                           std::uint32_t __stride, _BinFunc __func)
{
    using _histo_value_type = typename _HistAccessor::value_type;
    oneapi::dpl::__internal::__bin_idx_t __c = __func.get_bin(__x);
    if (__c >= 0)
    {
        __dpl_sycl::__atomic_ref<_histo_value_type, _AddressSpace> __local_bin(
            __wg_local_histogram[__c * __stride + __offset]);
        ++__local_bin;
    }
}

template <std::uint16_t __iters_per_work_item, typename _KernelName>
struct __histogram_general_local_atomics_submitter;

template <std::uint16_t __iters_per_work_item, typename... _KernelName>
struct __histogram_general_local_atomics_submitter<__iters_per_work_item,
                                                   __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _Range1, typename _Range2, typename _BinHashMgr>
    sycl::event
    operator()(sycl::queue& __q, const sycl::event& __init_event, std::uint16_t __work_group_size,
               std::uint32_t __num_slm_copies, _Range1&& __input, _Range2&& __bins,
               const _BinHashMgr& __binhash_manager)
    {
        const std::size_t __n = oneapi::dpl::__ranges::__size(__input);
        const std::uint16_t __num_bins = oneapi::dpl::__ranges::__size(__bins);
        using _local_histogram_type = std::uint32_t;
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using _extra_memory_type = typename _BinHashMgr::_extra_memory_type;

        auto __extra_SLM_elements = __binhash_manager.get_required_SLM_elements();
        std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
        return __q.submit([&](auto& __h) {
            __h.depends_on(__init_event);
            auto _device_copyable_func = __binhash_manager.prepare_device_binhash(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            // SLM histogram copies to reduce atomic contention. Number of copies is based
            // on device sub-group size, with sub-group lanes mapped to copies via modulo.
            __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(
                sycl::range(__num_slm_copies * __num_bins), __h);
            __dpl_sycl::__local_accessor<_extra_memory_type> __extra_SLM(sycl::range(__extra_SLM_elements), __h);
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    constexpr auto _atomic_address_space = sycl::access::address_space::local_space;
                    const std::size_t __self_lidx = __self_item.get_local_id(0);
                    const std::size_t __wgroup_idx = __self_item.get_group(0);
                    const std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;
                    auto __SLM_binhash = __make_SLM_binhash(_device_copyable_func, __extra_SLM, __self_item);
                    // Blocked SLM layout: replicas of bin B occupy [B*num_copies, (B+1)*num_copies).
                    // Each work-item picks a copy slot indexed by its sub-group lane id.
                    std::uint32_t __copy_slot = 0;
#if _ONEDPL_USE_SUB_GROUPS
                    if (__num_slm_copies > 1)
                    {
                        const std::uint32_t __lane_id = __self_item.get_sub_group().get_local_linear_id();
                        __copy_slot = __lane_id % __num_slm_copies;
                    }
#endif

                    __clear_wglocal_histograms(__local_histogram, 0, __num_slm_copies * __num_bins, __self_item);

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            __accum_local_atomics_iter<_atomic_address_space>(
                                __input[__seg_start + __idx * __work_group_size + __self_lidx], __local_histogram,
                                __copy_slot, __num_slm_copies, __SLM_binhash);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_atomics_iter<_atomic_address_space>(__input[__val_idx], __local_histogram,
                                                                                  __copy_slot, __num_slm_copies,
                                                                                  __SLM_binhash);
                            }
                        }
                    }

                    sycl::group_barrier(__self_item.get_group());

                    // Merge SLM histogram copies into global output via atomic add per bin.
                    // Iterate strided so all bins are covered when __num_bins exceeds work-group size.
                    for (std::uint16_t __bin = __self_lidx; __bin < __num_bins; __bin += __work_group_size)
                    {
                        _local_histogram_type __merged = 0;
                        const std::uint32_t __base = __bin * __num_slm_copies;
                        for (std::uint32_t __s = 0; __s < __num_slm_copies; ++__s)
                        {
                            __merged += __local_histogram[__base + __s];
                        }
                        __dpl_sycl::__atomic_ref<_bin_type, sycl::access::address_space::global_space> __global_bin(
                            __bins[__bin]);
                        __global_bin += __merged;
                    }
                });
        });
    }
};

template <typename _CustomName, std::uint16_t __iters_per_work_item, typename _Range1, typename _Range2,
          typename _BinHashMgr>
sycl::event
__histogram_general_local_atomics(sycl::queue& __q, const sycl::event& __init_event, std::uint16_t __work_group_size,
                                  std::uint32_t __num_slm_copies, _Range1&& __input, _Range2&& __bins,
                                  const _BinHashMgr& __binhash_manager)
{
    using _iters_per_work_item_t = ::std::integral_constant<::std::uint16_t, __iters_per_work_item>;

    // Required to include _iters_per_work_item_t in kernel name because we compile multiple kernels and decide between
    // them at runtime.  Other compile time arguments aren't required as it is the user's responsibility to provide a
    // unique kernel name to the policy for each call when using no-unamed-lambdas
    using _local_atomics_name = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_local_atomics<_iters_per_work_item_t, _CustomName>>;

    return __histogram_general_local_atomics_submitter<__iters_per_work_item, _local_atomics_name>()(
        __q, __init_event, __work_group_size, __num_slm_copies, std::forward<_Range1>(__input),
        std::forward<_Range2>(__bins), __binhash_manager);
}

template <typename _KernelName>
struct __histogram_general_private_global_atomics_submitter;

template <typename... _KernelName>
struct __histogram_general_private_global_atomics_submitter<__internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _Range1, typename _Range2, typename _BinHashMgr>
    sycl::event
    operator()(sycl::queue& __q, const sycl::event& __init_event, std::uint16_t __min_iters_per_work_item,
               std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
               const _BinHashMgr& __binhash_manager)
    {
        const ::std::size_t __n = oneapi::dpl::__ranges::__size(__input);
        const ::std::size_t __num_bins = oneapi::dpl::__ranges::__size(__bins);
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;

        const std::uint64_t __global_mem_size = __q.get_device().get_info<sycl::info::device::global_mem_size>();
        const std::uint64_t __max_groups =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __min_iters_per_work_item);
        const std::uint64_t __max_segments =
            std::min(__global_mem_size / (__num_bins * sizeof(_bin_type)), __max_groups);

        const std::size_t __iters_per_work_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __max_segments * __work_group_size);
        const std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);

        auto __private_histograms =
            oneapi::dpl::__par_backend_hetero::__buffer<_bin_type>(__segments * __num_bins).get_buffer();

        return __q.submit([&](auto& __h) {
            __h.depends_on(__init_event);
            auto _device_copyable_func = __binhash_manager.prepare_device_binhash(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            sycl::accessor __hacc_private{__private_histograms, __h, sycl::read_write, sycl::no_init};
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    constexpr auto _atomic_address_space = sycl::access::address_space::global_space;
                    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                    const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                    const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;

                    __clear_wglocal_histograms(__hacc_private, __wgroup_idx * __num_bins, __num_bins, __self_item);

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        for (::std::size_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            __accum_local_atomics_iter<_atomic_address_space>(__input[__val_idx], __hacc_private,
                                                                              __wgroup_idx * __num_bins, 1u,
                                                                              _device_copyable_func);
                        }
                    }
                    else
                    {
                        for (::std::size_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_atomics_iter<_atomic_address_space>(__input[__val_idx], __hacc_private,
                                                                                  __wgroup_idx * __num_bins, 1u,
                                                                                  _device_copyable_func);
                            }
                        }
                    }

                    sycl::group_barrier(__self_item.get_group());
                    const std::size_t __offset = __wgroup_idx * __num_bins;
                    for (std::size_t __bin = __self_lidx; __bin < __num_bins; __bin += __work_group_size)
                    {
                        __dpl_sycl::__atomic_ref<_bin_type, sycl::access::address_space::global_space> __global_bin(
                            __bins[__bin]);
                        __global_bin += __hacc_private[__offset + __bin];
                    }
                });
        });
    }
};
template <typename _CustomName, typename _Range1, typename _Range2, typename _BinHashMgr>
sycl::event
__histogram_general_private_global_atomics(sycl::queue& __q, const sycl::event& __init_event,
                                           std::uint16_t __min_iters_per_work_item, std::uint16_t __work_group_size,
                                           _Range1&& __input, _Range2&& __bins, const _BinHashMgr& __binhash_manager)
{
    using _global_atomics_name = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_private_glocal_atomics<_CustomName>>;

    return __histogram_general_private_global_atomics_submitter<_global_atomics_name>()(
        __q, __init_event, __min_iters_per_work_item, __work_group_size, std::forward<_Range1>(__input),
        std::forward<_Range2>(__bins), __binhash_manager);
}

template <typename _CustomName, std::uint16_t __iters_per_work_item, typename _Range1, typename _Range2,
          typename _BinHashMgr>
__future<sycl::event>
__parallel_histogram_select_kernel(sycl::queue& __q, const sycl::event& __init_event, _Range1&& __input,
                                   _Range2&& __bins, const _BinHashMgr& __binhash_manager)
{
    using _local_histogram_type = std::uint32_t;
    using _extra_memory_type = typename _BinHashMgr::_extra_memory_type;

    const std::uint32_t __num_bins = oneapi::dpl::__ranges::__size(__bins);
    // Limit the maximum work-group size for better performance. Empirically found value.
    std::uint16_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__q, std::uint16_t(1024));

    // Reserve some SLM as headroom for runtime overhead.
    const std::size_t __local_mem_size =
        (__q.get_device().template get_info<sycl::info::device::local_mem_size>() * 4) / 5;

    // Upper bound on useful copies: one per sub-group lane (replicas indexed by lane id).
    // Use device's max sub-group size (the realized SIMD width for ordinary kernels on Intel
    // GPUs), capped at 32 to bound SLM and merge cost on devices reporting larger maxima. If
    // the compiler ends up selecting a smaller SIMD width, surplus slots are simply unused;
    // the SLM-ratio guard further limits total allocation.
    // When sub-groups are unavailable we cannot map work-items to copies.
#if _ONEDPL_USE_SUB_GROUPS
    const std::uint32_t __max_useful_copies =
        std::min<std::uint32_t>(32u, oneapi::dpl::__internal::__max_sub_group_size(__q));
#else
    const std::uint32_t __max_useful_copies = 1;
#endif
    const std::size_t __extra_SLM_bytes = __binhash_manager.get_required_SLM_elements() * sizeof(_extra_memory_type);
    const std::size_t __per_copy_bytes = __num_bins * sizeof(_local_histogram_type);

    // Replication is only worth its clear + merge overhead when input is large enough to amortize it.
    // If we only have less than half of a single WG, replication does not make sense as contention is not an issue.
    const std::size_t __n = oneapi::dpl::__ranges::__size(__input);
    const bool __replicate = __n > __work_group_size * __iters_per_work_item / 2;

    // Try to fit within a subset of SLM to preserve occupancy with 2 concurrent work-groups
    const std::size_t __target_slm_size =
        (__local_mem_size / 2 > __extra_SLM_bytes) ? __local_mem_size / 2 - __extra_SLM_bytes : 0;

    // Use as many copies as fit within target memory limits, capped by the useful-copies bound.
    std::uint32_t __num_slm_copies = std::min<std::uint32_t>(__max_useful_copies, __target_slm_size / __per_copy_bytes);

    if (!__replicate || __num_slm_copies == 0)
    {
        // If the target size is too small, or its not worth it to replicate, try to fit at least one copy by
        // using all available SLM, this is better than global atomics
        __num_slm_copies = (__local_mem_size >= __per_copy_bytes + __extra_SLM_bytes) ? 1 : 0;
    }

    if (__num_slm_copies > 0)
    {
        return __future(__histogram_general_local_atomics<_CustomName, __iters_per_work_item>(
            __q, __init_event, __work_group_size, __num_slm_copies, std::forward<_Range1>(__input),
            std::forward<_Range2>(__bins), __binhash_manager));
    }
    else // otherwise, use global atomics (private copies per workgroup)
    {
        //Use __iters_per_work_item here as a runtime parameter, because only one kernel is created for
        // private_global_atomics with a variable number of iterations per workitem. __iters_per_work_item is just a
        // suggestion which but global memory limitations may increase this value to be able to fit the workgroup
        // private copies of the histogram bins in global memory.  No unrolling is taken advantage of here because it
        // is a runtime argument.
        return __future(__histogram_general_private_global_atomics<_CustomName>(
            __q, __init_event, __iters_per_work_item, __work_group_size, std::forward<_Range1>(__input),
            std::forward<_Range2>(__bins), __binhash_manager));
    }
}

template <typename _ExecutionPolicy, typename _Event, typename _Range1, typename _Range2, typename _BinHashMgr>
__future<sycl::event>
__parallel_histogram(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                     const _Event& __init_event, _Range1&& __input, _Range2&& __bins,
                     const _BinHashMgr& __binhash_manager)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    if (oneapi::dpl::__ranges::__size(__input) < 1048576) // 2^20
    {
        return __parallel_histogram_select_kernel<_CustomName, /*iters_per_workitem = */ 4>(
            __q_local, __init_event, std::forward<_Range1>(__input), std::forward<_Range2>(__bins), __binhash_manager);
    }
    else
    {
        return __parallel_histogram_select_kernel<_CustomName, /*iters_per_workitem = */ 32>(
            __q_local, __init_event, std::forward<_Range1>(__input), std::forward<_Range2>(__bins), __binhash_manager);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
