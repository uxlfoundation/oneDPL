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
    auto
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
    auto
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
    auto
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
class __histo_kernel_register_local_red;

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
    __dpl_sycl::__group_barrier(__self_item);
}

template <typename _BinIdxType, typename _ValueType, typename _HistReg, typename _BinFunc>
void
__accum_local_register_iter(const _ValueType& __x, _HistReg* __histogram, _BinFunc __func)
{
    _BinIdxType c = __func.get_bin(__x);
    if (c >= 0)
    {
        ++__histogram[c];
    }
}

template <typename _BinIdxType, sycl::access::address_space _AddressSpace, typename _ValueType, typename _HistAccessor,
          typename _OffsetT, typename _BinFunc>
void
__accum_local_atomics_iter(const _ValueType& __x, const _HistAccessor& __wg_local_histogram, const _OffsetT& __offset,
                           _BinFunc __func)
{
    using _histo_value_type = typename _HistAccessor::value_type;
    _BinIdxType __c = __func.get_bin(__x);
    if (__c >= 0)
    {
        __dpl_sycl::__atomic_ref<_histo_value_type, _AddressSpace> __local_bin(__wg_local_histogram[__offset + __c]);
        ++__local_bin;
    }
}

template <typename _BinType, typename _FactorType, typename _HistAccessorIn, typename _OffsetT,
          typename _HistAccessorOut, typename _Size>
void
__reduce_out_histograms(const _HistAccessorIn& __in_histogram, const _OffsetT& __offset,
                        const _HistAccessorOut& __out_histogram, _Size __num_bins, const sycl::nd_item<1>& __self_item)
{
    using _BinUint_t =
        ::std::conditional_t<(sizeof(_Size) >= sizeof(::std::uint32_t)), ::std::uint64_t, ::std::uint32_t>;
    _BinUint_t __gSize = __self_item.get_local_range()[0];
    ::std::uint32_t __self_lidx = __self_item.get_local_id(0);
    _FactorType __factor = oneapi::dpl::__internal::__dpl_ceiling_div(__num_bins, __gSize);
    _FactorType __k = 0;

    for (; __k < __factor - 1; ++__k)
    {
        __dpl_sycl::__atomic_ref<_BinType, sycl::access::address_space::global_space> __global_bin(
            __out_histogram[__gSize * __k + __self_lidx]);
        __global_bin += __in_histogram[__offset + __gSize * __k + __self_lidx];
    }
    // residual
    if (__gSize * __k + __self_lidx < __num_bins)
    {
        __dpl_sycl::__atomic_ref<_BinType, sycl::access::address_space::global_space> __global_bin(
            __out_histogram[__gSize * __k + __self_lidx]);
        __global_bin += __in_histogram[__offset + __gSize * __k + __self_lidx];
    }
}

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename _KernelName>
struct __histogram_general_registers_local_reduction_submitter;

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename... _KernelName>
struct __histogram_general_registers_local_reduction_submitter<__iters_per_work_item, __bins_per_work_item,
                                                               __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinHashMgr>
    auto
    operator()(_ExecutionPolicy&& __exec, const sycl::event& __init_event, ::std::uint16_t __work_group_size,
               _Range1&& __input, _Range2&& __bins, const _BinHashMgr& __binhash_manager)
    {
        const ::std::size_t __n = __input.size();
        const ::std::uint8_t __num_bins = __bins.size();
        using _local_histogram_type = ::std::uint32_t;
        using _private_histogram_type = ::std::uint16_t;
        using _histogram_index_type = ::std::int8_t;
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using _extra_memory_type = typename _BinHashMgr::_extra_memory_type;

        ::std::size_t __extra_SLM_elements = __binhash_manager.get_required_SLM_elements();
        ::std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
        return __exec.queue().submit([&](auto& __h) {
            __h.depends_on(__init_event);
            auto _device_copyable_func = __binhash_manager.prepare_device_binhash(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(sycl::range(__num_bins), __h);
            __dpl_sycl::__local_accessor<_extra_memory_type> __extra_SLM(sycl::range(__extra_SLM_elements), __h);
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                    const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                    const ::std::size_t __seg_start = __work_group_size * __iters_per_work_item * __wgroup_idx;
                    auto __SLM_binhash = __make_SLM_binhash(_device_copyable_func, __extra_SLM, __self_item);
                    __clear_wglocal_histograms(__local_histogram, 0, __num_bins, __self_item);
                    _private_histogram_type __histogram[__bins_per_work_item] = {0};

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            __accum_local_register_iter<_histogram_index_type>(
                                __input[__seg_start + __idx * __work_group_size + __self_lidx], __histogram,
                                __SLM_binhash);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_register_iter<_histogram_index_type>(__input[__val_idx], __histogram,
                                                                                   __SLM_binhash);
                            }
                        }
                    }

                    for (_histogram_index_type __k = 0; __k < __num_bins; ++__k)
                    {
                        __dpl_sycl::__atomic_ref<_local_histogram_type, sycl::access::address_space::local_space>
                            __local_bin(__local_histogram[__k]);
                        __local_bin += __histogram[__k];
                    }

                    __dpl_sycl::__group_barrier(__self_item);

                    __reduce_out_histograms<_bin_type, ::std::uint8_t>(__local_histogram, 0, __bins, __num_bins,
                                                                       __self_item);
                });
        });
    }
};

template <::std::uint16_t __iters_per_work_item, ::std::uint8_t __bins_per_work_item, typename _ExecutionPolicy,
          typename _Range1, typename _Range2, typename _BinHashMgr>
auto
__histogram_general_registers_local_reduction(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                              const sycl::event& __init_event, ::std::uint16_t __work_group_size,
                                              _Range1&& __input, _Range2&& __bins, const _BinHashMgr& __binhash_manager)
{
    using _kernel_base_name = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    using _iters_per_work_item_t = ::std::integral_constant<::std::uint16_t, __iters_per_work_item>;

    // Required to include _iters_per_work_item_t in kernel name because we compile multiple kernels and decide between
    // them at runtime.  Other compile time arguments aren't required as it is the user's responsibility to provide a
    // unique kernel name to the policy for each call when using no-unamed-lambdas
    using _RegistersLocalReducName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_register_local_red<_iters_per_work_item_t, _kernel_base_name>>;

    return __histogram_general_registers_local_reduction_submitter<__iters_per_work_item, __bins_per_work_item,
                                                                   _RegistersLocalReducName>()(
        ::std::forward<_ExecutionPolicy>(__exec), __init_event, __work_group_size, ::std::forward<_Range1>(__input),
        ::std::forward<_Range2>(__bins), __binhash_manager);
}

template <::std::uint16_t __iters_per_work_item, typename _KernelName>
struct __histogram_general_local_atomics_submitter;

template <::std::uint16_t __iters_per_work_item, typename... _KernelName>
struct __histogram_general_local_atomics_submitter<__iters_per_work_item,
                                                   __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinHashMgr>
    auto
    operator()(_ExecutionPolicy&& __exec, const sycl::event& __init_event, ::std::uint16_t __work_group_size,
               _Range1&& __input, _Range2&& __bins, const _BinHashMgr& __binhash_manager)
    {
        using _local_histogram_type = ::std::uint32_t;
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using _histogram_index_type = ::std::int16_t;
        using _extra_memory_type = typename _BinHashMgr::_extra_memory_type;

        ::std::size_t __extra_SLM_elements = __binhash_manager.get_required_SLM_elements();
        const ::std::size_t __n = __input.size();
        const ::std::size_t __num_bins = __bins.size();
        ::std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);
        return __exec.queue().submit([&](auto& __h) {
            __h.depends_on(__init_event);
            auto _device_copyable_func = __binhash_manager.prepare_device_binhash(__h);
            oneapi::dpl::__ranges::__require_access(__h, __input, __bins);
            // minimum type size for atomics
            __dpl_sycl::__local_accessor<_local_histogram_type> __local_histogram(sycl::range(__num_bins), __h);
            __dpl_sycl::__local_accessor<_extra_memory_type> __extra_SLM(sycl::range(__extra_SLM_elements), __h);
            __h.template parallel_for<_KernelName...>(
                sycl::nd_range<1>(__segments * __work_group_size, __work_group_size),
                [=](sycl::nd_item<1> __self_item) {
                    constexpr auto _atomic_address_space = sycl::access::address_space::local_space;
                    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                    const ::std::uint32_t __wgroup_idx = __self_item.get_group(0);
                    const ::std::size_t __seg_start = __work_group_size * __wgroup_idx * __iters_per_work_item;
                    auto __SLM_binhash = __make_SLM_binhash(_device_copyable_func, __extra_SLM, __self_item);

                    __clear_wglocal_histograms(__local_histogram, 0, __num_bins, __self_item);

                    if (__seg_start + __work_group_size * __iters_per_work_item < __n)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                __input[__val_idx], __local_histogram, 0, __SLM_binhash);
                        }
                    }
                    else
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (::std::uint8_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                    __input[__val_idx], __local_histogram, 0, __SLM_binhash);
                            }
                        }
                    }
                    __dpl_sycl::__group_barrier(__self_item);

                    __reduce_out_histograms<_bin_type, ::std::uint16_t>(__local_histogram, 0, __bins, __num_bins,
                                                                        __self_item);
                });
        });
    }
};

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _BinHashMgr>
auto
__histogram_general_local_atomics(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                  const sycl::event& __init_event, ::std::uint16_t __work_group_size, _Range1&& __input,
                                  _Range2&& __bins, const _BinHashMgr& __binhash_manager)
{
    using _kernel_base_name = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    using _iters_per_work_item_t = ::std::integral_constant<::std::uint16_t, __iters_per_work_item>;

    // Required to include _iters_per_work_item_t in kernel name because we compile multiple kernels and decide between
    // them at runtime.  Other compile time arguments aren't required as it is the user's responsibility to provide a
    // unique kernel name to the policy for each call when using no-unamed-lambdas
    using _local_atomics_name = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_local_atomics<_iters_per_work_item_t, _kernel_base_name>>;

    return __histogram_general_local_atomics_submitter<__iters_per_work_item, _local_atomics_name>()(
        ::std::forward<_ExecutionPolicy>(__exec), __init_event, __work_group_size, ::std::forward<_Range1>(__input),
        ::std::forward<_Range2>(__bins), __binhash_manager);
}

template <typename _KernelName>
struct __histogram_general_private_global_atomics_submitter;

template <typename... _KernelName>
struct __histogram_general_private_global_atomics_submitter<__internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinHashMgr>
    auto
    operator()(_BackendTag, _ExecutionPolicy&& __exec, const sycl::event& __init_event,
               ::std::uint16_t __min_iters_per_work_item, ::std::uint16_t __work_group_size, _Range1&& __input,
               _Range2&& __bins, const _BinHashMgr& __binhash_manager)
    {
        const ::std::size_t __n = __input.size();
        const ::std::size_t __num_bins = __bins.size();
        using _bin_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using _histogram_index_type = ::std::int32_t;

        auto __global_mem_size = __exec.queue().get_device().template get_info<sycl::info::device::global_mem_size>();
        const ::std::size_t __max_segments =
            ::std::min(__global_mem_size / (__num_bins * sizeof(_bin_type)),
                       oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __min_iters_per_work_item));
        const ::std::size_t __iters_per_work_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __max_segments * __work_group_size);
        ::std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __iters_per_work_item);

        auto __private_histograms =
            oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _bin_type>(__exec, __segments * __num_bins)
                .get_buffer();

        return __exec.queue().submit([&](auto& __h) {
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
                            __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                __input[__val_idx], __hacc_private, __wgroup_idx * __num_bins, _device_copyable_func);
                        }
                    }
                    else
                    {
                        for (::std::size_t __idx = 0; __idx < __iters_per_work_item; ++__idx)
                        {
                            ::std::size_t __val_idx = __seg_start + __idx * __work_group_size + __self_lidx;
                            if (__val_idx < __n)
                            {
                                __accum_local_atomics_iter<_histogram_index_type, _atomic_address_space>(
                                    __input[__val_idx], __hacc_private, __wgroup_idx * __num_bins,
                                    _device_copyable_func);
                            }
                        }
                    }

                    __dpl_sycl::__group_barrier(__self_item);

                    __reduce_out_histograms<_bin_type, ::std::uint32_t>(__hacc_private, __wgroup_idx * __num_bins,
                                                                        __bins, __num_bins, __self_item);
                });
        });
    }
};
template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinHashMgr>
auto
__histogram_general_private_global_atomics(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                           const sycl::event& __init_event, ::std::uint16_t __min_iters_per_work_item,
                                           ::std::uint16_t __work_group_size, _Range1&& __input, _Range2&& __bins,
                                           const _BinHashMgr& __binhash_manager)
{
    using _kernel_base_name = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    using _global_atomics_name = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __histo_kernel_private_glocal_atomics<_kernel_base_name>>;

    return __histogram_general_private_global_atomics_submitter<_global_atomics_name>()(
        oneapi::dpl::__internal::__device_backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __init_event,
        __min_iters_per_work_item, __work_group_size, ::std::forward<_Range1>(__input), ::std::forward<_Range2>(__bins),
        __binhash_manager);
}

template <::std::uint16_t __iters_per_work_item, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _BinHashMgr>
auto
__parallel_histogram_select_kernel(oneapi::dpl::__internal::__device_backend_tag __backend_tag,
                                   _ExecutionPolicy&& __exec, const sycl::event& __init_event, _Range1&& __input,
                                   _Range2&& __bins, const _BinHashMgr& __binhash_manager)
{
    using _private_histogram_type = ::std::uint16_t;
    using _local_histogram_type = ::std::uint32_t;
    using _extra_memory_type = typename _BinHashMgr::_extra_memory_type;

    const auto __num_bins = __bins.size();
    // Limit the maximum work-group size for better performance. Empirically found value.
    std::uint16_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__exec, std::uint16_t(1024));

    auto __local_mem_size = __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>();
    constexpr ::std::uint8_t __max_work_item_private_bins = 16 / sizeof(_private_histogram_type);

    // if bins fit into registers, use register private accumulation
    if (__num_bins <= __max_work_item_private_bins)
    {
        return __future(
            __histogram_general_registers_local_reduction<__iters_per_work_item, __max_work_item_private_bins>(
                __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __init_event, __work_group_size,
                ::std::forward<_Range1>(__input), ::std::forward<_Range2>(__bins), __binhash_manager));
    }
    // if bins fit into SLM, use local atomics
    else if (__num_bins * sizeof(_local_histogram_type) +
                 __binhash_manager.get_required_SLM_elements() * sizeof(_extra_memory_type) <
             __local_mem_size)
    {
        return __future(__histogram_general_local_atomics<__iters_per_work_item>(
            __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __init_event, __work_group_size,
            ::std::forward<_Range1>(__input), ::std::forward<_Range2>(__bins), __binhash_manager));
    }
    else // otherwise, use global atomics (private copies per workgroup)
    {
        //Use __iters_per_work_item here as a runtime parameter, because only one kernel is created for
        // private_global_atomics with a variable number of iterations per workitem. __iters_per_work_item is just a
        // suggestion which but global memory limitations may increase this value to be able to fit the workgroup
        // private copies of the histogram bins in global memory.  No unrolling is taken advantage of here because it
        // is a runtime argument.
        return __future(__histogram_general_private_global_atomics(
            __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __init_event, __iters_per_work_item,
            __work_group_size, ::std::forward<_Range1>(__input), ::std::forward<_Range2>(__bins), __binhash_manager));
    }
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinHashMgr>
auto
__parallel_histogram(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                     const sycl::event& __init_event, _Range1&& __input, _Range2&& __bins,
                     const _BinHashMgr& __binhash_manager)
{
    if (__input.size() < 1048576) // 2^20
    {
        return __parallel_histogram_select_kernel</*iters_per_workitem = */ 4>(
            __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __init_event, ::std::forward<_Range1>(__input),
            ::std::forward<_Range2>(__bins), __binhash_manager);
    }
    else
    {
        return __parallel_histogram_select_kernel</*iters_per_workitem = */ 32>(
            __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __init_event, ::std::forward<_Range1>(__input),
            ::std::forward<_Range2>(__bins), __binhash_manager);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_HISTOGRAM_H
