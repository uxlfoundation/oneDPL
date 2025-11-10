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

#ifndef _ONEDPL_KT_SINGLE_PASS_SCAN_H
#define _ONEDPL_KT_SINGLE_PASS_SCAN_H

#include "../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../pstl/hetero/dpcpp/unseq_backend_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl.h"
#include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"
#include "../../pstl/utils.h"

#include "internal/work_group/work_group_scan.h"
#include "internal/cooperative_lookback.h"

#include <cstdint>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

template <typename... _Name>
class __single_pass_scan_kernel;

template <std::uint8_t __sub_group_size, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size,
          typename _Type, typename _FlagType, typename _KernelName>
struct __single_pass_scan_submitter;

template <std::uint8_t __sub_group_size, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size,
          typename _Type, typename _FlagType, typename _InRng, typename _OutRng, typename _BinaryOp, typename _LocalAcc>
struct __single_pass_scan_kernel_func
{
    using _TileIdxT = typename _FlagType::_TileIdxT;
    static constexpr std::uint32_t __elems_in_tile = __workgroup_size * __data_per_workitem;

    _InRng __in_rng;
    _OutRng __out_rng;
    _BinaryOp __binary_op;
    std::size_t __n;
    _TileIdxT* __atomic_id_ptr;
    typename __scan_status_flag<__sub_group_size, _Type>::storage __lookback_storage;
    std::size_t __status_flags_size;
    _TileIdxT __num_tiles;
    _LocalAcc __slm;

    template <bool __is_full>
    void
    load_global_to_grf(oneapi::dpl::__internal::__lazy_ctor_storage<_Type> __grf_partials[__data_per_workitem],
                       const std::size_t __sub_group_current_offset, const std::uint32_t __sub_group_local_id) const
    {
        if constexpr (__is_full)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __grf_partials[__i].__setup(
                    __in_rng[__sub_group_current_offset + __sub_group_local_id + __sub_group_size * __i]);
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__sub_group_current_offset + __sub_group_local_id + __sub_group_size * __i < __n)
                {
                    __grf_partials[__i].__setup(
                        __in_rng[__sub_group_current_offset + __sub_group_local_id + __sub_group_size * __i]);
                }
                else // placeholder
                    __grf_partials[__i].__setup(__in_rng[__n - 1]);
            }
        }
    }

    template <bool __is_full>
    void
    store_grf_to_global(oneapi::dpl::__internal::__lazy_ctor_storage<_Type> __grf_partials[__data_per_workitem],
                        const std::size_t __sub_group_current_offset, const std::uint32_t __sub_group_local_id) const
    {
        if constexpr (__is_full)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __out_rng[__sub_group_current_offset + __sub_group_local_id + __sub_group_size * __i] =
                    __grf_partials[__i].__v;
                __grf_partials[__i].__destroy();
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__sub_group_current_offset + __sub_group_local_id + __sub_group_size * __i < __n)
                {
                    __out_rng[__sub_group_current_offset + __sub_group_local_id + __sub_group_size * __i] =
                        __grf_partials[__i].__v;
                }
                __grf_partials[__i].__destroy();
            }
        }
    }

    template <bool __is_full>
    void
    impl(const sycl::nd_item<1>& __item, const sycl::sub_group& __sub_group, std::uint32_t __tile_id,
         const std::size_t __work_group_offset, const std::size_t __sub_group_current_offset,
         const std::size_t __sub_group_next_offset) const
    {
        auto __sub_group_local_id = __sub_group.get_local_linear_id();
        auto __sub_group_group_id = __sub_group.get_group_linear_id();

        oneapi::dpl::__internal::__lazy_ctor_storage<_Type> __grf_partials[__data_per_workitem];

        auto __this_tile_elements = std::min<std::size_t>(__elems_in_tile, __n - __work_group_offset);
        // The first sub-group will query the previous tiles to find a prefix. For tile 0, we set it directly as full
        // The duplicated code in branches is critical for performance with IGC at the time of writing.
        if (__tile_id == 0)
        {
            load_global_to_grf<__is_full>(__grf_partials, __sub_group_current_offset, __sub_group_local_id);
            __cooperative_lookback_first_tile<__sub_group_size, _Type> __first_tile_callback{__lookback_storage,
                                                                                             __num_tiles, __tile_id};
            __work_group_scan<__sub_group_size, __data_per_workitem>(__item, __slm, __grf_partials, __binary_op,
                                                                     __first_tile_callback, __this_tile_elements);
            store_grf_to_global<__is_full>(__grf_partials, __sub_group_current_offset, __sub_group_local_id);
        }
        else
        {
            load_global_to_grf<__is_full>(__grf_partials, __sub_group_current_offset, __sub_group_local_id);
            __cooperative_lookback<__sub_group_size, _Type, _BinaryOp> __lookback_callback{__lookback_storage,
                                                                                           __tile_id, __binary_op};
            __work_group_scan<__sub_group_size, __data_per_workitem>(__item, __slm, __grf_partials, __binary_op,
                                                                     __lookback_callback, __this_tile_elements);
            store_grf_to_global<__is_full>(__grf_partials, __sub_group_current_offset, __sub_group_local_id);
        }
    }

    [[sycl::reqd_sub_group_size(__sub_group_size)]] void
    operator()(const sycl::nd_item<1>& __item) const
    {
        auto __group = __item.get_group();
        auto __subgroup = __item.get_sub_group();
        auto __local_id = __item.get_local_id(0);

        std::uint32_t __tile_id = 0;

        if (__num_tiles > 1)
        {
            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (__group.leader())
            {
                sycl::atomic_ref<_TileIdxT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    __idx_atomic(*__atomic_id_ptr);
                __tile_id = __idx_atomic.fetch_add(1);
            }
            __tile_id = sycl::group_broadcast(__group, __tile_id, 0);
        }
        auto __sub_group = __item.get_sub_group();
        auto __sub_group_local_id = __sub_group.get_local_linear_id();
        auto __sub_group_group_id = __sub_group.get_group_linear_id();

        std::size_t __work_group_offset = static_cast<std::size_t>(__tile_id) * __elems_in_tile;

        std::size_t __sub_group_current_offset =
            __work_group_offset + __sub_group_group_id * __data_per_workitem * __sub_group_size;
        std::size_t __sub_group_next_offset = __sub_group_current_offset + __sub_group_size * __data_per_workitem;

        // Making full / not full case a bool template parameter and compiling two separate functions significantly improves performance
        // over run-time checks immediately before load / store.
        if (__sub_group_next_offset <= __n)
            impl</*__is_full=*/true>(__item, __sub_group, __tile_id, __work_group_offset, __sub_group_current_offset,
                                     __sub_group_next_offset);
        else
            impl</*__is_full=*/false>(__item, __sub_group, __tile_id, __work_group_offset, __sub_group_current_offset,
                                      __sub_group_next_offset);
    }
};

template <std::uint8_t __sub_group_size, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size,
          typename _Type, typename _FlagType, typename... _Name>
struct __single_pass_scan_submitter<__sub_group_size, __data_per_workitem, __workgroup_size, _Type, _FlagType,
                                    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{

    template <typename _InRng, typename _OutRng, typename _BinaryOp>
    sycl::event
    operator()(sycl::queue __q, sycl::event __prev_event, _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOp __binary_op,
               std::size_t __n, std::uint32_t* __atomic_id_ptr,
               typename __scan_status_flag<__sub_group_size, _Type>::storage __lookback_storage,
               std::size_t __status_flags_size,
               typename __scan_status_flag<__sub_group_size, _Type>::_TileIdxT __num_wgs) const
    {
        using _LocalAccessorType = __dpl_sycl::__local_accessor<_Type, 1>;
        using _KernelFunc =
            __single_pass_scan_kernel_func<__sub_group_size, __data_per_workitem, __workgroup_size, _Type, _FlagType,
                                           std::decay_t<_InRng>, std::decay_t<_OutRng>, std::decay_t<_BinaryOp>,
                                           std::decay_t<_LocalAccessorType>>;

        return __q.submit([&](sycl::handler& __hdl) {
            auto __slm = _LocalAccessorType(
                oneapi::dpl::__internal::__dpl_ceiling_div(__workgroup_size, __sub_group_size), __hdl);
            __hdl.depends_on(__prev_event);
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
            __hdl.parallel_for(sycl::nd_range<1>(__num_wgs * __workgroup_size, __workgroup_size),
                               _KernelFunc{__in_rng, __out_rng, __binary_op, __n, __atomic_id_ptr, __lookback_storage,
                                           __status_flags_size, __num_wgs, __slm});
        });
    }
};

template <bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp, typename _KernelParam>
sycl::event
__single_pass_scan(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op, _KernelParam)
{
    constexpr std::uint8_t __sub_group_size = 32;

    using _Type = oneapi::dpl::__internal::__value_t<_OutRange>;
    using _FlagType = __scan_status_flag<__sub_group_size, _Type>;
    using _TileIdxT = typename _FlagType::_TileIdxT;
    using _FlagStorageType = typename _FlagType::storage;
    using _KernelName = typename _KernelParam::kernel_name;
    using _LookbackInitKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_init_kernel<_KernelName, _Type, _BinaryOp>>;
    using _SinglePassScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __single_pass_scan_kernel<_KernelName, _Type, _BinaryOp>>;

    const std::size_t __n = __in_rng.size();

    if (__n == 0)
        return sycl::event{};

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");
    assert("This device does not support 64-bit atomics" &&
           (sizeof(_Type) < 8 || __queue.get_device().has(sycl::aspect::atomic64)));

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = ::oneapi::dpl::__internal::__dpl_bit_ceil(__n);

    constexpr std::uint16_t __workgroup_size = _KernelParam::workgroup_size;
    constexpr std::uint16_t __data_per_workitem = _KernelParam::data_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of workgroup_size
    std::size_t __elems_in_tile = __workgroup_size * __data_per_workitem;
    _TileIdxT __num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);

    std::size_t __status_flags_size = 0;
    std::size_t __mem_bytes = 0;
    std::byte* __device_mem = nullptr;
    std::uint32_t* __atomic_id_ptr = nullptr;
    const bool __is_single_tile = (__num_wgs == 1);
    _FlagStorageType __lookback_storage(__device_mem, __mem_bytes, __status_flags_size);
    sycl::event __fill_event{};
    if (!__is_single_tile)
    {
        constexpr int __status_flag_padding = __sub_group_size;
        __status_flags_size = __num_wgs + __status_flag_padding;
        __mem_bytes = _FlagStorageType::get_reqd_storage(__status_flags_size) + sizeof(std::uint32_t);
        __device_mem = reinterpret_cast<std::byte*>(sycl::malloc_device(__mem_bytes, __queue));
        __atomic_id_ptr = reinterpret_cast<std::uint32_t*>(__device_mem + __mem_bytes - sizeof(std::uint32_t));
        if (!__device_mem)
            throw std::bad_alloc();
        __lookback_storage = _FlagStorageType(__device_mem, __mem_bytes, __status_flags_size);
        __fill_event =
            __lookback_init_submitter<__sub_group_size, _FlagType, _InRange, _Type, _BinaryOp, _LookbackInitKernel>{}(
                __queue, __atomic_id_ptr, __in_rng, __lookback_storage, __status_flags_size, __status_flag_padding);
    }
    sycl::event __prev_event = __single_pass_scan_submitter<__sub_group_size, __data_per_workitem, __workgroup_size,
                                                            _Type, _FlagType, _SinglePassScanKernel>{}(
        __queue, __fill_event, std::forward<_InRange>(__in_rng), std::forward<_OutRange>(__out_rng), __binary_op, __n,
        __atomic_id_ptr, __lookback_storage, __status_flags_size, __num_wgs);
    // In the single tile case, we can return the event asynchronously as we do not need to free temporary storage.
    if (__is_single_tile)
        return __prev_event;
    // TODO: Currently, the following portion of code makes this entire function synchronous.
    // Ideally, we should be able to use the asynchronous free below, but we have found that doing
    // so introduces a large unexplainable slowdown. Once this slowdown has been identified and corrected,
    // we should replace this code with the asynchronous version below.
    if (0)
    {
        return __queue.submit([=](sycl::handler& __hdl) {
            __hdl.depends_on(__prev_event);
            __hdl.host_task([=]() { sycl::free(__device_mem, __queue); });
        });
    }
    else
    {
        __prev_event.wait();
        sycl::free(__device_mem, __queue);
        return __prev_event;
    }
}

} // namespace __impl

template <typename _InRng, typename _OutRng, typename _BinaryOp, typename _KernelParam>
sycl::event
inclusive_scan(sycl::queue __queue, _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOp __binary_op,
               _KernelParam __param = {})
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    return __impl::__single_pass_scan<true>(__queue, std::move(__in_view), std::move(__out_view), __binary_op, __param);
}

template <typename _InIterator, typename _OutIterator, typename _BinaryOp, typename _KernelParam>
sycl::event
inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin,
               _BinaryOp __binary_op, _KernelParam __param = {})
{
    auto __n = __in_end - __in_begin;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    return __impl::__single_pass_scan<true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op, __param);
}

} // namespace gpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_KT_SINGLE_PASS_SCAN_H */
