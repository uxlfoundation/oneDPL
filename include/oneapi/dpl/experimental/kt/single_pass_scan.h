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
#include "../../pstl/functional_impl.h" // for oneapi::dpl::identity

#include "internal/work_group/work_group_scan.h"
#include "internal/cooperative_lookback.h"

#include <cstdint>
#include <cassert>
#include <cstddef>
#include <type_traits>

#define PRINTF(format, ...) sycl::ext::oneapi::experimental::printf(format, __VA_ARGS__)

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

template <typename... _Name>
class __lookback_init_kernel;

template <typename... _Name>
class __lookback_kernel;

template <typename _FlagType, typename _Type, typename _BinaryOp, typename _KernelName>
struct __lookback_init_submitter;

template <typename _FlagType, typename _Type, typename _BinaryOp, typename... _Name>
struct __lookback_init_submitter<_FlagType, _Type, _BinaryOp,
                                 oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _T>
    sycl::event
    operator()(sycl::queue __q, __cooperative_lookback_storage<_T> __lookback_storage, std::size_t __status_flags_size,
               std::uint16_t __status_flag_padding) const
    {
        return __q.submit([&](sycl::handler& __hdl) {
            __hdl.parallel_for<_Name...>(sycl::range<1>{__status_flags_size}, [=](const sycl::item<1>& __item) {
                auto __id = __item.get_linear_id();
                auto __identity = oneapi::dpl::unseq_backend::__known_identity<_BinaryOp, _Type>;
                __scan_status_flag<_T> __current_tile(__lookback_storage, int(__id) - int(__status_flag_padding));
                // TODO: we do not need atomics here
                if (__id < __status_flag_padding)
                    __current_tile.set_oob(__identity);
                else
                    __current_tile.set_init(__identity);
            });
        });
    }
};

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Type, typename _FlagType,
          typename _KernelName>
struct __lookback_submitter;

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Type, typename _FlagType,
          typename _InRng, typename _OutRng, typename _BinaryOp, typename _LocalAcc>
struct __lookback_kernel_func
{
    using _FlagStorageType = typename _FlagType::_FlagStorageType;
    static constexpr std::uint32_t __elems_in_tile = __workgroup_size * __data_per_workitem;

    _InRng __in_rng;
    _OutRng __out_rng;
    _BinaryOp __binary_op;
    std::size_t __n;
    std::uint32_t* __atomic_id_ptr;
    __cooperative_lookback_storage<_Type> __lookback_storage;
    std::size_t __status_flags_size;
    std::size_t __current_num_items;
    _LocalAcc __slm;

    template <bool __is_full>
    [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] void
    impl(const sycl::nd_item<1>& __item, const sycl::sub_group& __sub_group, std::uint32_t __tile_id,
         const std::size_t __work_group_offset, const std::size_t __sub_group_current_offset,
         const std::size_t __sub_group_next_offset) const
    {
        auto __sub_group_local_id = __sub_group.get_local_linear_id();
        auto __sub_group_group_id = __sub_group.get_group_linear_id();
        _Type __grf_partials[__data_per_workitem];

        // Global load into general register file
        if constexpr (__is_full)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __grf_partials[__i] =
                    __in_rng[__sub_group_current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i];
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__sub_group_current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i < __n)
                {
                    __grf_partials[__i] =
                        __in_rng[__sub_group_current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i];
                }
            }
        }
        auto __this_tile_elements = std::min<std::size_t>(__elems_in_tile, __n - __work_group_offset);
        _Type __local_reduction =
            work_group_scan<SUBGROUP_SIZE, __data_per_workitem>(item_array_order::sub_group_stride{}, __item, __slm, __grf_partials, __binary_op, __this_tile_elements);
        _Type __prev_tile_reduction{};

        // The first sub-group will query the previous tiles to find a prefix. For tile 0, we set it directly as full
        if (__tile_id == 0)
        {
            if (__item.get_local_id(0) == 0)
            {
                _FlagType __flag(__lookback_storage, __tile_id);
                __flag.set_full(__local_reduction);
            }
        }
        else if (__sub_group.get_group_id() == 0)
        {
            _FlagType __flag(__lookback_storage, __tile_id);

            if (__sub_group.get_local_id() == 0)
            {
                __flag.set_partial(__local_reduction);
            }
            __prev_tile_reduction = __cooperative_lookback(__lookback_storage, __sub_group, __tile_id, __binary_op);

            if (__sub_group.get_local_id() == 0)
            {
                __flag.set_full(__binary_op(__prev_tile_reduction, __local_reduction));
            }
        }

        __prev_tile_reduction = sycl::group_broadcast(__item.get_group(), __prev_tile_reduction, 0);

        if constexpr (__is_full)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __out_rng[__sub_group_current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i] =
                    __binary_op(__prev_tile_reduction, __grf_partials[__i]);
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__sub_group_current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i < __n)
                {
                    __out_rng[__sub_group_current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i] =
                        __binary_op(__prev_tile_reduction, __grf_partials[__i]);
                }
            }
        }
    }

    [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] void
    operator()(const sycl::nd_item<1>& __item) const
    {
        auto __group = __item.get_group();
        auto __subgroup = __item.get_sub_group();
        auto __local_id = __item.get_local_id(0);

        std::uint32_t __tile_id = 0;

        // Obtain unique ID for this work-group that will be used in decoupled lookback
        if (__group.leader())
        {
            sycl::atomic_ref<_FlagStorageType, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                __idx_atomic(*__atomic_id_ptr);
            __tile_id = __idx_atomic.fetch_add(1);
        }

        __tile_id = sycl::group_broadcast(__group, __tile_id, 0);
        auto __sub_group = __item.get_sub_group();
        auto __sub_group_local_id = __sub_group.get_local_linear_id();
        auto __sub_group_group_id = __sub_group.get_group_linear_id();

        std::size_t __work_group_offset = static_cast<std::size_t>(__tile_id) * __elems_in_tile;

        if (__work_group_offset >= __n)
            return;

        std::size_t __sub_group_current_offset = __work_group_offset + __sub_group_group_id * __data_per_workitem * SUBGROUP_SIZE;
        std::size_t __sub_group_next_offset = __sub_group_current_offset + SUBGROUP_SIZE * __data_per_workitem;
        auto __out_begin = __out_rng.begin() + __sub_group_current_offset;

        // Making full / not full case a bool template parameter and compiling two separate functions significantly improves performance
        // over run-time checks immediately before load / store. 
        if (__sub_group_next_offset <= __n)
            impl<true>(__item, __sub_group, __tile_id, __work_group_offset, __sub_group_current_offset, __sub_group_next_offset);
        else
            impl<false>(__item, __sub_group, __tile_id, __work_group_offset, __sub_group_current_offset, __sub_group_next_offset);
    }
};

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Type, typename _FlagType,
          typename... _Name>
struct __lookback_submitter<__data_per_workitem, __workgroup_size, _Type, _FlagType,
                            oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{

    template <typename _InRng, typename _OutRng, typename _BinaryOp, typename _T>
    sycl::event
    operator()(sycl::queue __q, sycl::event __prev_event, _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOp __binary_op,
               std::size_t __n, std::uint32_t* __atomic_id_ptr, __cooperative_lookback_storage<_T> __lookback_storage,
               std::size_t __status_flags_size, std::size_t __current_num_items) const
    {
        using _LocalAccessorType = __dpl_sycl::__local_accessor<_Type, 1>;
        using _KernelFunc =
            __lookback_kernel_func<__data_per_workitem, __workgroup_size, _Type, _FlagType, std::decay_t<_InRng>,
                                   std::decay_t<_OutRng>, std::decay_t<_BinaryOp>, std::decay_t<_LocalAccessorType>>;
        return __q.submit([&](sycl::handler& __hdl) {
            auto __slm = _LocalAccessorType(oneapi::dpl::__internal::__dpl_ceiling_div(__workgroup_size, SUBGROUP_SIZE), __hdl);
            __hdl.depends_on(__prev_event);
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
            __hdl.parallel_for<_Name...>(sycl::nd_range<1>(__current_num_items, __workgroup_size),
                                         _KernelFunc{__in_rng, __out_rng, __binary_op, __n, __atomic_id_ptr,
                                                     __lookback_storage, __status_flags_size, __current_num_items,
                                                     __slm});
        });
    }
};

template <bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp, typename _KernelParam>
sycl::event
__single_pass_scan(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op, _KernelParam)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;
    using _FlagType = __scan_status_flag<_Type>;
    using _FlagStorageType = typename _FlagType::_FlagStorageType;

    using _KernelName = typename _KernelParam::kernel_name;
    using _LookbackInitKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_init_kernel<_KernelName, _Type, _BinaryOp>>;
    using _LookbackKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_kernel<_KernelName, _Type, _BinaryOp>>;

    const std::size_t __n = __in_rng.size();

    if (__n == 0)
        return sycl::event{};

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");
    static_assert(oneapi::dpl::unseq_backend::__has_known_identity<_BinaryOp, _Type>::value,
                  "Only binary operators with known identity values are supported");

    assert("This device does not support 64-bit atomics" &&
           (sizeof(_Type) < 8 || __queue.get_device().has(sycl::aspect::atomic64)));

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = ::oneapi::dpl::__internal::__dpl_bit_ceil(__n);

    // Perform a single-work group scan if the input is small
    if (oneapi::dpl::__par_backend_hetero::__group_scan_fits_in_slm<_Type>(__queue, __n, __n_uniform, /*limit=*/16384))
    {
        return oneapi::dpl::__par_backend_hetero::__parallel_transform_scan_single_group<_KernelName>(
            __queue, std::forward<_InRange>(__in_rng), std::forward<_OutRange>(__out_rng), __n, oneapi::dpl::identity{},
            unseq_backend::__no_init_value<_Type>{}, __binary_op, std::true_type{});
    }

    constexpr std::size_t __workgroup_size = _KernelParam::workgroup_size;
    constexpr std::size_t __data_per_workitem = _KernelParam::data_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of workgroup_size
    std::size_t __elems_in_tile = __workgroup_size * __data_per_workitem;
    std::size_t __num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);
    constexpr int __status_flag_padding = SUBGROUP_SIZE;
    std::size_t __status_flags_size = __num_wgs + 1 + __status_flag_padding;
    const ::std::size_t __mem_bytes = __cooperative_lookback_storage<_Type>::get_reqd_storage(__status_flags_size);
    std::byte* __device_mem = reinterpret_cast<std::byte*>(sycl::malloc_device(__mem_bytes, __queue));
    if (!__device_mem)
        throw std::bad_alloc();

    // TODO: temp workaround until I figure out what's wrong
    std::uint32_t* __atomic_id_ptr = sycl::malloc_device<std::uint32_t>(1, __queue);
    __queue.fill(__atomic_id_ptr, 0, 1).wait();
    __cooperative_lookback_storage<_Type> __lookback_storage(__device_mem, __mem_bytes, __status_flags_size);
    auto __fill_event = __lookback_init_submitter<_FlagType, _Type, _BinaryOp, _LookbackInitKernel>{}(
        __queue, __lookback_storage, __status_flags_size, __status_flag_padding);

    std::size_t __current_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);
    std::size_t __current_num_items = __current_num_wgs * __workgroup_size;

    auto __prev_event =
        __lookback_submitter<__data_per_workitem, __workgroup_size, _Type, _FlagType, _LookbackKernel>{}(
            __queue, __fill_event, __in_rng, __out_rng, __binary_op, __n, __atomic_id_ptr, __lookback_storage,
            __status_flags_size, __current_num_items);

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
