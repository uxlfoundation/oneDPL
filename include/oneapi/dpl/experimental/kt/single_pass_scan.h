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
class __lookback_init_kernel;

template <typename... _Name>
class __lookback_kernel;

static constexpr int SUBGROUP_SIZE = 32;

template <typename _T>
struct __can_combine_status_prefix_flags : std::bool_constant<sizeof(_T) <= 4 && std::is_trivially_copyable_v<_T>>
{
};

template <typename _T, typename = void>
struct __cooperative_lookback_storage;

template <typename _T>
struct __cooperative_lookback_storage<_T, std::enable_if_t<__can_combine_status_prefix_flags<_T>::value>>
{
    using _PackedStatusPrefixT = std::uint64_t;
    __cooperative_lookback_storage(std::byte* __device_mem, std::size_t /*__mem_bytes*/,
                                   std::size_t /*__status_flags_size*/)
        : __packed_flags_begin(reinterpret_cast<_PackedStatusPrefixT*>(__device_mem))
    {
    }

    static std::size_t
    get_reqd_storage(std::size_t __status_flags_size)
    {
        return __status_flags_size * sizeof(_PackedStatusPrefixT);
    }

    _PackedStatusPrefixT* __packed_flags_begin;
};

template <typename _T>
struct __cooperative_lookback_storage<_T, std::enable_if_t<!__can_combine_status_prefix_flags<_T>::value>>
{
    using _FlagStorageType = std::uint32_t;
    __cooperative_lookback_storage(std::byte* __device_mem, std::size_t __mem_bytes, std::size_t __status_flags_size)
    {
        std::size_t __status_flags_bytes = __status_flags_size * sizeof(_FlagStorageType);
        std::size_t __status_vals_full_offset_bytes = __status_flags_size * sizeof(_T);
        __flags_begin = reinterpret_cast<_FlagStorageType*>(__device_mem);
        std::size_t __remainder = __mem_bytes - __status_flags_bytes;
        void* __vals_base_ptr = reinterpret_cast<void*>(__device_mem + __status_flags_bytes);
        void* __vals_aligned_ptr =
            std::align(std::alignment_of_v<_T>, __status_vals_full_offset_bytes, __vals_base_ptr, __remainder);
        __full_vals_begin = reinterpret_cast<_T*>(__vals_aligned_ptr);
        __partial_vals_begin = reinterpret_cast<_T*>(__full_vals_begin + __status_vals_full_offset_bytes / sizeof(_T));
    }

    static std::size_t
    get_reqd_storage(std::size_t __status_flags_size)
    {
        std::size_t __mem_align_pad = sizeof(_T);
        std::size_t __status_flags_bytes = __status_flags_size * sizeof(_FlagStorageType);
        std::size_t __status_vals_full_offset_bytes = __status_flags_size * sizeof(_T);
        std::size_t __status_vals_partial_offset_bytes = __status_flags_size * sizeof(_T);
        std::size_t __mem_bytes = __status_flags_bytes + __status_vals_full_offset_bytes +
                                  __status_vals_partial_offset_bytes + __mem_align_pad;
        return __mem_bytes;
    }

    _FlagStorageType* __flags_begin;
    _T* __full_vals_begin;
    _T* __partial_vals_begin;
};

template <typename _T, typename = void>
struct __scan_status_flag;

template <typename _T>
struct __scan_status_flag<_T, std::enable_if_t<__can_combine_status_prefix_flags<_T>::value>>
{
    using _PackedStatusPrefixT = std::uint64_t;
    using _FlagStorageType = std::uint32_t;
    using _AtomicPackedStatusPrefixT =
        sycl::atomic_ref<_PackedStatusPrefixT, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>;
    static constexpr _PackedStatusPrefixT __initialized_status = 0;
    static constexpr _PackedStatusPrefixT __partial_status = 1;
    static constexpr _PackedStatusPrefixT __full_status = 2;
    static constexpr _PackedStatusPrefixT __oob_status = 3;
    static constexpr int __padding = SUBGROUP_SIZE;

    template <typename _TileIdT>
    __scan_status_flag(const __cooperative_lookback_storage<_T> __temp_storage, const _TileIdT __tile_id)
        : __atomic_packed_flag(*(__temp_storage.__packed_flags_begin + __tile_id + __padding))
    {
    }

    void
    set_partial(const _T __val)
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __partial_status;
        __packed_flag |= _PackedStatusPrefixT(__val) << __shift_factor;
        __atomic_packed_flag.store(__packed_flag, sycl::memory_order::release);
    }

    void
    set_full(const _T __val)
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __full_status;
        __packed_flag |= _PackedStatusPrefixT(__val) << __shift_factor;
        __atomic_packed_flag.store(__packed_flag, sycl::memory_order::release);
    }

    void
    set_oob(const _T __known_identity)
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __oob_status;
        __packed_flag |= _PackedStatusPrefixT{__known_identity} << __shift_factor;
        __atomic_packed_flag.store(__packed_flag, sycl::memory_order::release);
    }

    void
    set_init(const _T __known_identity)
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __initialized_status;
        __packed_flag |= _PackedStatusPrefixT{__known_identity} << __shift_factor;
        __atomic_packed_flag.store(__packed_flag, sycl::memory_order::release);
    }

    auto
    get_status(_PackedStatusPrefixT __packed) const
    {
        constexpr int __shift_factor = sizeof(_PackedStatusPrefixT) * 4;
        _PackedStatusPrefixT __prefix_mask = ~_PackedStatusPrefixT(0) >> __shift_factor;
        return _FlagStorageType(__packed & __prefix_mask);
    }

    auto
    get_value(_PackedStatusPrefixT __packed) const
    {
        constexpr int __shift_factor = sizeof(_PackedStatusPrefixT) * 4;
        _PackedStatusPrefixT __prefix_mask = ~_PackedStatusPrefixT(0) << __shift_factor;
        return _T((__packed & __prefix_mask) >> __shift_factor);
    }

    std::pair<_FlagStorageType, _T>
    spin_and_get(const sycl::sub_group& __sub_group) const
    {
        _PackedStatusPrefixT __tile_status_prefix;
        _FlagStorageType __tile_flag = __initialized_status;
        // Load flag from a previous tile based on my local id.
        // Spin until every work-item in this subgroup reads a valid status
        do
        {
            __tile_status_prefix = __atomic_packed_flag.load(sycl::memory_order::acquire);
            __tile_flag = get_status(__tile_status_prefix);
        } while (!sycl::all_of_group(__sub_group, __tile_flag != __initialized_status));
        _T __value = get_value(__tile_status_prefix);
        return {__tile_flag, __tile_flag};
    }

    _AtomicPackedStatusPrefixT __atomic_packed_flag;
};

template <typename _T>
struct __scan_status_flag<_T, std::enable_if_t<!__can_combine_status_prefix_flags<_T>::value>>
{
    using _FlagStorageType = uint32_t;
    using _AtomicFlagT = sycl::atomic_ref<_FlagStorageType, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                          sycl::access::address_space::global_space>;
    using _AtomicValueT = sycl::atomic_ref<_T, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>;

    static constexpr _FlagStorageType __initialized_status = 0;
    static constexpr _FlagStorageType __partial_status = 1;
    static constexpr _FlagStorageType __full_status = 2;
    static constexpr _FlagStorageType __oob_status = 3;

    static constexpr int __padding = SUBGROUP_SIZE;

    template <typename _TileIdT>
    __scan_status_flag(const __cooperative_lookback_storage<_T>& __temp_storage, const _TileIdT __tile_id)
        : __atomic_flag(*(__temp_storage.__flags_begin + __tile_id + __padding)),
          __atomic_partial_value(*(__temp_storage.__partial_vals_begin + __tile_id + __padding)),
          __atomic_full_value(*(__temp_storage.__full_vals_begin + __tile_id + __padding))
    {
    }

    void
    set_partial(const _T __val)
    {
        __atomic_partial_value.store(__val, sycl::memory_order::release);
        __atomic_flag.store(__partial_status, sycl::memory_order::release);
    }

    void
    set_full(const _T __val)
    {
        __atomic_full_value.store(__val, sycl::memory_order::release);
        __atomic_flag.store(__full_status, sycl::memory_order::release);
    }

    void
    set_init(const _T __known_identity)
    {
        __atomic_partial_value.store(__known_identity, sycl::memory_order::release);
        __atomic_flag.store(__initialized_status, sycl::memory_order::release);
    }

    void
    set_oob(const _T __known_identity)
    {
        __atomic_partial_value.store(__known_identity, sycl::memory_order::release);
        __atomic_flag.store(__oob_status, sycl::memory_order::release);
    }

    _FlagStorageType
    get_status() const
    {
        return __atomic_flag.load(sycl::memory_order::acquire);
    }

    _T
    get_value(_FlagStorageType __status) const
    {
        return __status == __full_status ? __atomic_full_value.load(sycl::memory_order::acquire)
                                         : __atomic_partial_value.load(sycl::memory_order::acquire);
    }

    std::pair<_FlagStorageType, _T>
    spin_and_get(const sycl::sub_group& __sub_group) const
    {
        _FlagStorageType __tile_flag;
        // Load flag from a previous tile based on my local id.
        // Spin until every work-item in this subgroup reads a valid status
        do
        {
            __tile_flag = __atomic_flag.load(sycl::memory_order::acquire);
        } while (!sycl::all_of_group(__sub_group, __tile_flag != __initialized_status));
        _T __tile_value = get_value(__tile_flag);
        return {__tile_flag, __tile_value};
    }

    _AtomicFlagT __atomic_flag;
    _AtomicValueT __atomic_partial_value;
    _AtomicValueT __atomic_full_value;
};

template <typename _Subgroup, typename _T, typename _BinaryOp>
_T
cooperative_lookback(__cooperative_lookback_storage<_T> __lookback_storage, const _Subgroup& __subgroup,
                     std::uint32_t __tile_id, _BinaryOp __binary_op)
{
    _T __running = oneapi::dpl::unseq_backend::__known_identity<_BinaryOp, _T>;
    auto __local_id = __subgroup.get_local_id();

    for (int __tile = static_cast<int>(__tile_id) - 1; __tile >= 0; __tile -= SUBGROUP_SIZE)
    {
        int t = __tile - int(__local_id);
        __scan_status_flag<_T> __current_tile(__lookback_storage, t);
        const auto [__tile_flag, __tile_value] = __current_tile.spin_and_get(__subgroup);

        bool __is_full = __tile_flag == __scan_status_flag<_T>::__full_status;
        auto __is_full_ballot = sycl::ext::oneapi::group_ballot(__subgroup, __is_full);
        std::uint32_t __is_full_ballot_bits{};
        __is_full_ballot.extract_bits(__is_full_ballot_bits);

        auto __lowest_item_with_full = sycl::ctz(__is_full_ballot_bits);
        _T __contribution = __local_id <= __lowest_item_with_full
                                ? __tile_value
                                : oneapi::dpl::unseq_backend::__known_identity<_BinaryOp, _T>;

        // Running reduction of all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
        __running = __binary_op(__running, sycl::reduce_over_group(__subgroup, __contribution, __binary_op));

        // If we found a full value, we can stop looking at previous tiles. Otherwise,
        // keep going through tiles until we either find a full tile or we've completely
        // recomputed the prefix using partial values
        if (__is_full_ballot_bits)
            break;
    }
    return __running;
}

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
                __scan_status_flag<_T> __current_tile(__lookback_storage, __id);
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

        _Type __grf_partials[__data_per_workitem];

        auto __sub_group = __item.get_sub_group();
        auto __sub_group_local_id = __sub_group.get_local_linear_id();
        auto __sub_group_group_id = __sub_group.get_group_linear_id();

        std::size_t __work_group_offset = static_cast<std::size_t>(__tile_id) * __elems_in_tile;
        std::size_t __current_offset = __work_group_offset + __sub_group_group_id * __data_per_workitem * SUBGROUP_SIZE;
        std::size_t __next_offset = __current_offset + SUBGROUP_SIZE * __data_per_workitem;
        auto __out_begin = __out_rng.begin() + __current_offset;

        if (__work_group_offset >= __n)
            return;

        // Global load into general register file
        if (__next_offset <= __n)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __grf_partials[__i] =
                    __in_rng[__current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i];
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i < __n)
                {
                    __grf_partials[__i] =
                        __in_rng[__current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i];
                }
            }
        }
        auto __this_tile_elements = std::min<std::size_t>(__elems_in_tile, __n - __work_group_offset);
        _Type __local_reduction =
            work_group_scan<SUBGROUP_SIZE, __data_per_workitem>(item_array_order::sub_group_stride{}, __item, __slm, __grf_partials, __binary_op, __this_tile_elements);
        _Type __prev_tile_reduction{};

        // The first sub-group will query the previous tiles to find a prefix
        if (__tile_id == 0)
        {
            if (__item.get_local_id(0) == 0)
            {
                _FlagType __flag(__lookback_storage, __tile_id);
                __flag.set_full(__local_reduction);
            }
        }
        else if (__subgroup.get_group_id() == 0)
        {
            _FlagType __flag(__lookback_storage, __tile_id);

            if (__subgroup.get_local_id() == 0)
            {
                __flag.set_partial(__local_reduction);
            }
            __prev_tile_reduction = cooperative_lookback(__lookback_storage, __subgroup, __tile_id, __binary_op);

            if (__subgroup.get_local_id() == 0)
            {
                __flag.set_full(__binary_op(__prev_tile_reduction, __local_reduction));
            }
        }

        __prev_tile_reduction = sycl::group_broadcast(__group, __prev_tile_reduction, 0);

        if (__next_offset <= __n)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                __out_rng[__current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i] =
                    __binary_op(__prev_tile_reduction, __grf_partials[__i]);
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_workitem; ++__i)
            {
                if (__current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i < __n)
                {
                    __out_rng[__current_offset + __sub_group_local_id + SUBGROUP_SIZE * __i] =
                        __binary_op(__prev_tile_reduction, __grf_partials[__i]);
                }
            }
        }
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

    std::uint32_t* __atomic_id_ptr = reinterpret_cast<std::uint32_t*>(__device_mem + __mem_bytes - 4);
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
