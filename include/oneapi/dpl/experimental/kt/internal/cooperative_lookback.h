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
//
#ifndef _ONEDPL_KT_COOPERATIVE_LOOKBACK_H
#define _ONEDPL_KT_COOPERATIVE_LOOKBACK_H

#include <cstdint>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <sycl/sycl.hpp>

#include "sub_group/sub_group_scan.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

// TODO: we should probably remove a hardcoded constant here
static constexpr int SUBGROUP_SIZE = 32;

// Some hardware may support atomic operations over vector types enabling support for types larger than
// 4-bytes but this is not supported in SYCL.
template <typename _T>
struct __can_combine_status_prefix_flags
    : std::bool_constant<(sizeof(_T) == 1 || sizeof(_T) == 2 || sizeof(_T) == 4) && std::is_trivially_copyable_v<_T>>
{
};

template <typename _T, typename = void>
struct __scan_status_flag;

template <typename _T>
struct __scan_status_flag<_T, std::enable_if_t<__can_combine_status_prefix_flags<_T>::value>>
{
    using _PackedStatusPrefixT = std::conditional_t<sizeof(_T) == 4, std::uint64_t, std::uint32_t>;
    using _TileIdxT = uint32_t;
    using _FlagStorageType = std::conditional_t<sizeof(_T) == 4, std::uint32_t, std::uint16_t>;
    using _TIntegralBitsType = std::conditional_t<sizeof(_T) == 4, std::uint32_t,
                                                  std::conditional_t<sizeof(_T) == 2, std::uint16_t, std::uint8_t>>;
    using _AtomicPackedStatusPrefixT =
        sycl::atomic_ref<_PackedStatusPrefixT, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>;

    static constexpr _FlagStorageType __initialized_status = 0;
    static constexpr _FlagStorageType __partial_status = 1;
    static constexpr _FlagStorageType __full_status = 2;
    static constexpr _FlagStorageType __oob_status = 3;
    static constexpr int __padding = SUBGROUP_SIZE;

    struct storage
    {
        storage(std::byte* __device_mem, std::size_t /*__mem_bytes*/, std::size_t /*__status_flags_size*/)
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

    __scan_status_flag(const storage& __temp_storage, const int __tile_id)
        : __atomic_packed_flag(*(__temp_storage.__packed_flags_begin + __tile_id + __padding))
    {
    }

    void
    set_partial(const _T __val)
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __partial_status;
        _PackedStatusPrefixT __integral_bits =
            static_cast<_PackedStatusPrefixT>(sycl::bit_cast<_TIntegralBitsType, _T>(__val));
        __packed_flag |= __integral_bits << __shift_factor;
        __atomic_packed_flag.store(__packed_flag);
    }

    void
    set_full(const _T __val)
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __full_status;
        _PackedStatusPrefixT __integral_bits =
            static_cast<_PackedStatusPrefixT>(sycl::bit_cast<_TIntegralBitsType, _T>(__val));
        __packed_flag |= __integral_bits << __shift_factor;
        __atomic_packed_flag.store(__packed_flag);
    }

    void
    set_oob()
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __oob_status;
        __atomic_packed_flag.store(__packed_flag);
    }

    void
    set_init()
    {
        constexpr int __shift_factor = 4 * sizeof(_PackedStatusPrefixT);
        _PackedStatusPrefixT __packed_flag = __initialized_status;
        __atomic_packed_flag.store(__packed_flag);
    }

    auto
    get_status(_PackedStatusPrefixT __packed) const
    {
        constexpr int __shift_factor = sizeof(_PackedStatusPrefixT) * 4;
        _PackedStatusPrefixT __prefix_mask = ~_PackedStatusPrefixT(0) >> __shift_factor;
        return static_cast<_FlagStorageType>(__packed & __prefix_mask);
    }

    auto
    get_value(_PackedStatusPrefixT __packed) const
    {
        constexpr int __shift_factor = sizeof(_PackedStatusPrefixT) * 4;
        _PackedStatusPrefixT __prefix_mask = ~_PackedStatusPrefixT(0) << __shift_factor;
        _TIntegralBitsType __integral_bits =
            static_cast<_TIntegralBitsType>((__packed & __prefix_mask) >> __shift_factor);
        return sycl::bit_cast<_T, _TIntegralBitsType>(__integral_bits);
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
            if (__tile_flag == __initialized_status)
            {
                __tile_status_prefix = __atomic_packed_flag.load();
                __tile_flag = get_status(__tile_status_prefix);
            }
        } while (sycl::any_of_group(__sub_group, __tile_flag == __initialized_status));
        _T __tile_value = get_value(__tile_status_prefix);
        return {__tile_flag, __tile_value};
    }

    _AtomicPackedStatusPrefixT __atomic_packed_flag;
};

template <typename _T>
struct __scan_status_flag<_T, std::enable_if_t<!__can_combine_status_prefix_flags<_T>::value>>
{
    using _FlagStorageType = uint32_t;
    using _TileIdxT = uint32_t;
    using _AtomicFlagT = sycl::atomic_ref<_FlagStorageType, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                          sycl::access::address_space::global_space>;
    using _AtomicValueT = sycl::atomic_ref<_T, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>;

    static constexpr _FlagStorageType __initialized_status = 0;
    static constexpr _FlagStorageType __partial_status = 1;
    static constexpr _FlagStorageType __full_status = 2;
    static constexpr _FlagStorageType __oob_status = 3;

    static constexpr int __padding = SUBGROUP_SIZE;

    struct storage
    {
        storage(std::byte* __device_mem, std::size_t __mem_bytes, std::size_t __status_flags_size)
        {
            std::size_t __status_flags_bytes = __status_flags_size * sizeof(_FlagStorageType);
            std::size_t __status_vals_full_offset_bytes = __status_flags_size * sizeof(_T);
            __flags_begin = reinterpret_cast<_FlagStorageType*>(__device_mem);
            std::size_t __remainder = __mem_bytes - __status_flags_bytes;
            void* __vals_base_ptr = reinterpret_cast<void*>(__device_mem + __status_flags_bytes);
            void* __vals_aligned_ptr =
                std::align(std::alignment_of_v<_T>, __status_vals_full_offset_bytes, __vals_base_ptr, __remainder);
            __full_vals_begin = reinterpret_cast<_T*>(__vals_aligned_ptr);
            __partial_vals_begin =
                reinterpret_cast<_T*>(__full_vals_begin + __status_vals_full_offset_bytes / sizeof(_T));
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

    __scan_status_flag(const storage& __temp_storage, std::int32_t __tile_id)
        : __atomic_flag(*(__temp_storage.__flags_begin + __tile_id + __padding)),
          __atomic_partial_value(*(__temp_storage.__partial_vals_begin + __tile_id + __padding)),
          __atomic_full_value(*(__temp_storage.__full_vals_begin + __tile_id + __padding))
    {
    }

    void
    set_partial(const _T __val)
    {
        __atomic_partial_value.store(__val);
        __atomic_flag.store(__partial_status);
    }

    void
    set_full(const _T __val)
    {
        __atomic_full_value.store(__val);
        __atomic_flag.store(__full_status);
    }

    void
    set_init()
    {
        __atomic_partial_value.store(_T{});
        __atomic_flag.store(__initialized_status);
    }

    void
    set_oob()
    {
        __atomic_partial_value.store(_T{});
        __atomic_flag.store(__oob_status);
    }

    _FlagStorageType
    get_status() const
    {
        return __atomic_flag.load();
    }

    _T
    get_value(_FlagStorageType __status) const
    {
        return __status == __full_status ? __atomic_full_value.load() : __atomic_partial_value.load();
    }

    std::pair<_FlagStorageType, _T>
    spin_and_get(const sycl::sub_group& __sub_group) const
    {
        _FlagStorageType __tile_flag = __initialized_status;
        // Load flag from a previous tile based on my local id.
        // Spin until every work-item in this subgroup reads a valid status
        do
        {
            if (__tile_flag == __initialized_status)
                __tile_flag = __atomic_flag.load();
        } while (sycl::any_of_group(__sub_group, __tile_flag == __initialized_status));
        _T __tile_value = get_value(__tile_flag);
        return {__tile_flag, __tile_value};
    }

    _AtomicFlagT __atomic_flag;
    _AtomicValueT __atomic_partial_value;
    _AtomicValueT __atomic_full_value;
};

template <typename _T, typename _Subgroup, typename _BinaryOp>
_T
__cooperative_lookback(typename __scan_status_flag<_T>::storage __lookback_storage, const _Subgroup& __subgroup,
                       std::uint32_t __tile_id, _BinaryOp __binary_op)
{
    _T __running{};
    auto __local_id = __subgroup.get_local_id();
    auto __lookback_iter = [&](auto __is_initialized, int __tile) {
        __scan_status_flag<_T> __current_tile(__lookback_storage, __tile - __local_id);
        auto [__tile_flag, __tile_value] = __current_tile.spin_and_get(__subgroup);

        bool __is_full = __tile_flag == __scan_status_flag<_T>::__full_status;
        auto __is_full_ballot = sycl::ext::oneapi::group_ballot(__subgroup, __is_full);
        std::uint32_t __is_full_ballot_bits{};
        __is_full_ballot.extract_bits(__is_full_ballot_bits);

        auto __lowest_item_with_full = sycl::ctz(__is_full_ballot_bits);

        // If we found a full value, we can stop looking at previous tiles. Otherwise,
        // keep going through tiles until we either find a full tile or we've completely
        // recomputed the prefix using partial values
        if (__is_full_ballot_bits)
        {
            __sub_group_scan_partial<SUBGROUP_SIZE, true, decltype(__is_initialized)::value>(
                __subgroup, __tile_value, __binary_op, __running, __lowest_item_with_full + 1);
            return true;
        }
        else
        {
            __sub_group_scan<SUBGROUP_SIZE, true, decltype(__is_initialized)::value>(__subgroup, __tile_value,
                                                                                     __binary_op, __running);
            return false;
        }
    };
    int __tile = static_cast<int>(__tile_id) - 1;
    bool __full_tile_found = false;
    // If the zeroth tile never calls __cooperative_lookback, then this is unnecessary.
    if (__tile >= 0)
        __full_tile_found = __lookback_iter(/*__is_initialized*/ std::false_type{}, __tile);
    __tile -= SUBGROUP_SIZE;
    for (; __tile >= 0 && !__full_tile_found; __tile -= SUBGROUP_SIZE)
    {
        __full_tile_found = __lookback_iter(/*__is_initialized*/ std::true_type{}, __tile);
    }
    return __running;
}

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
