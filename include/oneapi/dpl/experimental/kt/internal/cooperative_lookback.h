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

#include "../../../pstl/utils.h"
#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h"
#include "sub_group/sub_group_scan.h"

namespace oneapi::dpl::experimental::kt
{

namespace gpu
{

namespace __impl
{

// Some hardware may support atomic operations over vector types enabling support for types larger than
// 4-bytes but this is not supported in SYCL.
template <typename _T>
struct __can_combine_status_prefix_flags
    : std::bool_constant<(sizeof(_T) == 1 || sizeof(_T) == 2 || sizeof(_T) == 4) && std::is_trivially_copyable_v<_T>>
{
};

template <std::uint8_t __sub_group_size, typename _T, typename = void>
struct __scan_status_flag;

// __scan_status_flag specialization that combines a scan tile's status and actual prefix value into a single element
// and extract with bit logic. This minimizes temporary storage requirements and number of atomic operations that need
// to be performed during updates / spinning. In particular, each tile owns 1 element of type _PackedStatusPrefixT
// across the underlying buffer where the upper bits are used to store the scan prefix and the lower bits are used to
// store the scan flag.
template <std::uint8_t __sub_group_size, typename _T>
struct __scan_status_flag<__sub_group_size, _T, std::enable_if_t<__can_combine_status_prefix_flags<_T>::value>>
{
    // For 4-byte types, we need 8-bytes per tile to implement this approach. For 2-byte and 1-byte types, only 4-bytes
    // per tile is required.
    using _PackedStatusPrefixT = std::conditional_t<sizeof(_T) == 4, std::uint64_t, std::uint32_t>;
    using _TileIdxT = std::uint32_t;
    using _FlagStorageType = std::conditional_t<sizeof(_T) == 4, std::uint32_t, std::uint16_t>;
    using _TIntegralBitsType = std::conditional_t<sizeof(_T) == 4, std::uint32_t,
                                                  std::conditional_t<sizeof(_T) == 2, std::uint16_t, std::uint8_t>>;
    using _AtomicPackedStatusPrefixT =
        sycl::atomic_ref<_PackedStatusPrefixT, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>;

    static constexpr std::uint32_t __half_status_prefix_bits = 4 * sizeof(_PackedStatusPrefixT);

    static constexpr _FlagStorageType __initialized_status = 0;
    static constexpr _FlagStorageType __partial_status = 1;
    static constexpr _FlagStorageType __full_status = 2;
    static constexpr _FlagStorageType __oob_status = 3;
    static constexpr int __padding = __sub_group_size;

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
        : __packed_flag_ref(*(__temp_storage.__packed_flags_begin + __tile_id + __padding)),
          __atomic_packed_flag(__packed_flag_ref)
    {
    }

    void
    set_partial(const _T __val)
    {
        _PackedStatusPrefixT __packed_flag = __partial_status;
        _PackedStatusPrefixT __integral_bits =
            static_cast<_PackedStatusPrefixT>(sycl::bit_cast<_TIntegralBitsType, _T>(__val));
        __packed_flag |= __integral_bits << __half_status_prefix_bits;
        __atomic_packed_flag.store(__packed_flag);
    }

    void
    set_full(const _T __val)
    {
        _PackedStatusPrefixT __packed_flag = __full_status;
        _PackedStatusPrefixT __integral_bits =
            static_cast<_PackedStatusPrefixT>(sycl::bit_cast<_TIntegralBitsType, _T>(__val));
        __packed_flag |= __integral_bits << __half_status_prefix_bits;
        __atomic_packed_flag.store(__packed_flag);
    }

    void
    set_oob(const _T __dummy)
    {
        _PackedStatusPrefixT __packed_flag = __oob_status;
        _PackedStatusPrefixT __integral_bits =
            static_cast<_PackedStatusPrefixT>(sycl::bit_cast<_TIntegralBitsType, _T>(__dummy));
        __packed_flag |= __integral_bits << __half_status_prefix_bits;
        // For initialization routines, we do not need atomicity, so we can write through the
        // reference directly.
        __packed_flag_ref = _PackedStatusPrefixT{__packed_flag};
    }

    void
    set_init(const _T __dummy)
    {
        _PackedStatusPrefixT __packed_flag = __initialized_status;
        _PackedStatusPrefixT __integral_bits =
            static_cast<_PackedStatusPrefixT>(sycl::bit_cast<_TIntegralBitsType, _T>(__dummy));
        __packed_flag |= __integral_bits << __half_status_prefix_bits;
        // For initialization routines, we do not need atomicity, so we can write through the
        // reference directly.
        __packed_flag_ref = _PackedStatusPrefixT{__packed_flag};
    }

    _FlagStorageType
    get_status(_PackedStatusPrefixT __packed) const
    {
        constexpr _PackedStatusPrefixT __prefix_mask = ~_PackedStatusPrefixT(0) >> __half_status_prefix_bits;
        return static_cast<_FlagStorageType>(__packed & __prefix_mask);
    }

    _T
    get_value(_PackedStatusPrefixT __packed) const
    {
        constexpr _PackedStatusPrefixT __prefix_mask = ~_PackedStatusPrefixT(0) << __half_status_prefix_bits;
        _TIntegralBitsType __integral_bits =
            static_cast<_TIntegralBitsType>((__packed & __prefix_mask) >> __half_status_prefix_bits);
        return sycl::bit_cast<_T, _TIntegralBitsType>(__integral_bits);
    }

    std::pair<_FlagStorageType, _T>
    spin_and_get(const __dpl_sycl::__sub_group& __sub_group) const
    {
        _PackedStatusPrefixT __tile_status_prefix;
        _FlagStorageType __tile_flag = __initialized_status;
        // Load flag from a previous tile based on my local id.
        // Spin until every work-item in this subgroup reads a valid (non-initial) status
        do
        {
            if (__tile_flag == __initialized_status)
            {
                __tile_status_prefix = __atomic_packed_flag.load();
                __tile_flag = get_status(__tile_status_prefix);
            }
        } while (__dpl_sycl::__any_of_group(__sub_group, __tile_flag == __initialized_status));
        _T __tile_value = get_value(__tile_status_prefix);
        return {__tile_flag, __tile_value};
    }

    _PackedStatusPrefixT& __packed_flag_ref;
    _AtomicPackedStatusPrefixT __atomic_packed_flag;
};

// __scan_status_flag specialization for types where we cannot combine prefix and status flag. Each tile owns 3
// elements across the underlying buffer: a status flag, a partial scan value consisting of the tile's own local
// reduction, and a full scan value consisting of the reduction of the current tile along with all preceding tiles.
template <std::uint8_t __sub_group_size, typename _T>
struct __scan_status_flag<__sub_group_size, _T, std::enable_if_t<!__can_combine_status_prefix_flags<_T>::value>>
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

    static constexpr int __padding = __sub_group_size;

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
        : __flag_ref(*(__temp_storage.__flags_begin + __tile_id + __padding)),
          __partial_value_ref(*(__temp_storage.__partial_vals_begin + __tile_id + __padding)),
          __full_value_ref(*(__temp_storage.__full_vals_begin + __tile_id + __padding)), __atomic_flag(__flag_ref),
          __atomic_partial_value(__partial_value_ref), __atomic_full_value(__full_value_ref)
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

    // For initialization routines, we do not need atomicity, so we can write through the ptr
    // member variable.
    void
    set_init(const _T __dummy)
    {
        __partial_value_ref = __dummy;
        __full_value_ref = __dummy;
        __flag_ref = _FlagStorageType{__initialized_status};
    }

    // For initialization routines, we do not need atomicity, so we can write through the ptr
    // member variable.
    void
    set_oob(const _T __dummy)
    {
        __partial_value_ref = __dummy;
        __full_value_ref = __dummy;
        __flag_ref = _FlagStorageType{__oob_status};
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
    spin_and_get(const __dpl_sycl::__sub_group& __sub_group) const
    {
        _FlagStorageType __tile_flag = __initialized_status;
        // Load flag from a previous tile based on my local id.
        // Spin until every work-item in this subgroup reads a valid (non-initial) status
        do
        {
            if (__tile_flag == __initialized_status)
                __tile_flag = __atomic_flag.load();
        } while (__dpl_sycl::__any_of_group(__sub_group, __tile_flag == __initialized_status));
        _T __tile_value = get_value(__tile_flag);
        return {__tile_flag, __tile_value};
    }

    _FlagStorageType& __flag_ref;
    _T& __partial_value_ref;
    _T& __full_value_ref;

    _AtomicFlagT __atomic_flag;
    _AtomicValueT __atomic_partial_value;
    _AtomicValueT __atomic_full_value;
};

// Function object intended to be provided to __work_group_scan as an __init_callback
template <std::uint8_t __sub_group_size, typename _T, typename _BinaryOp>
struct __cooperative_lookback
{
    void
    operator()(_T& __prefix_ref, const __dpl_sycl::__sub_group& __subgroup, _T __local_reduction) const
    {
        __scan_status_flag<__sub_group_size, _T> __local_flag(__lookback_storage, __tile_id);
        if (__subgroup.get_local_id() == 0)
        {
            __local_flag.set_partial(__local_reduction);
        }
        oneapi::dpl::__internal::__lazy_ctor_storage<_T> __running;
        oneapi::dpl::__internal::__scoped_destroyer<_T> __destroy_when_leaving_scope{__running};
        std::uint8_t __local_id = __subgroup.get_local_id();
        auto __lookback_iter = [&](auto __is_initialized, int __tile) {
            __scan_status_flag<__sub_group_size, _T> __current_tile(__lookback_storage, __tile - __local_id);
            auto [__tile_flag, __tile_value] = __current_tile.spin_and_get(__subgroup);

            bool __is_full = __tile_flag == __scan_status_flag<__sub_group_size, _T>::__full_status;
            auto __is_full_ballot = sycl::ext::oneapi::group_ballot(__subgroup, __is_full);
            std::uint32_t __is_full_ballot_bits{};
            __is_full_ballot.extract_bits(__is_full_ballot_bits);

            auto __lowest_item_with_full = sycl::ctz(__is_full_ballot_bits);

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (__is_full_ballot_bits)
            {
                oneapi::dpl::__par_backend_hetero::__sub_group_scan_partial<
                    __sub_group_size, /*__is_inclusive*/ true,
                    /*__init_present*/ decltype(__is_initialized)::value>(__subgroup, __tile_value, __binary_op,
                                                                          __running, __lowest_item_with_full + 1);
                return true;
            }
            else
            {
                oneapi::dpl::__par_backend_hetero::__sub_group_scan<
                    __sub_group_size, /*__is_inclusive*/ true,
                    /*__init_present*/ decltype(__is_initialized)::value>(__subgroup, __tile_value, __binary_op,
                                                                          __running);
                return false;
            }
        };
        int __tile = static_cast<int>(__tile_id) - 1;
        bool __full_tile_found = false;
        __full_tile_found = __lookback_iter(/*__is_initialized*/ std::false_type{}, __tile);
        __tile -= __sub_group_size;
        for (; __tile >= 0 && !__full_tile_found; __tile -= __sub_group_size)
        {
            __full_tile_found = __lookback_iter(/*__is_initialized*/ std::true_type{}, __tile);
        }
        if (__subgroup.get_local_id() == 0)
        {
            __local_flag.set_full(__binary_op(__running.__v, __local_reduction));
        }
        __prefix_ref = __running.__v;
    }
    // This callback is used for tiles after the first, so we should apply the tile prefix value.
    static constexpr bool __apply_prefix = true;
    typename __scan_status_flag<__sub_group_size, _T>::storage __lookback_storage;
    typename __scan_status_flag<__sub_group_size, _T>::_TileIdxT __tile_id;
    _BinaryOp __binary_op;
};

template <std::uint8_t __sub_group_size, typename _T>
struct __cooperative_lookback_first_tile
{
    void
    operator()(const _T& /*__dummy_prefix_ref*/, const __dpl_sycl::__sub_group& __subgroup, _T __local_reduction) const
    {
        if (__num_tiles > 1 && __subgroup.get_local_id() == 0)
        {
            __scan_status_flag<__sub_group_size, _T> __local_flag(__lookback_storage, __tile_id);
            __local_flag.set_full(__local_reduction);
        }
    }
    // This callback is used for the first tile, so there is no init to apply.
    static constexpr bool __apply_prefix = false;
    typename __scan_status_flag<__sub_group_size, _T>::storage __lookback_storage;
    typename __scan_status_flag<__sub_group_size, _T>::_TileIdxT __num_tiles;
    typename __scan_status_flag<__sub_group_size, _T>::_TileIdxT __tile_id;
};

template <typename... _Name>
class __lookback_init_kernel;

template <std::uint8_t __sub_group_size, typename _FlagType, typename _InRange, typename _Type, typename _BinaryOp,
          typename _KernelName>
struct __lookback_init_submitter;

template <std::uint8_t __sub_group_size, typename _FlagType, typename _InRange, typename _Type, typename _BinaryOp,
          typename... _Name>
struct __lookback_init_submitter<__sub_group_size, _FlagType, _InRange, _Type, _BinaryOp,
                                 oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    sycl::event
    operator()(sycl::queue __q, std::uint32_t* __atomic_id_ptr, const _InRange& __in_rng,
               typename __scan_status_flag<__sub_group_size, _Type>::storage __lookback_storage,
               std::size_t __status_flags_size, std::uint16_t __status_flag_padding) const
    {
        return __q.submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng);
            __hdl.parallel_for<_Name...>(sycl::range<1>{__status_flags_size}, [=](const sycl::item<1>& __item) {
                const std::uint32_t __id = __item.get_linear_id();
                // Negative values are valid here up until -sub_group_size for initialization of OOB elements.
                const int __id_offset = int(__id) - int(__status_flag_padding);
                // Use __in_rng[0] to ensure we have valid objects in all locations to prevent reading uninitialized
                // memory in lookback.
                __scan_status_flag<__sub_group_size, _Type> __current_tile(__lookback_storage, __id_offset);
                if (__id < __status_flag_padding)
                {
                    __current_tile.set_oob(__in_rng[0]);
                    if (__id == 0)
                        *__atomic_id_ptr = 0;
                }
                else
                    __current_tile.set_init(__in_rng[0]);
            });
        });
    }
};

} // namespace __impl
} // namespace gpu
} // namespace oneapi::dpl::experimental::kt

#endif
