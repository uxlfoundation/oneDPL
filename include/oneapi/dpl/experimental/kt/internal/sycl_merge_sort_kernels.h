// -*- C++ -*-
//===-- sycl_merge_sort_kernels.h -----------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_MERGE_SORT_KERNELS_H
#define _ONEDPL_KT_SYCL_MERGE_SORT_KERNELS_H

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <utility>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../../pstl/utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <typename... _Name>
class __batched_merge_sort_kernel_name;

// Placeholder for the values of a keys-only sort
struct __no_values
{
};

//-----------------------------------------------------------------------------
// In-register stable sort of one work-item's elements
//-----------------------------------------------------------------------------
// Odd-even transposition sort: only adjacent elements which are strictly out of order are swapped, so it is stable
// without a tie-break, and all indices are static after unrolling. It needs more compare-exchanges than a bitonic
// network, but each is cheaper than a bitonic one with the index tie-break required for stability, and it has no
// index array adding register pressure. It was measured to be as fast or faster for data_per_workitem 4 to 16.
// Only the first __count elements are sorted: the remaining slots are unused, and are never compared or moved, as no
// key can be assumed to order after every other under an arbitrary comparator.
template <bool __has_values, std::uint16_t _N, typename _KeyT, typename _ValT, typename _Compare>
inline void
__work_item_stable_sort(_KeyT (&__keys)[_N], _ValT (&__vals)[_N], std::uint32_t __count, _Compare __comp)
{
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint16_t __round = 0; __round < _N; ++__round)
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = __round % 2; __i + 1 < _N; __i += 2)
        {
            if (__i + 1 < __count && __comp(__keys[__i + 1], __keys[__i]))
            {
                std::swap(__keys[__i], __keys[__i + 1]);
                if constexpr (__has_values)
                    std::swap(__vals[__i], __vals[__i + 1]);
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Work-group merge path sort kernel
//-----------------------------------------------------------------------------

// Sorts one or more whole segments per work-group. Each segment is served by
// ceil(segment_size / data_per_workitem) consecutive work-items, each owning data_per_workitem consecutive
// elements of the segment; the last work-item of a segment may own fewer. Work-items beyond the last segment of
// the group own no elements but take part in every barrier.
//
// 1. The group's segments (contiguous in memory) are copied global -> SLM with coalesced accesses.
// 2. Each work-item loads its elements into registers, stable sorts them, then stores them back to SLM.
// 3. Merge rounds: each work-item finds the start of its output with a merge path (co-rank) binary search, merges
//    data_per_workitem elements from SLM into registers, barrier, then stores them back to SLM. Ties are taken
//    from the left run, which keeps the sort stable. Runs are clamped to the segment, so padding never takes part.
// 4. SLM -> coalesced global stores.
//
// Keys are only ordered by _Compare, which must be a strict weak ordering of the input keys. It is default constructed
// in the kernel.
//
// Every element of a group is loaded before any is stored, and groups only touch their own segments, so the
// input and output may alias (in-place sort).
template <typename _Compare, std::uint16_t __data_per_work_item, std::uint16_t __work_group_size, typename _InRngPack,
          typename _OutRngPack>
struct __batched_merge_sort_kernel
{
    using _KeyT = typename _InRngPack::_KeyT;
    using _ValT = typename _InRngPack::_ValT;
    static constexpr bool __has_values = _InRngPack::__has_values;
    using _ValStorageT = std::conditional_t<__has_values, _ValT, __no_values>;
    using _KeysSlmAcc = sycl::local_accessor<_KeyT, 1>;
    using _ValsSlmAcc = std::conditional_t<__has_values, sycl::local_accessor<_ValStorageT, 1>, __no_values>;

    std::size_t __m_segment_count;
    std::uint32_t __m_segment_size;
    std::uint32_t __m_segments_per_group;
    _InRngPack __m_in_pack;
    _OutRngPack __m_out_pack;
    _KeysSlmAcc __m_slm_keys;
    _ValsSlmAcc __m_slm_vals;

    __batched_merge_sort_kernel(std::size_t __segment_count, std::uint32_t __segment_size,
                                std::uint32_t __segments_per_group, const _InRngPack& __in_pack,
                                const _OutRngPack& __out_pack, const _KeysSlmAcc& __slm_keys,
                                const _ValsSlmAcc& __slm_vals)
        : __m_segment_count(__segment_count), __m_segment_size(__segment_size),
          __m_segments_per_group(__segments_per_group), __m_in_pack(__in_pack), __m_out_pack(__out_pack),
          __m_slm_keys(__slm_keys), __m_slm_vals(__slm_vals)
    {
    }

    void
    operator()(sycl::nd_item<1> __item) const
    {
        const auto __group = __item.get_group();
        const std::uint32_t __lid = __item.get_local_linear_id();
        const std::uint32_t __items_per_segment =
            oneapi::dpl::__internal::__dpl_ceiling_div(__m_segment_size, __data_per_work_item);

        const std::size_t __first_segment = __item.get_group_linear_id() * std::size_t(__m_segments_per_group);
        const std::uint32_t __group_segment_count =
            std::min<std::size_t>(__m_segments_per_group, __m_segment_count - __first_segment);
        const std::size_t __group_offset = __first_segment * __m_segment_size;
        const std::uint32_t __group_size = __group_segment_count * __m_segment_size;

        // This work-item's segment within the group, and its slice of the segment
        const std::uint32_t __local_segment = __lid / __items_per_segment;
        const bool __is_active = __local_segment < __group_segment_count;
        const std::uint32_t __start = (__lid - __local_segment * __items_per_segment) * __data_per_work_item;
        const std::uint32_t __count =
            __is_active ? std::min<std::uint32_t>(__data_per_work_item, __m_segment_size - __start) : 0;
        const std::uint32_t __slm_segment_offset = __is_active ? __local_segment * __m_segment_size : 0;

        const auto& __keys_in = __m_in_pack.__keys_rng();
        const auto& __keys_out = __m_out_pack.__keys_rng();
        const _Compare __comp{};

        for (std::uint32_t __i = __lid; __i < __group_size; __i += __work_group_size)
        {
            __m_slm_keys[__i] = __keys_in[__group_offset + __i];
            if constexpr (__has_values)
                __m_slm_vals[__i] = __m_in_pack.__vals_rng()[__group_offset + __i];
        }
        sycl::group_barrier(__group);

        // Slots at and past __count are unused: they are never compared, moved or stored
        _KeyT __keys[__data_per_work_item];
        _ValStorageT __vals[__data_per_work_item];
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __s = 0; __s < __data_per_work_item; ++__s)
        {
            if (__s < __count)
            {
                __keys[__s] = __m_slm_keys[__slm_segment_offset + __start + __s];
                if constexpr (__has_values)
                    __vals[__s] = __m_slm_vals[__slm_segment_offset + __start + __s];
            }
        }

        __work_item_stable_sort<__has_values>(__keys, __vals, __count, __comp);

        // data stays within work-item so no barrier required before storing
        __store_to_slm(__keys, __vals, __slm_segment_offset + __start, __count);
        sycl::group_barrier(__group);

        for (std::uint32_t __run = __data_per_work_item; __run < __m_segment_size; __run <<= 1)
        {
            if (__count > 0)
                __merge_from_slm(__keys, __vals, __slm_segment_offset, __start, __count, __run, __comp);
            // All work-items have finished reading from SLM before any stores
            sycl::group_barrier(__group);

            __store_to_slm(__keys, __vals, __slm_segment_offset + __start, __count);
            // after storing, synchronize so that work-items can safely read the updated data
            sycl::group_barrier(__group);
        }

        for (std::uint32_t __i = __lid; __i < __group_size; __i += __work_group_size)
        {
            __keys_out[__group_offset + __i] = __m_slm_keys[__i];
            if constexpr (__has_values)
                __m_out_pack.__vals_rng()[__group_offset + __i] = __m_slm_vals[__i];
        }
    }

  private:
    void
    __store_to_slm(const _KeyT (&__keys)[__data_per_work_item], const _ValStorageT (&__vals)[__data_per_work_item],
                   std::uint32_t __offset, std::uint32_t __count) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __s = 0; __s < __data_per_work_item; ++__s)
        {
            if (__s < __count)
            {
                __m_slm_keys[__offset + __s] = __keys[__s];
                if constexpr (__has_values)
                    __m_slm_vals[__offset + __s] = __vals[__s];
            }
        }
    }

    // Merges __count elements, starting at output position __start of the segment, from the pair of sorted runs of
    // length __run which contains that position.
    void
    __merge_from_slm(_KeyT (&__keys)[__data_per_work_item], _ValStorageT (&__vals)[__data_per_work_item],
                     std::uint32_t __slm_segment_offset, std::uint32_t __start, std::uint32_t __count,
                     std::uint32_t __run, const _Compare& __comp) const
    {
        const std::uint32_t __a_begin = (__start / (2 * __run)) * (2 * __run);
        const std::uint32_t __a_end = std::min(__a_begin + __run, __m_segment_size);
        const std::uint32_t __b_end = std::min(__a_begin + 2 * __run, __m_segment_size);
        const std::uint32_t __len_a = __a_end - __a_begin;
        const std::uint32_t __len_b = __b_end - __a_end;
        const std::uint32_t __diag = __start - __a_begin;
        const std::uint32_t __a_slm = __slm_segment_offset + __a_begin;
        const std::uint32_t __b_slm = __slm_segment_offset + __a_end;
        // find work item's merge path starting point along it's diagonal
        using _CountIt = oneapi::dpl::counting_iterator<std::uint32_t>;
        const _CountIt __first(__diag > __len_b ? __diag - __len_b : 0);
        const _CountIt __last(std::min(__diag, __len_a));
        auto __a_precedes = [&](std::uint32_t __i) {
            return !__comp(__m_slm_keys[__b_slm + (__diag - __i - 1)], __m_slm_keys[__a_slm + __i]);
        };
        const std::uint32_t __starting_point = *std::partition_point(__first, __last, __a_precedes);

        // SLM positions of the heads of A and B, and one past their last elements
        std::uint32_t __index_a = __a_slm + __starting_point;
        std::uint32_t __index_b = __b_slm + (__diag - __starting_point);
        const std::uint32_t __a_last = __a_slm + __len_a;
        const std::uint32_t __b_last = __b_slm + __len_b;
        // A is never empty, so both clamped positions are valid elements of the segment. A head read from a clamped
        // position belongs to an exhausted run and is never taken.
        auto __load_head = [&](std::uint32_t __pos, std::uint32_t __last) {
            return __m_slm_keys[std::min(__pos, __last - 1)];
        };

        // The heads of A and B are kept in registers, so each output reads one key from SLM: the one replacing the
        // head that was taken
        _KeyT __a = __load_head(__index_a, __a_last);
        _KeyT __b = __load_head(__index_b, __b_last);

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __s = 0; __s < __data_per_work_item; ++__s)
        {
            if (__s < __count)
            {
                // Ties are taken from A, which keeps the merge stable
                const bool __take_a = __index_a < __a_last && (__index_b >= __b_last || !__comp(__b, __a));
                __keys[__s] = __take_a ? __a : __b;
                if constexpr (__has_values)
                    __vals[__s] = __m_slm_vals[__take_a ? __index_a : __index_b];
                __index_a += __take_a;
                __index_b += !__take_a;
                if (__s + 1 < __data_per_work_item)
                {
                    const _KeyT __next = __load_head(__take_a ? __index_a : __index_b, __take_a ? __a_last : __b_last);
                    __a = __take_a ? __next : __a;
                    __b = __take_a ? __b : __next;
                }
            }
        }
    }
};

//-----------------------------------------------------------------------------
// Submitter
//-----------------------------------------------------------------------------
template <typename _Compare, std::uint16_t __data_per_work_item, std::uint16_t __work_group_size, typename _KernelName>
struct __batched_merge_sort_submitter;

template <typename _Compare, std::uint16_t __data_per_work_item, std::uint16_t __work_group_size, typename... _Name>
struct __batched_merge_sort_submitter<_Compare, __data_per_work_item, __work_group_size,
                                      oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, std::size_t __n,
               std::uint32_t __segment_size) const
    {
        using _KernelType = __batched_merge_sort_kernel<_Compare, __data_per_work_item, __work_group_size,
                                                        std::decay_t<_InRngPack>, std::decay_t<_OutRngPack>>;
        using _ValStorageT = typename _KernelType::_ValStorageT;
        constexpr bool __has_values = _KernelType::__has_values;

        const std::size_t __segment_count = __n / __segment_size;
        const std::uint32_t __items_per_segment =
            oneapi::dpl::__internal::__dpl_ceiling_div(__segment_size, __data_per_work_item);
        const std::uint32_t __segments_per_group =
            std::min<std::size_t>(__work_group_size / __items_per_segment, __segment_count);
        const std::size_t __group_count =
            oneapi::dpl::__internal::__dpl_ceiling_div(__segment_count, __segments_per_group);
        const std::size_t __slm_count = std::size_t(__segments_per_group) * __segment_size;

        sycl::nd_range<1> __nd_range(__group_count * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            typename _KernelType::_KeysSlmAcc __slm_keys(__slm_count, __cgh);
            auto __slm_vals = [&]() {
                if constexpr (__has_values)
                    return sycl::local_accessor<_ValStorageT, 1>(__slm_count, __cgh);
                else
                    return __no_values{};
            }();
            _KernelType __kernel(__segment_count, __segment_size, __segments_per_group,
                                 std::forward<_InRngPack>(__in_pack), std::forward<_OutRngPack>(__out_pack), __slm_keys,
                                 __slm_vals);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

//-----------------------------------------------------------------------------
// Dispatcher
//-----------------------------------------------------------------------------
template <typename _Compare, typename _RngPack1, typename _RngPack2, typename _KernelParam>
sycl::event
__batched_merge_sort(sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __segment_size,
                     _KernelParam)
{
    const std::size_t __n = __pack_in.__keys_rng().size();
    assert(__n > 0);
    assert(__segment_size <= std::size_t(_KernelParam::data_per_workitem) * _KernelParam::workgroup_size &&
           "merge_single_group_tag requires the segment size not to exceed data_per_workitem * workgroup_size");

    _PRINT_INFO_IN_DEBUG_MODE(__q);
    // The comparator and range pack types are part of the name, so one custom name serves every comparator and data
    // passing mechanism
    using _KernelName =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__batched_merge_sort_kernel_name<
            _Compare, std::decay_t<_RngPack1>, std::decay_t<_RngPack2>, typename _KernelParam::kernel_name>>;

    return __batched_merge_sort_submitter<_Compare, _KernelParam::data_per_workitem, _KernelParam::workgroup_size,
                                          _KernelName>()(__q, std::forward<_RngPack1>(__pack_in),
                                                         std::forward<_RngPack2>(__pack_out), __n,
                                                         static_cast<std::uint32_t>(__segment_size));
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_MERGE_SORT_KERNELS_H
