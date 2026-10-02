// -*- C++ -*-
//===-- batched_merge_sort_kernels.h --------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_BATCHED_MERGE_SORT_KERNELS_H
#define _ONEDPL_KT_BATCHED_MERGE_SORT_KERNELS_H

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

#include "radix_sort_utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <typename... _Name>
class __batched_merge_sort_kernel_name;

//-----------------------------------------------------------------------------
// Parameter validation
//-----------------------------------------------------------------------------
template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size>
inline void
__check_batched_merge_sort_params([[maybe_unused]] std::size_t __n, [[maybe_unused]] std::size_t __segment_size)
{
    static_assert(__data_per_workitem > 0 && __workgroup_size > 0);
    assert(__segment_size > 0 && "The segment size must be greater than zero");
    assert(__n % __segment_size == 0 && "The number of elements must be a multiple of the segment size");
    assert(__segment_size <= std::size_t(__data_per_workitem) * __workgroup_size &&
           "The segment size must not exceed data_per_workitem * workgroup_size in batched merge sort");
}

//-----------------------------------------------------------------------------
// Key ordering
//-----------------------------------------------------------------------------

// Compares keys through the same order-preserving bit transformation as radix sort, so that both batched sorts
// produce the same order, including for floating-point keys (-0.0 and +0.0 are equal).
template <bool __is_ascending>
struct __radix_order_less
{
    template <typename _T>
    bool
    operator()(const _T& __a, const _T& __b) const
    {
        return oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__a) <
               oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__b);
    }
};

// Placeholder for the values of a keys-only sort
struct __no_values
{
};

//-----------------------------------------------------------------------------
// In-register stable sort of one work-item's elements
//-----------------------------------------------------------------------------
template <bool __has_values, std::uint16_t _N, typename _KeyT, typename _ValT, typename _Less>
inline void
__work_item_stable_sort(_KeyT (&__keys)[_N], _ValT (&__vals)[_N], _Less __less)
{
    auto __swap = [&](std::uint16_t __i, std::uint16_t __j) {
        std::swap(__keys[__i], __keys[__j]);
        if constexpr (__has_values)
            std::swap(__vals[__i], __vals[__j]);
    };

    if constexpr ((_N & (_N - 1)) == 0)
    {
        // Bitonic network with a slot index tie-break, which makes it stable: indices are unique, so no two
        // elements compare equal. All indices are static after unrolling.
        std::uint16_t __idx[_N];
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 0; __i < _N; ++__i)
            __idx[__i] = __i;

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __k = 2; __k <= _N; __k <<= 1)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint16_t __j = __k >> 1; __j > 0; __j >>= 1)
            {
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint16_t __s = 0; __s < _N; ++__s)
                {
                    const std::uint16_t __t = __s ^ __j;
                    if (__t > __s)
                    {
                        const bool __ascending = (__s & __k) == 0;
                        const bool __t_before_s = __less(__keys[__t], __keys[__s]) ||
                                                  (!__less(__keys[__s], __keys[__t]) && __idx[__t] < __idx[__s]);
                        if (__ascending == __t_before_s)
                        {
                            __swap(__s, __t);
                            std::swap(__idx[__s], __idx[__t]);
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Odd-even transposition sort: only adjacent elements which are strictly out of order are swapped,
        // so it is stable without a tie-break.
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __round = 0; __round < _N; ++__round)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint16_t __i = __round % 2; __i + 1 < _N; __i += 2)
            {
                if (__less(__keys[__i + 1], __keys[__i]))
                    __swap(__i, __i + 1);
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
// 2. Each work-item loads its elements into registers, pads unused slots with the sort identity and stable sorts
//    them.
// 3. Merge rounds: registers -> SLM, barrier, each work-item finds the start of its output with a merge path
//    (co-rank) binary search, then merges data_per_workitem elements from SLM back into registers. Ties are taken
//    from the left run, which keeps the sort stable. Runs are clamped to the segment, so padding never takes part.
// 4. Registers -> SLM -> coalesced global stores.
//
// Every element of a group is loaded before any is stored, and groups only touch their own segments, so the
// input and output may alias (in-place sort).
template <bool __is_ascending, std::uint16_t __data_per_work_item, std::uint16_t __work_group_size, typename _InRngPack,
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
        const __radix_order_less<__is_ascending> __less{};
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

        for (std::uint32_t __i = __lid; __i < __group_size; __i += __work_group_size)
        {
            __m_slm_keys[__i] = __keys_in[__group_offset + __i];
            if constexpr (__has_values)
                __m_slm_vals[__i] = __m_in_pack.__vals_rng()[__group_offset + __i];
        }
        sycl::group_barrier(__group);

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
            else
            {
                __keys[__s] = __sort_identity<_KeyT, __is_ascending>();
            }
        }

        __work_item_stable_sort<__has_values>(__keys, __vals, __less);

        for (std::uint32_t __run = __data_per_work_item; __run < __m_segment_size; __run <<= 1)
        {
            // The previous round's reads from SLM are complete
            sycl::group_barrier(__group);
            __store_to_slm(__keys, __vals, __slm_segment_offset + __start, __count);
            sycl::group_barrier(__group);

            if (__count > 0)
                __merge_from_slm(__keys, __vals, __slm_segment_offset, __start, __count, __run, __less);
        }

        sycl::group_barrier(__group);
        __store_to_slm(__keys, __vals, __slm_segment_offset + __start, __count);
        sycl::group_barrier(__group);

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
    template <typename _Less>
    void
    __merge_from_slm(_KeyT (&__keys)[__data_per_work_item], _ValStorageT (&__vals)[__data_per_work_item],
                     std::uint32_t __slm_segment_offset, std::uint32_t __start, std::uint32_t __count,
                     std::uint32_t __run, _Less __less) const
    {
        const std::uint32_t __a_begin = (__start / (2 * __run)) * (2 * __run);
        const std::uint32_t __a_end = std::min(__a_begin + __run, __m_segment_size);
        const std::uint32_t __b_end = std::min(__a_begin + 2 * __run, __m_segment_size);
        const std::uint32_t __len_a = __a_end - __a_begin;
        const std::uint32_t __len_b = __b_end - __a_end;
        const std::uint32_t __diag = __start - __a_begin;
        const std::uint32_t __a_slm = __slm_segment_offset + __a_begin;
        const std::uint32_t __b_slm = __slm_segment_offset + __a_end;

        // Co-rank: how many of the first __diag outputs come from A. Ties go to A.
        std::uint32_t __lo = __diag > __len_b ? __diag - __len_b : 0;
        std::uint32_t __hi = std::min(__diag, __len_a);
        while (__lo < __hi)
        {
            const std::uint32_t __mid = (__lo + __hi) / 2;
            if (!__less(__m_slm_keys[__b_slm + (__diag - __mid - 1)], __m_slm_keys[__a_slm + __mid]))
                __lo = __mid + 1;
            else
                __hi = __mid;
        }
        std::uint32_t __ia = __lo;
        std::uint32_t __ib = __diag - __lo;

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __s = 0; __s < __data_per_work_item; ++__s)
        {
            if (__s < __count)
            {
                const bool __take_a = __ia < __len_a && (__ib >= __len_b || !__less(__m_slm_keys[__b_slm + __ib],
                                                                                    __m_slm_keys[__a_slm + __ia]));
                const std::uint32_t __src = __take_a ? __a_slm + __ia : __b_slm + __ib;
                __keys[__s] = __m_slm_keys[__src];
                if constexpr (__has_values)
                    __vals[__s] = __m_slm_vals[__src];
                __ia += __take_a;
                __ib += !__take_a;
            }
        }
    }
};

//-----------------------------------------------------------------------------
// Submitter
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint16_t __data_per_work_item, std::uint16_t __work_group_size,
          typename _KernelName>
struct __batched_merge_sort_submitter;

template <bool __is_ascending, std::uint16_t __data_per_work_item, std::uint16_t __work_group_size, typename... _Name>
struct __batched_merge_sort_submitter<__is_ascending, __data_per_work_item, __work_group_size,
                                      oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, std::size_t __n,
               std::uint32_t __segment_size) const
    {
        using _KernelType = __batched_merge_sort_kernel<__is_ascending, __data_per_work_item, __work_group_size,
                                                        std::decay_t<_InRngPack>, std::decay_t<_OutRngPack>>;
        using _KeyT = typename _KernelType::_KeyT;
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
            sycl::local_accessor<_KeyT, 1> __slm_keys(__slm_count, __cgh);
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
template <bool __is_ascending, typename _RngPack1, typename _RngPack2, typename _KernelParam>
sycl::event
__batched_merge_sort(sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __segment_size,
                     _KernelParam)
{
    const std::size_t __n = __pack_in.__keys_rng().size();
    assert(__n > 0);

    _PRINT_INFO_IN_DEBUG_MODE(__q);
    // The range pack types are part of the name, so one custom name serves every data passing mechanism
    using _KernelName =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__batched_merge_sort_kernel_name<
            std::decay_t<_RngPack1>, std::decay_t<_RngPack2>, typename _KernelParam::kernel_name>>;

    return __batched_merge_sort_submitter<__is_ascending, _KernelParam::data_per_workitem, _KernelParam::workgroup_size,
                                          _KernelName>()(__q, std::forward<_RngPack1>(__pack_in),
                                                         std::forward<_RngPack2>(__pack_out), __n,
                                                         static_cast<std::uint32_t>(__segment_size));
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_BATCHED_MERGE_SORT_KERNELS_H
