// -*- C++ -*-
//===-- sycl_batched_one_wg_radix_sort_kernels.h --------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_BATCHED_ONE_WG_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_SYCL_BATCHED_ONE_WG_RADIX_SORT_KERNELS_H

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <utility>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../../pstl/utils.h"

#include "radix_sort_utils.h"
#include "sycl_radix_sort_kernels.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

//-----------------------------------------------------------------------------
// One work-group batched radix sort kernel
//-----------------------------------------------------------------------------

// Sorts one or more whole segments per work-group, with all radix stages in a single kernel. Each segment is owned
// by ceil(segment_size / (32 * data_per_workitem)) consecutive sub-groups, and several segments are packed into a
// work-group when they fit. Lane l of the k-th sub-group of a segment holds the segment's elements
// k * 32 * data_per_workitem + i * 32 + l in registers; slots past the segment end hold the sort identity, which
// stays in the last bin and so at the end of the segment in every stage. Sub-groups past the last segment of the
// group own no elements but take part in every barrier.
//
// Per stage:
// 1. Each sub-group ranks its elements with ballots and leaves its per-bin counts in an SLM histogram row.
// 2. The rows of a segment are scanned per bin, and the last row of each segment is then scanned across bins, giving
//    the segment's bin offsets.
// 3. Elements are reordered through SLM, where each segment occupies the same slots it owns in registers, and read
//    back into registers.
//
// Every element of a group is loaded before any is stored, and groups only touch their own segments, so the
// input and output may alias (in-place sort).
template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __batched_one_wg_radix_sort_kernel
{
    using _KeyT = typename _InRngPack::_KeyT;
    using _ValT = typename _InRngPack::_ValT;
    static constexpr bool __has_values = _InRngPack::__has_values;

    using _LocOffsetT = std::uint16_t;
    using _GlobOffsetT = std::uint32_t;

    static constexpr std::uint32_t __sub_group_size = 32;
    static constexpr std::uint32_t __num_sub_groups = __work_group_size / __sub_group_size;
    static constexpr std::uint32_t __data_per_sub_group = __data_per_work_item * __sub_group_size;
    static constexpr std::uint32_t __tile_size = __data_per_work_item * __work_group_size;

    static constexpr std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr _LocOffsetT __mask = __bin_count - 1;
    static constexpr std::uint32_t __stage_count =
        oneapi::dpl::__internal::__dpl_ceiling_div(sizeof(_KeyT) * 8, __radix_bits);

    static_assert(__work_group_size % __sub_group_size == 0);
    static_assert(__tile_size <= std::numeric_limits<_LocOffsetT>::max(),
                  "data_per_workitem * workgroup_size must fit the 16-bit local offsets");

    static constexpr std::uint32_t
    __calc_slm_alloc()
    {
        // The reorder buffer overlaps the histograms, which are consumed before the reorder starts
        std::uint32_t __reorder_size = __tile_size * sizeof(_KeyT);
        if constexpr (__has_values)
            __reorder_size += __tile_size * sizeof(_ValT);
        constexpr std::uint32_t __hists_size = __num_sub_groups * __bin_count * sizeof(_LocOffsetT);
        return std::max(__reorder_size, __hists_size);
    }

    _GlobOffsetT __m_segment_size;
    _GlobOffsetT __m_segment_count;
    std::uint32_t __m_sub_groups_per_segment;
    std::uint32_t __m_segments_per_group;
    _InRngPack __m_in_pack;
    _OutRngPack __m_out_pack;
    sycl::local_accessor<unsigned char, 1> __m_slm;

    __batched_one_wg_radix_sort_kernel(_GlobOffsetT __segment_size, _GlobOffsetT __segment_count,
                                       std::uint32_t __sub_groups_per_segment, std::uint32_t __segments_per_group,
                                       const _InRngPack& __in_pack, const _OutRngPack& __out_pack,
                                       const sycl::local_accessor<unsigned char, 1>& __slm)
        : __m_segment_size(__segment_size), __m_segment_count(__segment_count),
          __m_sub_groups_per_segment(__sub_groups_per_segment), __m_segments_per_group(__segments_per_group),
          __m_in_pack(__in_pack), __m_out_pack(__out_pack), __m_slm(__slm)
    {
    }

    auto
    get(syclex::properties_tag) const
    {
        return syclex::properties{syclex::sub_group_size<__sub_group_size>};
    }

    void
    operator()(sycl::nd_item<1> __idx) const
    {
        const sycl::group<1> __group = __idx.get_group();
        const sycl::sub_group __sub_group = __idx.get_sub_group();
        const std::uint32_t __sg_id = __sub_group.get_group_linear_id();
        const std::uint32_t __sg_local_id = __sub_group.get_local_linear_id();

        const _GlobOffsetT __first_segment = __group.get_group_linear_id() * __m_segments_per_group;
        const std::uint32_t __group_segment_count =
            std::min<_GlobOffsetT>(__m_segments_per_group, __m_segment_count - __first_segment);

        // This sub-group's segment within the group, and its position within the segment
        const std::uint32_t __local_segment = __sg_id / __m_sub_groups_per_segment;
        const std::uint32_t __sg_in_segment = __sg_id - __local_segment * __m_sub_groups_per_segment;
        const bool __is_active = __local_segment < __group_segment_count;
        const _GlobOffsetT __segment_begin = (__first_segment + __local_segment) * __m_segment_size;
        // First element of this lane, relative to the segment start
        const std::uint32_t __lane_offset = __sg_in_segment * __data_per_sub_group + __sg_local_id;
        // The segment's first slot in SLM
        const std::uint32_t __slm_segment_offset = __local_segment * __m_sub_groups_per_segment * __data_per_sub_group;

        unsigned char* __slm_raw = __m_slm.template get_multi_ptr<sycl::access::decorated::no>().get();
        _LocOffsetT* __slm_hists = reinterpret_cast<_LocOffsetT*>(__slm_raw);
        _KeyT* __slm_keys = reinterpret_cast<_KeyT*>(__slm_raw);
        _ValT* __slm_vals = nullptr;
        if constexpr (__has_values)
            __slm_vals = reinterpret_cast<_ValT*>(__slm_raw + __tile_size * sizeof(_KeyT));

        auto __pack = __make_key_value_pack<__data_per_work_item, _KeyT, _ValT>();
        if (__is_active)
            __load(__pack, __segment_begin, __lane_offset);

        for (std::uint32_t __stage = 0; __stage < __stage_count; ++__stage)
        {
            _LocOffsetT __bins[__data_per_work_item];
            _LocOffsetT __ranks[__data_per_work_item];
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __bins[__i] = __get_bucket_scalar<__mask>(
                    oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__pack.__keys[__i]),
                    __stage * __radix_bits);
            }

            // 1. Rank within the sub-group
            if (__is_active)
                __sub_group_rank<__radix_bits>(__sub_group, __ranks, __bins, __slm_hists + __sg_id * __bin_count,
                                               __sg_local_id);
            sycl::group_barrier(__group);

            // 2. Offsets of each sub-group and bin within its segment
            if (__m_sub_groups_per_segment > 1)
            {
                __scan_rows_per_bin(__idx.get_local_linear_id(), __group_segment_count, __slm_hists);
                sycl::group_barrier(__group);
            }
            if (__sg_id < __group_segment_count)
                __scan_segment_bins(__sub_group, __sg_local_id, __segment_last_row(__sg_id, __slm_hists));
            sycl::group_barrier(__group);

            if (__is_active)
            {
                const _LocOffsetT* __bin_offsets = __segment_last_row(__local_segment, __slm_hists);
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    const _LocOffsetT __bin = __bins[__i];
                    const _LocOffsetT __offset_in_bin =
                        __sg_in_segment == 0 ? 0 : __slm_hists[(__sg_id - 1) * __bin_count + __bin];
                    __ranks[__i] += __slm_segment_offset + __bin_offsets[__bin] + __offset_in_bin;
                }
            }
            // The reorder buffer overlaps the histograms
            sycl::group_barrier(__group);

            // 3. Reorder through SLM
            if (__is_active)
            {
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    __slm_keys[__ranks[__i]] = __pack.__keys[__i];
                    if constexpr (__has_values)
                        __slm_vals[__ranks[__i]] = __pack.__vals[__i];
                }
            }
            sycl::group_barrier(__group);
            if (__is_active)
            {
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    const std::uint32_t __slm_idx = __slm_segment_offset + __lane_offset + __i * __sub_group_size;
                    __pack.__keys[__i] = __slm_keys[__slm_idx];
                    if constexpr (__has_values)
                        __pack.__vals[__i] = __slm_vals[__slm_idx];
                }
            }
            // The next stage's histograms overlap the reorder buffer
            sycl::group_barrier(__group);
        }

        if (__is_active)
            __store(__pack, __segment_begin, __lane_offset);
    }

  private:
    template <typename _KVPack>
    void
    __load(_KVPack& __pack, _GlobOffsetT __segment_begin, std::uint32_t __lane_offset) const
    {
        const auto& __keys_in = __m_in_pack.__keys_rng();
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            const std::uint32_t __idx = __lane_offset + __i * __sub_group_size;
            if (__idx < __m_segment_size)
            {
                __pack.__keys[__i] = __keys_in[__segment_begin + __idx];
                if constexpr (__has_values)
                    __pack.__vals[__i] = __m_in_pack.__vals_rng()[__segment_begin + __idx];
            }
            else
            {
                __pack.__keys[__i] = __sort_identity<_KeyT, __is_ascending>();
            }
        }
    }

    template <typename _KVPack>
    void
    __store(const _KVPack& __pack, _GlobOffsetT __segment_begin, std::uint32_t __lane_offset) const
    {
        const auto& __keys_out = __m_out_pack.__keys_rng();
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            const std::uint32_t __idx = __lane_offset + __i * __sub_group_size;
            if (__idx < __m_segment_size)
            {
                __keys_out[__segment_begin + __idx] = __pack.__keys[__i];
                if constexpr (__has_values)
                    __m_out_pack.__vals_rng()[__segment_begin + __idx] = __pack.__vals[__i];
            }
        }
    }

    _LocOffsetT*
    __segment_last_row(std::uint32_t __local_segment, _LocOffsetT* __slm_hists) const
    {
        return __slm_hists + ((__local_segment + 1) * __m_sub_groups_per_segment - 1) * __bin_count;
    }

    // Inclusive scan of each bin over the histogram rows of a segment
    void
    __scan_rows_per_bin(std::uint32_t __local_id, std::uint32_t __group_segment_count, _LocOffsetT* __slm_hists) const
    {
        for (std::uint32_t __p = __local_id; __p < __group_segment_count * __bin_count; __p += __work_group_size)
        {
            const std::uint32_t __segment = __p / __bin_count;
            _LocOffsetT* __col = __slm_hists + __segment * __m_sub_groups_per_segment * __bin_count + __p % __bin_count;
            _LocOffsetT __sum = 0;
            for (std::uint32_t __row = 0; __row < __m_sub_groups_per_segment; ++__row)
            {
                __sum += __col[__row * __bin_count];
                __col[__row * __bin_count] = __sum;
            }
        }
    }

    // Exclusive scan across the bins of a segment's totals, in place
    void
    __scan_segment_bins(const sycl::sub_group& __sub_group, std::uint32_t __sg_local_id, _LocOffsetT* __row) const
    {
        constexpr std::uint32_t __bins_per_lane =
            oneapi::dpl::__internal::__dpl_ceiling_div(__bin_count, __sub_group_size);
        const std::uint32_t __first_bin = __sg_local_id * __bins_per_lane;

        std::uint32_t __lane_sum = 0;
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __j = 0; __j < __bins_per_lane; ++__j)
        {
            if (__first_bin + __j < __bin_count)
                __lane_sum += __row[__first_bin + __j];
        }
        std::uint32_t __prefix = sycl::exclusive_scan_over_group(__sub_group, __lane_sum, sycl::plus<std::uint32_t>{});
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __j = 0; __j < __bins_per_lane; ++__j)
        {
            if (__first_bin + __j < __bin_count)
            {
                const _LocOffsetT __count = __row[__first_bin + __j];
                __row[__first_bin + __j] = __prefix;
                __prefix += __count;
            }
        }
    }
};

//-----------------------------------------------------------------------------
// Submitter
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KernelName>
struct __batched_one_wg_radix_sort_submitter;

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename... _Name>
struct __batched_one_wg_radix_sort_submitter<
    __is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, std::size_t __n,
               std::uint32_t __segment_size) const
    {
        using _KernelType = __batched_one_wg_radix_sort_kernel<__is_ascending, __radix_bits, __data_per_work_item,
                                                               __work_group_size, std::decay_t<_InRngPack>,
                                                               std::decay_t<_OutRngPack>>;
        constexpr bool __has_values = _KernelType::__has_values;

        const std::uint32_t __segment_count = __n / __segment_size;
        const std::uint32_t __sub_groups_per_segment =
            oneapi::dpl::__internal::__dpl_ceiling_div(__segment_size, _KernelType::__data_per_sub_group);
        assert(__sub_groups_per_segment <= _KernelType::__num_sub_groups);
        const std::uint32_t __segments_per_group =
            std::min(_KernelType::__num_sub_groups / __sub_groups_per_segment, __segment_count);
        const std::size_t __group_count =
            oneapi::dpl::__internal::__dpl_ceiling_div(__segment_count, __segments_per_group);

        sycl::nd_range<1> __nd_range(__group_count * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            sycl::local_accessor<unsigned char, 1> __slm(_KernelType::__calc_slm_alloc(), __cgh);
            _KernelType __kernel(__segment_size, __segment_count, __sub_groups_per_segment, __segments_per_group,
                                 std::forward<_InRngPack>(__in_pack), std::forward<_OutRngPack>(__out_pack), __slm);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_BATCHED_ONE_WG_RADIX_SORT_KERNELS_H
