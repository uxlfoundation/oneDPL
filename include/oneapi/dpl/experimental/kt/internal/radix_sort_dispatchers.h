// -*- C++ -*-
//===-- radix_sort_dispatchers.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_RADIX_SORT_DISPATCHERS_H
#define _ONEDPL_KT_RADIX_SORT_DISPATCHERS_H

#include <new>
#include <memory>
#include <cstdint>
#include <cassert>
#include <utility>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"

#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "radix_sort_submitters.h"
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
#    include "sycl_batched_one_wg_radix_sort_kernels.h"
#endif

namespace oneapi::dpl::experimental::kt::gpu::__impl
{
template <typename _KtTag, typename... _Name>
class __radix_sort_one_wg;

template <typename _KtTag, typename... _Name>
class __radix_sort_onesweep_histogram;

template <typename _KtTag, typename... _Name>
class __radix_sort_onesweep_scan;

template <typename _KtTag, typename... _Name>
class __radix_sort_onesweep;

template <typename _KtTag, typename... _Name>
class __radix_sort_onesweep_copyback;

template <typename... _Name>
class __batched_radix_sort_one_wg;

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KtTag, typename _RngPack>
sycl::event
__one_wg(_KtTag __kt_tag, sycl::queue __q, _RngPack&& __pack, std::size_t __n)
{
    using _KeyT = typename ::std::decay_t<_RngPack>::_KeyT;
    using _RadixSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__radix_sort_one_wg<_KtTag, _KernelName>>;

    return __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                         _RadixSortKernel>()(__kt_tag, __q, __pack, __pack, __n);
}

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KtTag, typename _RngPack1, typename _RngPack2>
sycl::event
__one_wg(_KtTag __kt_tag, sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __n)
{
    using _KeyT = typename ::std::decay_t<_RngPack1>::_KeyT;
    using _RadixSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__radix_sort_one_wg<_KtTag, _KernelName>>;

    return __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                         _RadixSortKernel>()(__kt_tag, __q, std::forward<_RngPack1>(__pack_in),
                                                             std::forward<_RngPack2>(__pack_out), __n);
}

template <typename _HistT, typename _KeyT, typename _ValT = void>
class __onesweep_memory_holder
{
    static constexpr bool __has_values = !::std::is_void_v<_ValT>;

    ::std::uint8_t* __m_raw_mem_ptr = nullptr;
    // Memory to store global histograms, where each stage has its own histogram
    _HistT* __m_global_hist_ptr = nullptr;
    // Memory to store group histograms, which contain offsets relative to "previous" groups
    _HistT* __m_group_hist_ptr = nullptr;
    // Memory to store intermediate results of sorting
    _KeyT* __m_keys_ptr = nullptr;
    _ValT* __m_vals_ptr = nullptr;

    ::std::size_t __m_raw_mem_bytes = 0;
    ::std::size_t __m_keys_bytes = 0;
    ::std::size_t __m_vals_bytes = 0;
    ::std::size_t __m_global_hist_bytes = 0;
    ::std::size_t __m_group_hist_bytes = 0;

    sycl::queue __m_q;


    void
    __allocate_raw_memory()
    {
        void* __mem = sycl::malloc_device(__m_raw_mem_bytes, __m_q);
        __m_raw_mem_ptr = reinterpret_cast<::std::uint8_t*>(__mem);
        if (!__m_raw_mem_ptr)
            throw std::bad_alloc();
    }

    void
    __appoint_aligned_memory_regions()
    {
        // Memory provided by the caller may not be aligned for _HistT
        void* __base_ptr = __m_raw_mem_ptr;
        std::size_t __remainder = __m_raw_mem_bytes;
        const std::size_t __hist_bytes = __m_global_hist_bytes + __m_group_hist_bytes;
        void* __aligned_ptr = std::align(std::alignment_of_v<_HistT>, __hist_bytes, __base_ptr, __remainder);
        __m_global_hist_ptr = reinterpret_cast<_HistT*>(__aligned_ptr);
        __m_group_hist_ptr = __m_global_hist_ptr + __m_global_hist_bytes / sizeof(_HistT);

        __base_ptr = reinterpret_cast<std::uint8_t*>(__aligned_ptr) + __hist_bytes;
        __remainder -= __hist_bytes;
        __aligned_ptr = std::align(std::alignment_of_v<_KeyT>, __m_keys_bytes, __base_ptr, __remainder);
        __m_keys_ptr = reinterpret_cast<_KeyT*>(__aligned_ptr);

        if constexpr (__has_values)
        {
            __base_ptr = reinterpret_cast<std::uint8_t*>(__aligned_ptr) + __m_keys_bytes;
            __remainder -= __m_keys_bytes;
            __aligned_ptr = std::align(std::alignment_of_v<_ValT>, __m_vals_bytes, __base_ptr, __remainder);
            __m_vals_ptr = reinterpret_cast<_ValT*>(__aligned_ptr);
        }
    }

    void
    __calculate_raw_memory_amount() noexcept
    {
        __m_raw_mem_bytes = __raw_memory_bytes(__m_keys_bytes / sizeof(_KeyT), __m_global_hist_bytes / sizeof(_HistT),
                                               __m_group_hist_bytes / sizeof(_HistT));
    }

  public:
    __onesweep_memory_holder(sycl::queue __q) : __m_q(__q) {}

    // Bytes of raw memory needed to hold every region, including the padding to align each of them
    static constexpr std::size_t
    __raw_memory_bytes(std::size_t __key_count, std::size_t __global_hist_item_count,
                       std::size_t __group_hist_item_count) noexcept
    {
        std::size_t __bytes = (__global_hist_item_count + __group_hist_item_count) * sizeof(_HistT) +
                              (std::alignment_of_v<_HistT> - 1) + __key_count * sizeof(_KeyT) +
                              (std::alignment_of_v<_KeyT> - 1);
        if constexpr (__has_values)
            __bytes += __key_count * sizeof(_ValT) + (std::alignment_of_v<_ValT> - 1);
        return __bytes;
    }

    void
    __keys_alloc_count(::std::size_t __key_count) noexcept
    {
        __m_keys_bytes = __key_count * sizeof(_KeyT);
    }
    void
    __vals_alloc_count(::std::size_t __values_count) noexcept
    {
        __m_vals_bytes = __values_count * sizeof(_ValT);
    }
    void
    __global_hist_item_alloc_count(::std::size_t __global_hist_item_count) noexcept
    {
        __m_global_hist_bytes = __global_hist_item_count * sizeof(_HistT);
    }
    void
    __group_hist_item_alloc_count(::std::size_t __group_hist_item_count) noexcept
    {
        __m_group_hist_bytes = __group_hist_item_count * sizeof(_HistT);
    }

    _KeyT*
    __keys_ptr() const noexcept
    {
        return __m_keys_ptr;
    }
    _ValT*
    __vals_ptr() const noexcept
    {
        return __m_vals_ptr;
    }
    _HistT*
    __global_hist_ptr() const noexcept
    {
        return __m_global_hist_ptr;
    }
    _HistT*
    __group_hist_ptr() const noexcept
    {
        return __m_group_hist_ptr;
    }

    void
    __allocate()
    {
        __calculate_raw_memory_amount();
        __allocate_raw_memory();
        __appoint_aligned_memory_regions();
    }

    // Places the regions in memory owned by the caller instead of allocating it
    void
    __assign(void* __mem, [[maybe_unused]] std::size_t __bytes)
    {
        __calculate_raw_memory_amount();
        assert(__mem != nullptr && "The temporary storage must not be null");
        assert(__bytes >= __m_raw_mem_bytes && "The temporary storage is smaller than the required number of bytes");
        __m_raw_mem_ptr = reinterpret_cast<std::uint8_t*>(__mem);
        __appoint_aligned_memory_regions();
    }

    sycl::event
    __async_deallocate(sycl::event __dep_event)
    {
        auto __dealloc_task = [__q = __m_q, __e = __dep_event, __mem = __m_raw_mem_ptr](sycl::handler& __cgh) {
            __cgh.depends_on(__e);
            __cgh.host_task([=]() { sycl::free(__mem, __q); });
        };
        return __m_q.submit(__dealloc_task);
    }
};

// Segment geometry and histogram sizes of a onesweep sort of __n elements in segments of __segment_size elements
template <typename _KeyT, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size>
struct __onesweep_layout
{
    static constexpr std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr std::uint32_t __stage_count =
        oneapi::dpl::__internal::__dpl_ceiling_div(sizeof(_KeyT) * 8, __radix_bits);

    __onesweep_segments __segments;
    std::size_t __global_hist_item_count;
    std::size_t __group_hist_item_count;

    constexpr __onesweep_layout(std::size_t __n, std::size_t __segment_size)
        : __segments{static_cast<std::uint32_t>(__segment_size), static_cast<std::uint32_t>(__n / __segment_size),
                     static_cast<std::uint32_t>(oneapi::dpl::__internal::__dpl_ceiling_div(
                         __segment_size, __work_group_size * __data_per_work_item))},
          __global_hist_item_count(std::size_t(__bin_count) * __stage_count * __segments.__segment_count),
          __group_hist_item_count(std::size_t(__bin_count) * __stage_count * __sweep_work_group_count())
    {
    }

    // Tiles never straddle a segment boundary
    constexpr std::uint32_t
    __sweep_work_group_count() const
    {
        return __segments.__segment_count * __segments.__tiles_per_segment;
    }

    // Global histograms are laid out as [stage][segment][bin] and group histograms as [stage][tile][bin]
    constexpr std::size_t
    __global_hist_stage_offset(std::uint32_t __stage) const
    {
        return std::size_t(__segments.__segment_count) * __bin_count * __stage;
    }

    constexpr std::size_t
    __group_hist_stage_offset(std::uint32_t __stage) const
    {
        return std::size_t(__sweep_work_group_count()) * __bin_count * __stage;
    }
};

// Segments are described by __layout. The non-batched sort is a single segment of __n elements.
template <typename _KernelName, bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KtTag, typename _RngPack1, typename _RngPack2, typename _RngPack3,
          typename _MemHolder, typename _Layout>
sycl::event
__onesweep_impl(_KtTag __kt_tag, sycl::queue __q, _RngPack1&& __input_pack, _RngPack2&& __virt_pack1,
                _RngPack3&& __virt_pack2, const _MemHolder& __mem_holder, std::size_t __n, const _Layout& __layout)
{
    using _RadixSortHistogram = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __radix_sort_onesweep_histogram<_KtTag, _KernelName>>;
    using _RadixSortScan = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __radix_sort_onesweep_scan<_KtTag, _KernelName>>;
    using _RadixSortSweepInitial = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __radix_sort_onesweep<_KtTag, std::decay_t<_RngPack1>, std::decay_t<_RngPack2>, _KernelName>>;
    using _RadixSortSweepEven = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __radix_sort_onesweep<_KtTag, std::decay_t<_RngPack3>, std::decay_t<_RngPack2>, _KernelName>>;
    using _RadixSortSweepOdd = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __radix_sort_onesweep<_KtTag, std::decay_t<_RngPack2>, std::decay_t<_RngPack3>, _KernelName>>;
    constexpr std::uint32_t __bin_count = _Layout::__bin_count;
    constexpr std::uint32_t __stage_count = _Layout::__stage_count;
    const __onesweep_segments& __segments = __layout.__segments;
    const std::uint32_t __sweep_work_group_count = __layout.__sweep_work_group_count();

    // TODO: check if it is more performant to fill it inside the histogram kernel
    // This line assumes that global and group histograms are stored sequentially
    sycl::event __event_chain =
        __q.memset(__mem_holder.__global_hist_ptr(), 0,
                   (__layout.__global_hist_item_count + __layout.__group_hist_item_count) * sizeof(__onesweep_hist_t));

    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    constexpr std::uint32_t __hist_work_group_count = __radix_sort_histogram_params<_KtTag>::__work_group_count;
    constexpr std::uint32_t __hist_work_group_size = __radix_sort_histogram_params<_KtTag>::__work_group_size;
    __event_chain = __radix_sort_histogram_submitter<__is_ascending, __radix_bits, __hist_work_group_count,
                                                     __hist_work_group_size, _RadixSortHistogram>()(
        __kt_tag, __q, __input_pack.__keys_rng(), __mem_holder.__global_hist_ptr(), __n, __segments, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<__stage_count, __bin_count, _RadixSortScan>()(
        __kt_tag, __q, __mem_holder.__global_hist_ptr(), __segments.__segment_count, __event_chain);

    __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item,
                                                    __work_group_size, _RadixSortSweepInitial>()(
        __kt_tag, __q, __input_pack, __virt_pack1, __mem_holder.__global_hist_ptr(), __mem_holder.__group_hist_ptr(),
        __sweep_work_group_count, __n, 0, __segments, __event_chain);

    for (::std::uint32_t __stage = 1; __stage < __stage_count; __stage++)
    {
        __onesweep_hist_t* __p_global_hist =
            __mem_holder.__global_hist_ptr() + __layout.__global_hist_stage_offset(__stage);
        __onesweep_hist_t* __p_group_hists =
            __mem_holder.__group_hist_ptr() + __layout.__group_hist_stage_offset(__stage);

        if (__stage % 2 != 0)
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item,
                                                            __work_group_size, _RadixSortSweepOdd>()(
                __kt_tag, __q, __virt_pack1, __virt_pack2, __p_global_hist, __p_group_hists, __sweep_work_group_count,
                __n, __stage, __segments, __event_chain);
        }
        else
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item,
                                                            __work_group_size, _RadixSortSweepEven>()(
                __kt_tag, __q, __virt_pack2, __virt_pack1, __p_global_hist, __p_group_hists, __sweep_work_group_count,
                __n, __stage, __segments, __event_chain);
        }
    }

    return __event_chain;
}

// Bytes of temporary memory used by a onesweep sort of __n elements in segments of __segment_size elements
template <typename _KeyT, typename _ValT, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size>
constexpr std::size_t
__onesweep_temp_bytes(std::size_t __n, std::size_t __segment_size)
{
    const __onesweep_layout<_KeyT, __radix_bits, __data_per_work_item, __work_group_size> __layout(__n, __segment_size);
    return __onesweep_memory_holder<__onesweep_hist_t, _KeyT, _ValT>::__raw_memory_bytes(
        __n, __layout.__global_hist_item_count, __layout.__group_hist_item_count);
}

// Selects internally allocated temporary memory, which is freed once the sort completes
struct __internal_temp_storage
{
};

// Selects temporary memory provided by the caller, which is neither allocated nor freed
struct __external_temp_storage
{
    void* __ptr;
    std::size_t __bytes;
};

// The input is sorted as independent segments of __segment_size elements. The temporary memory is either allocated
// internally (__internal_temp_storage) or provided by the caller (__external_temp_storage).
template <typename _KernelName, bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, bool __in_place, typename _KtTag, typename _RngPack1, typename _RngPack2,
          typename _TempStorage = __internal_temp_storage>
sycl::event
__onesweep(_KtTag __kt_tag, sycl::queue __q, _RngPack1&& __pack, _RngPack2&& __pack_out, std::size_t __n,
           std::size_t __segment_size, _TempStorage __temp_storage = {})
{
    using _KeyT = typename std::decay_t<_RngPack1>::_KeyT;
    using _ValT = typename std::decay_t<_RngPack1>::_ValT;
    constexpr bool __has_values = std::decay_t<_RngPack1>::__has_values;
    constexpr bool __internal_alloc = std::is_same_v<_TempStorage, __internal_temp_storage>;

    using _RadixSortCopyback = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __radix_sort_onesweep_copyback<_KtTag, _KernelName>>;

    using _Layout = __onesweep_layout<_KeyT, __radix_bits, __data_per_work_item, __work_group_size>;
    constexpr std::uint32_t __stage_count = _Layout::__stage_count;
    const _Layout __layout(__n, __segment_size);

    // Memory is not going to be allocated for void value type
    // TODO: make this more explicit to reduce coupling between __onesweep_memory_holder and __rng_pack
    __onesweep_memory_holder<__onesweep_hist_t, _KeyT, _ValT> __mem_holder(__q);

    __mem_holder.__keys_alloc_count(__n);
    if constexpr (__has_values)
    {
        __mem_holder.__vals_alloc_count(__n);
    }
    __mem_holder.__global_hist_item_alloc_count(__layout.__global_hist_item_count);
    __mem_holder.__group_hist_item_alloc_count(__layout.__group_hist_item_count);
    if constexpr (__internal_alloc)
        __mem_holder.__allocate();
    else
        __mem_holder.__assign(__temp_storage.__ptr, __temp_storage.__bytes);

    auto __get_tmp_pack = [&]() {
        auto __keys_tmp_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
        auto __keys_tmp_rng = __keys_tmp_keep(__mem_holder.__keys_ptr(), __mem_holder.__keys_ptr() + __n).all_view();

        if constexpr (__has_values)
        {
            auto __vals_tmp_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
            auto __vals_tmp_rng =
                __vals_tmp_keep(__mem_holder.__vals_ptr(), __mem_holder.__vals_ptr() + __n).all_view();
            return __rng_pack(::std::move(__keys_tmp_rng), ::std::move(__vals_tmp_rng));
        }
        else
        {
            return __rng_pack(::std::move(__keys_tmp_rng));
        }
    };
    auto __tmp_pack = __get_tmp_pack();
    auto __select_pack = [](const auto& __pack1, const auto& __pack2) -> const auto& {
        if constexpr (__in_place || (__stage_count % 2 == 0))
            return __pack1;
        else
            return __pack2;
    };

    const auto& __virt_pack1 = __select_pack(__tmp_pack, __pack_out);
    const auto& __virt_pack2 = __select_pack(__pack_out, __tmp_pack);
    sycl::event __event_chain =
        __onesweep_impl<_KernelName, __is_ascending, __radix_bits, __data_per_work_item, __work_group_size>(
            __kt_tag, __q, __pack, __virt_pack1, __virt_pack2, __mem_holder, __n, __layout);

    if constexpr (__in_place && (__stage_count % 2 != 0))
    {
        __event_chain = __radix_sort_copyback_submitter<_RadixSortCopyback>()(__kt_tag, __q, __tmp_pack, __pack, __n,
                                                                              __event_chain);
    }

    if constexpr (__internal_alloc)
        return __mem_holder.__async_deallocate(__event_chain);
    else
        return __event_chain;
}

// TODO: allow calling it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <bool __is_ascending, std::uint8_t __radix_bits, bool __in_place, typename _KtTag, typename _RngPack1,
          typename _RngPack2, typename _KernelParam>
sycl::event
__radix_sort(_KtTag __kt_tag, sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, _KernelParam)
{
    const auto __n = __pack_in.__keys_rng().size();
    assert(__n > 0);

    _PRINT_INFO_IN_DEBUG_MODE(__q);
    constexpr auto __data_per_workitem = _KernelParam::data_per_workitem;
    constexpr auto __workgroup_size = _KernelParam::workgroup_size;
    using _KernelName = typename _KernelParam::kernel_name;

    // TODO: enable sort_by_key for one-work-group implementation
    if constexpr (::std::decay_t<_RngPack1>::__has_values)
    {
        return __onesweep<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size, __in_place>(
            __kt_tag, __q, std::forward<_RngPack1>(__pack_in), std::forward<_RngPack2>(__pack_out), __n,
            /*__segment_size=*/__n);
    }
    else
    {
        // At the time of writing, we have decided not to add a SYCL single work-group implementation that is dispatched
        // at runtime.
        if constexpr (std::is_same_v<_KtTag, __esimd_tag>)
        {
            constexpr std::uint32_t __one_wg_cap = __data_per_workitem * __workgroup_size;
            if (__n <= __one_wg_cap)
            {
                // TODO: support different RadixBits values (only 7, 8, 9 are currently supported for ESIMD)
                // TODO: support more granular DataPerWorkItem and WorkGroupSize

                return __one_wg<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
                    __kt_tag, __q, std::forward<_RngPack1>(__pack_in), std::forward<_RngPack2>(__pack_out), __n);
            }
        }
        // TODO: avoid kernel duplication (generate the output storage with the same type as input storage and use swap)
        // TODO: support different RadixBits
        // TODO: support more granular DataPerWorkItem and WorkGroupSize
        return __onesweep<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size, __in_place>(
            __kt_tag, __q, std::forward<_RngPack1>(__pack_in), std::forward<_RngPack2>(__pack_out), __n,
            /*__segment_size=*/__n);
    }
}

#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
// Batched sort: each consecutive run of __segment_size elements is sorted independently in one onesweep pass
template <bool __is_ascending, std::uint8_t __radix_bits, bool __in_place, typename _RngPack1, typename _RngPack2,
          typename _KernelParam, typename _TempStorage>
sycl::event
__batched_onesweep(sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __segment_size,
                   _KernelParam, _TempStorage __temp_storage)
{
    const std::size_t __n = __pack_in.__keys_rng().size();
    assert(__n > 0);

    _PRINT_INFO_IN_DEBUG_MODE(__q);
    return __onesweep<typename _KernelParam::kernel_name, __is_ascending, __radix_bits, _KernelParam::data_per_workitem,
                      _KernelParam::workgroup_size, __in_place>(__sycl_tag{}, __q, std::forward<_RngPack1>(__pack_in),
                                                                std::forward<_RngPack2>(__pack_out), __n,
                                                                __segment_size, __temp_storage);
}

// Batched sort: each consecutive run of __segment_size elements, which must fit in a work-group tile, is sorted
// independently by the one work-group kernel
template <bool __is_ascending, std::uint8_t __radix_bits, typename _RngPack1, typename _RngPack2, typename _KernelParam>
sycl::event
__batched_single_group(sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __segment_size,
                       _KernelParam)
{
    const std::size_t __n = __pack_in.__keys_rng().size();
    assert(__n > 0);
    assert(__segment_size <= std::size_t(_KernelParam::data_per_workitem) * _KernelParam::workgroup_size &&
           "radix_single_group_tag requires the segment size not to exceed data_per_workitem * workgroup_size");

    _PRINT_INFO_IN_DEBUG_MODE(__q);
    using _KernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __batched_radix_sort_one_wg<std::decay_t<_RngPack1>, std::decay_t<_RngPack2>,
                                    typename _KernelParam::kernel_name>>;
    return __batched_one_wg_radix_sort_submitter<__is_ascending, __radix_bits, _KernelParam::data_per_workitem,
                                                 _KernelParam::workgroup_size, _KernelName>()(
        __q, std::forward<_RngPack1>(__pack_in), std::forward<_RngPack2>(__pack_out), __n,
        static_cast<std::uint32_t>(__segment_size));
}
#endif // _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_RADIX_SORT_DISPATCHERS_H
