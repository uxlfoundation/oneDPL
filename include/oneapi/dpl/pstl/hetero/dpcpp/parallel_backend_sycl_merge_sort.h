// -*- C++ -*-
//===-- parallel_backend_sycl_merge_sort.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H

#include <limits>      // std::numeric_limits
#include <cassert>     // assert
#include <utility>     // std::swap, std::pair
#include <cstdint>     // std::uint32_t, ...
#include <algorithm>   // std::min, std::max_element
#include <type_traits> // std::decay_t, std::integral_constant

#include "sycl_defs.h"                   // __dpl_sycl::__local_accessor, __dpl_sycl::__group_barrier
#include "sycl_traits.h"                 // SYCL traits specialization for some oneDPL types.
#include "../../utils.h"                 // __dpl_bit_floor, __dpl_bit_ceil
#include "../../utils_ranges.h"          // __difference_t
#include "parallel_backend_sycl_merge.h" // __find_start_point, __serial_merge

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

struct __subgroup_bubble_sorter
{
    template <typename _StorageAcc, typename _Compare>
    void
    sort(const _StorageAcc& __storage_acc, _Compare __comp, std::uint32_t __start, std::uint32_t __end) const
    {
        for (std::uint32_t i = __start; i < __end; ++i)
        {
            for (std::uint32_t j = __start + 1; j < __start + __end - i; ++j)
            {
                auto& __first_item = __storage_acc[j - 1];
                auto& __second_item = __storage_acc[j];
                if (__comp(__second_item, __first_item))
                {
                    using std::swap;
                    swap(__first_item, __second_item);
                }
            }
        }
    }
};

struct __group_merge_path_sorter
{
    template <typename _StorageAcc, typename _Compare>
    bool
    sort(const sycl::nd_item<1>& __item, const _StorageAcc& __storage_acc, _Compare __comp, std::uint32_t __start,
         std::uint32_t __end, std::uint32_t __sorted, std::uint32_t __data_per_workitem,
         std::uint32_t __workgroup_size) const
    {
        const std::uint32_t __sorted_final = __data_per_workitem * __workgroup_size;

        const std::uint32_t __id = __item.get_local_linear_id() * __data_per_workitem;

        bool __data_in_temp = false;
        std::uint32_t __next_sorted = __sorted * 2;
        // ctz precisely calculates log2 of an integral value which is a power of 2, while
        // std::log2 may be prone to rounding errors on some architectures
        std::int16_t __iters = sycl::ctz(__sorted_final) - sycl::ctz(__sorted);
        for (std::int16_t __i = 0; __i < __iters; ++__i)
        {
            const std::uint32_t __id_local = __id % __next_sorted;
            // Borders of the ranges to be merged
            const std::uint32_t __start1 = std::min(__id - __id_local, __end);
            const std::uint32_t __end1 = std::min(__start1 + __sorted, __end);
            const std::uint32_t __start2 = __end1;
            const std::uint32_t __end2 = std::min(__start2 + __sorted, __end);
            const std::uint32_t __n1 = __end1 - __start1;
            const std::uint32_t __n2 = __end2 - __start2;

            auto __in_ptr = __dpl_sycl::__get_accessor_ptr(__storage_acc) + __data_in_temp * __sorted_final;
            auto __out_ptr = __dpl_sycl::__get_accessor_ptr(__storage_acc) + (!__data_in_temp) * __sorted_final;
            auto __in_ptr1 = __in_ptr + __start1;
            auto __in_ptr2 = __in_ptr + __start2;

            const std::pair<std::uint32_t, std::uint32_t> __start = __find_start_point(
                __in_ptr1, std::uint32_t{0}, __n1, __in_ptr2, std::uint32_t{0}, __n2, __id_local, __comp);
            // TODO: copy the data into registers before the merge to halve the required amount of SLM
            __serial_merge(__in_ptr1, __in_ptr2, __out_ptr, __start.first, __start.second, __id, __data_per_workitem,
                           __n1, __n2, __comp);
            __dpl_sycl::__group_barrier(__item);

            __sorted = __next_sorted;
            __next_sorted *= 2;
            __data_in_temp = !__data_in_temp;
        }
        return __data_in_temp;
    }
};

template <typename _Range, typename _Compare>
struct __leaf_sorter
{
    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Range>;
    using _StorageAcc = __dpl_sycl::__local_accessor<_Tp>;
    // TODO: select a better sub-group sorter depending on sort stability,
    // a type (e.g. it can be trivially copied for shuffling within a sub-group)
    using _SubGroupSorter = __subgroup_bubble_sorter;
    using _GroupSorter = __group_merge_path_sorter;

    static std::uint32_t
    storage_size(std::uint16_t __future_data_per_workitem, std::uint32_t __future_workgroup_size)
    {
        return 2 * __future_data_per_workitem * __future_workgroup_size;
    }

    _StorageAcc
    create_storage_accessor(sycl::handler& __cgh) const
    {
        return _StorageAcc(storage_size(__data_per_workitem, __workgroup_size), __cgh);
    }

    __leaf_sorter(const _Range& __rng, _Compare __comp, std::uint16_t __data_per_workitem,
                  std::uint32_t __workgroup_size)
        : __rng(__rng), __comp(__comp), __n(__rng.size()), __data_per_workitem(__data_per_workitem),
          __workgroup_size(__workgroup_size), __process_size(__data_per_workitem * __workgroup_size),
          __sub_group_sorter(), __group_sorter()
    {
        assert((__process_size & (__process_size - 1)) == 0 && "Process size must be a power of 2");
    }

    void
    sort(const sycl::nd_item<1>& __item, const _StorageAcc& __storage_acc) const
    {
        sycl::sub_group __sg = __item.get_sub_group();
        const std::uint32_t __wg_id = __item.get_group_linear_id();
        const std::uint32_t __sg_id = __sg.get_group_linear_id();
        const std::uint32_t __sg_size = __sg.get_local_linear_range();
        const std::uint32_t __sg_local_id = __sg.get_local_linear_id();
        const std::uint32_t __sg_process_size = __sg_size * __data_per_workitem;
        const std::size_t __wg_start = __wg_id * __process_size;
        const std::uint32_t __sg_start = __sg_id * __sg_process_size;
        const std::size_t __wg_end = __wg_start + std::min<std::size_t>(__process_size, __n - __wg_start);
        const std::uint32_t __adjusted_process_size = __wg_end - __wg_start;

        // 1. Load
        // TODO: add a specialization for a case __global_value_id < __n condition is true for the whole work-group
        for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            const std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            const std::uint32_t __local_value_id = __sg_offset + __sg_local_id;
            const std::size_t __global_value_id = __wg_start + __local_value_id;
            if (__global_value_id < __n)
            {
                __storage_acc[__local_value_id] = std::move(__rng[__global_value_id]);
            }
        }
        sycl::group_barrier(__sg);

        // 2. Sort on sub-group level
        // TODO: move border selection inside the sub-group algorithm since it depends on a particular implementation
        // TODO: set a threshold for bubble sorter (likely 4 items)
        std::uint32_t __item_start = __sg_start + __sg_local_id * __data_per_workitem;
        std::uint32_t __item_end = __item_start + __data_per_workitem;
        __item_start = std::min(__item_start, __adjusted_process_size);
        __item_end = std::min(__item_end, __adjusted_process_size);
        __sub_group_sorter.sort(__storage_acc, __comp, __item_start, __item_end);
        __dpl_sycl::__group_barrier(__item);

        // 3. Sort on work-group level
        bool __data_in_temp =
            __group_sorter.sort(__item, __storage_acc, __comp, std::uint32_t{0}, __adjusted_process_size,
                                /*sorted per sub-group*/ __data_per_workitem, __data_per_workitem, __workgroup_size);
        // barrier is not needed here because of the barrier inside the sort method

        // 4. Store
        for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            const std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            const std::uint32_t __local_value_id = __sg_offset + __sg_local_id;
            const std::size_t __global_value_id = __wg_start + __local_value_id;
            if (__global_value_id < __n)
            {
                __rng[__global_value_id] = std::move(__storage_acc[__local_value_id + __data_in_temp * __process_size]);
            }
        }
    }

    _Range __rng;
    _Compare __comp;
    _Size __n;
    std::uint16_t __data_per_workitem;
    std::uint32_t __workgroup_size;
    std::uint32_t __process_size;
    _SubGroupSorter __sub_group_sorter;
    _GroupSorter __group_sorter;
};

template <typename _LeafSortName>
struct __merge_sort_leaf_submitter;

template <typename... _LeafSortName>
struct __merge_sort_leaf_submitter<__internal::__optional_kernel_name<_LeafSortName...>>
{
    template <typename _Range, typename _LeafSorter>
    sycl::event
    operator()(sycl::queue& __q, _Range& __rng, _LeafSorter& __leaf_sorter) const
    {
        return __q.submit([&__rng, &__leaf_sorter](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __storage_acc = __leaf_sorter.create_storage_accessor(__cgh);
            const std::uint32_t __wg_count =
                oneapi::dpl::__internal::__dpl_ceiling_div(__rng.size(), __leaf_sorter.__process_size);
            const sycl::nd_range<1> __nd_range(sycl::range<1>(__wg_count * __leaf_sorter.__workgroup_size),
                                               sycl::range<1>(__leaf_sorter.__workgroup_size));
            __cgh.parallel_for<_LeafSortName...>(
                __nd_range, [=](sycl::nd_item<1> __item) { __leaf_sorter.sort(__item, __storage_acc); });
        });
    }
};

template <typename _IndexT, typename _DiagonalsKernelName, typename _GlobalSortName1, typename _GlobalSortName2>
struct __merge_sort_global_submitter;

template <typename _IndexT, typename... _DiagonalsKernelName, typename... _GlobalSortName1,
          typename... _GlobalSortName2>
struct __merge_sort_global_submitter<_IndexT, __internal::__optional_kernel_name<_DiagonalsKernelName...>,
                                     __internal::__optional_kernel_name<_GlobalSortName1...>,
                                     __internal::__optional_kernel_name<_GlobalSortName2...>>
{
  private:
    using _merge_split_point_t = _split_point_t<_IndexT>;

    struct nd_range_params
    {
        std::size_t base_diag_count = 0;
        std::size_t steps_between_two_base_diags = 0;
        _IndexT chunk = 0;
        _IndexT steps = 0;
    };

    struct WorkDataArea
    {
        // How WorkDataArea is implemented :
        //
        //                              i_elem_local
        //                                |
        //                      offset    |    i_elem
        //                        |       |      |
        //                        V       V      V
        //                 +------+-------+------+-----+
        //                 |      |       |      /     |
        //                 |      |       |     /      |
        //                 |      |       |    /       |
        //                 |      |       |   /        |
        //                 |      |       |  /         |
        //                 |      |       | /          |
        //       offset -> +------+---n1--+       <----+---- whole data area : size == __n
        //                 |      |      /|            |
        //                 |      |   <-/-+------------+---- working data area : sizeof(rng1) <= __n_sorted, sizeof(rng2) <= __n_sorted
        //                 |      |    /  |            |
        //                 |     n2   /   |            |
        //                 |      |  /    |            |
        //                 |      | /     |            |
        //                 |      |/      |            |
        // i_elem_local -> +------+-------+            |
        //                 |     /                     |
        //                 |    /                      |
        //                 |   /                       |
        //                 |  /                        |
        //                 | /                         |
        //       i_elem -> +/                          |
        //                 |                           |
        //                 |                           |
        //                 |                           |
        //                 |                           |
        //                 |                           |
        //                 +---------------------------+

        _IndexT i_elem = 0;       // Global diagonal index
        _IndexT i_elem_local = 0; // Local diagonal index
        // Offset to the first element in the subrange (i.e. the first element of the first subrange for merge)
        _IndexT offset = 0;
        _IndexT n1 = 0; // Size of the first subrange
        _IndexT n2 = 0; // Size of the second subrange

        WorkDataArea(const std::size_t __n, const std::size_t __n_sorted, const std::size_t __linear_id,
                     const std::size_t __chunk)
        {
            // Calculate global diagonal index
            i_elem = __linear_id * __chunk;

            // Calculate local diagonal index
            i_elem_local = i_elem % (__n_sorted * 2);

            // Calculate offset to the first element in the subrange (i.e. the first element of the first subrange for merge)
            offset = std::min<_IndexT>(i_elem - i_elem_local, __n);

            // Calculate size of the first and the second subranges
            n1 = std::min<_IndexT>(offset + __n_sorted, __n) - offset;
            n2 = std::min<_IndexT>(offset + __n_sorted + n1, __n) - (offset + n1);
        }

        inline bool
        is_i_elem_local_inside_merge_matrix() const
        {
            return i_elem_local < n1 + n2;
        }
    };

    template <typename Rng>
    struct DropViews
    {
        using __drop_view_simple_t = oneapi::dpl::__ranges::drop_view_simple<Rng, _IndexT>;

        __drop_view_simple_t rng1;
        __drop_view_simple_t rng2;

        DropViews(Rng& __rng, const WorkDataArea& __data_area)
            : rng1(__rng, __data_area.offset), rng2(__rng, __data_area.offset + __data_area.n1)
        {
        }
    };
    // Clang 17 and earlier, as well as other compilers based on them, such as DPC++ 2023.2
    // are prone to https://github.com/llvm/llvm-project/issues/46200,
    // which prevents automatic template argument deduction of a nested class.
    template <typename Rng>
    DropViews(Rng&, const WorkDataArea&) -> DropViews<Rng>;

    std::size_t
    get_max_base_diags_count(const sycl::queue& __q, const _IndexT __chunk, std::size_t __n) const
    {
        const std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__q);
        return oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk * __max_wg_size);
    }

    // Calculate nd-range params
    nd_range_params
    eval_nd_range_params(const sycl::queue& __q, const std::size_t __rng_size, const _IndexT __n_sorted) const
    {
        const bool __is_cpu = __q.get_device().is_cpu();
        // The chunk size must not exceed two sorted sub-sequences to be merged,
        // ensuring that at least one work-item processes them.
        const _IndexT __chunk = std::min<_IndexT>(__is_cpu ? 32 : 4, __n_sorted * 2);
        const _IndexT __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__rng_size, __chunk);

        _IndexT __base_diag_count = get_max_base_diags_count(__q, __chunk, __n_sorted);
        _IndexT __steps_between_two_base_diags = oneapi::dpl::__internal::__dpl_ceiling_div(__steps, __base_diag_count);

        return {__base_diag_count, __steps_between_two_base_diags, __chunk, __steps};
    }

    template <typename DropViews, typename _Compare>
    inline static _merge_split_point_t
    __find_start_point(const WorkDataArea& __data_area, const DropViews& __views, _Compare __comp)
    {
        return oneapi::dpl::__par_backend_hetero::__find_start_point(__views.rng1, _IndexT{0}, __data_area.n1,
                                                                     __views.rng2, _IndexT{0}, __data_area.n2,
                                                                     __data_area.i_elem_local, __comp);
    }

    template <typename DropViews, typename _Rng, typename _Compare>
    inline static void
    __serial_merge(const nd_range_params& __nd_range_params, const WorkDataArea& __data_area, const DropViews& __views,
                   _Rng& __rng, const _merge_split_point_t& __sp, _Compare __comp)
    {
        oneapi::dpl::__par_backend_hetero::__serial_merge(
            __views.rng1, __views.rng2, __rng /* rng3 */, __sp.first /* start1 */, __sp.second /* start2 */,
            __data_area.i_elem /* start3 */, __nd_range_params.chunk, __data_area.n1, __data_area.n2, __comp);
    }

    // Calculation of split points on each base diagonal
    template <typename _Range, typename _TempBuf, typename _Compare, typename _Storage>
    sycl::event
    eval_split_points_for_groups(const sycl::event& __event_chain, const _IndexT __n_sorted, const bool __data_in_temp,
                                 sycl::queue& __q, const _Range& __rng, _TempBuf& __temp_buf, _Compare __comp,
                                 const nd_range_params& __nd_range_params,
                                 _Storage& __base_diagonals_sp_global_storage) const
    {
        const _IndexT __n = __rng.size();

        return __q.submit([&__event_chain, __n_sorted, __data_in_temp, &__rng, &__temp_buf, __comp, __nd_range_params,
                           &__base_diagonals_sp_global_storage, __n](sycl::handler& __cgh) {
            __cgh.depends_on(__event_chain);

            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __base_diagonals_sp_global_acc =
                __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::write>(
                    __cgh, __dpl_sycl::__no_init{});

            sycl::accessor __dst(__temp_buf, __cgh, sycl::read_write, sycl::no_init);

            const std::size_t __chunk = __nd_range_params.chunk * __nd_range_params.steps_between_two_base_diags;

            __cgh.parallel_for<_DiagonalsKernelName...>(
                sycl::range</*dim=*/1>(__nd_range_params.base_diag_count), [=](sycl::item</*dim=*/1> __item_id) {
                    const std::size_t __linear_id = __item_id.get_linear_id();

                    auto __base_diagonals_sp_global_ptr =
                        _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    const WorkDataArea __data_area(__n, __n_sorted, __linear_id, __chunk);

                    const auto __sp =
                        __data_area.is_i_elem_local_inside_merge_matrix()
                            ? (__data_in_temp ? __find_start_point(__data_area, DropViews(__dst, __data_area), __comp)
                                              : __find_start_point(__data_area, DropViews(__rng, __data_area), __comp))
                            : _merge_split_point_t{__data_area.n1, __data_area.n2};
                    __base_diagonals_sp_global_ptr[__linear_id] = __sp;
                });
        });
    }

    template <typename _BaseDiagonalsSPStorage>
    inline static _merge_split_point_t
    __get_right_sp(_BaseDiagonalsSPStorage __base_diagonals_sp_global_ptr, const std::size_t __diagonal_idx,
                   const WorkDataArea& __data_area)
    {
        _merge_split_point_t __result = __base_diagonals_sp_global_ptr[__diagonal_idx];
        __result =
            __result.first + __result.second > 0 ? __result : _merge_split_point_t{__data_area.n1, __data_area.n2};

        return __result;
    }

    template <typename DropViews, typename _Compare, typename _BaseDiagonalsSPStorage>
    inline static _merge_split_point_t
    __lookup_sp(const std::size_t __linear_id_in_steps_range, const nd_range_params& __nd_range_params,
                const WorkDataArea& __data_area, const DropViews& __views, _Compare __comp,
                _BaseDiagonalsSPStorage __base_diagonals_sp_global_ptr)
    {
        //   |                  subrange 0                |                subrange 1                  |                subrange 2                  |                subrange 3                  | subrange 4
        //   |        contains (2 * __n_sorted values)    |        contains (2 * __n_sorted values)    |        contains (2 * __n_sorted values)    |        contains (2 * __n_sorted values)    | contains the rest of data...  < Data parts
        //   |----/----/----/----/----/----/----/----/----|----/----/----/----/----/----/----/----/----|----/----/----/----/----/----/----/----/----|----/----/----/----/----/----/----/----/----|----/---                       < Steps
        //   ^              ^              ^              ^              ^              ^              ^              ^         ^    ^              ^              ^              ^              ^
        //   |              |              |              |              |              |              |              |         |    |              |              |              |              |
        // bd00           bd01           bd02           bd10           bd11           bd12           bd20           bd21        |  bd22           bd30           bd31           bd32           bd40                              < Base diagonals
        //                  ^              ^              ^              ^              ^              ^              ^         |    ^              ^              ^              ^              ^
        //   0              1              2              3              4              5              6         |    7              8              9              10             11             12                   xIdx       < Indexes in the base diagonal's SP storage
        //   0    1    2    3    4    5    6    7    8    9    10   11   12   13   14  15   16    17   18   19   20   20   21   |    23   24   25   26   27   28   29   30   31   32   33   34   35    36                        < Linear IDs: __linear_id_in_steps_range
        //   ^                                            ^                                            ^              |         |    |              ^                                            ^                    ^
        //   (0,0)                                        (0,0)                                        (0,0)        __sp_left   |  __sp_right       (0,0)                                        (0,0)                (0,0)      < Every first base diagonal of sub-task is (0,0)
        //                                                                                                                      |                                                                                     final additional split-point
        //                                                                                                          __linear_id_in_steps_range

        const std::size_t __diagonal_idx = __linear_id_in_steps_range / __nd_range_params.steps_between_two_base_diags;

        if (__linear_id_in_steps_range % __nd_range_params.steps_between_two_base_diags != 0)
        {
            // We are between two base diagonals (__sp_left, __sp_right)
            const _merge_split_point_t __sp_left = __base_diagonals_sp_global_ptr[__diagonal_idx];
            const _merge_split_point_t __sp_right =
                __get_right_sp(__base_diagonals_sp_global_ptr, __diagonal_idx + 1, __data_area);

            return oneapi::dpl::__par_backend_hetero::__find_start_point(
                __views.rng1, __sp_left.first, __sp_right.first, __views.rng2, __sp_left.second, __sp_right.second,
                __data_area.i_elem_local, __comp);
        }

        // We are on base diagonal so just simple return split-point from them
        return __base_diagonals_sp_global_ptr[__diagonal_idx];
    }

    // Process parallel merge
    template <typename _Range, typename _TempBuf, typename _Compare>
    sycl::event
    run_parallel_merge(const sycl::event& __event_chain, const _IndexT __n_sorted, const bool __data_in_temp,
                       sycl::queue& __q, _Range& __rng, _TempBuf& __temp_buf, _Compare __comp,
                       const nd_range_params& __nd_range_params) const
    {
        const _IndexT __n = __rng.size();

        return __q.submit([&__event_chain, __n_sorted, __data_in_temp, &__rng, &__temp_buf, __comp, __nd_range_params,
                           __n](sycl::handler& __cgh) {
            __cgh.depends_on(__event_chain);

            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            sycl::accessor __dst(__temp_buf, __cgh, sycl::read_write, sycl::no_init);

            __cgh.parallel_for<_GlobalSortName1...>(
                sycl::range</*dim=*/1>(__nd_range_params.steps), [=](sycl::item</*dim=*/1> __item_id) {
                    const std::size_t __linear_id = __item_id.get_linear_id();

                    const WorkDataArea __data_area(__n, __n_sorted, __linear_id, __nd_range_params.chunk);

                    if (__data_area.is_i_elem_local_inside_merge_matrix())
                    {
                        if (__data_in_temp)
                        {
                            DropViews __views(__dst, __data_area);
                            __serial_merge(__nd_range_params, __data_area, __views, __rng,
                                           __find_start_point(__data_area, __views, __comp), __comp);
                        }
                        else
                        {
                            DropViews __views(__rng, __data_area);
                            __serial_merge(__nd_range_params, __data_area, __views, __dst,
                                           __find_start_point(__data_area, __views, __comp), __comp);
                        }
                    }
                });
        });
    }

    // Process parallel merge with usage of split-points on base diagonals
    template <typename _Range, typename _TempBuf, typename _Compare, typename _Storage>
    sycl::event
    run_parallel_merge_from_diagonals(const sycl::event& __event_chain, const _IndexT __n_sorted,
                                      const bool __data_in_temp, sycl::queue& __q, _Range& __rng, _TempBuf& __temp_buf,
                                      _Compare __comp, const nd_range_params& __nd_range_params,
                                      _Storage& __base_diagonals_sp_global_storage) const
    {
        const _IndexT __n = __rng.size();

        return __q.submit([&__event_chain, __n_sorted, __data_in_temp, &__rng, &__temp_buf, __comp, __nd_range_params,
                           &__base_diagonals_sp_global_storage, __n](sycl::handler& __cgh) {
            __cgh.depends_on(__event_chain);

            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            sycl::accessor __dst(__temp_buf, __cgh, sycl::read_write, sycl::no_init);

            auto __base_diagonals_sp_global_acc =
                __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read>(__cgh);

            __cgh.parallel_for<_GlobalSortName2...>(
                sycl::range</*dim=*/1>(__nd_range_params.steps), [=](sycl::item</*dim=*/1> __item_id) {
                    const std::size_t __linear_id = __item_id.get_linear_id();

                    auto __base_diagonals_sp_global_ptr =
                        _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    const WorkDataArea __data_area(__n, __n_sorted, __linear_id, __nd_range_params.chunk);

                    if (__data_area.is_i_elem_local_inside_merge_matrix())
                    {
                        if (__data_in_temp)
                        {
                            DropViews __views(__dst, __data_area);
                            __serial_merge(__nd_range_params, __data_area, __views, __rng,
                                           __lookup_sp(__linear_id, __nd_range_params, __data_area, __views, __comp,
                                                       __base_diagonals_sp_global_ptr),
                                           __comp);
                        }
                        else
                        {
                            DropViews __views(__rng, __data_area);
                            __serial_merge(__nd_range_params, __data_area, __views, __dst,
                                           __lookup_sp(__linear_id, __nd_range_params, __data_area, __views, __comp,
                                                       __base_diagonals_sp_global_ptr),
                                           __comp);
                        }
                    }
                });
        });
    }

  public:
    template <typename _Range, typename _Compare, typename _TempBuf, typename _LeafSizeT>
    std::tuple<sycl::event, bool, std::shared_ptr<__result_and_scratch_storage_base>>
    operator()(sycl::queue& __q, _Range& __rng, _Compare __comp, _LeafSizeT __leaf_size, _TempBuf& __temp_buf,
               sycl::event __event_chain) const
    {
        // 1 final base diagonal for save final sp(0,0)
        constexpr std::size_t __1_final_base_diag = 1;

        const _IndexT __n = __rng.size();
        _IndexT __n_sorted = __leaf_size;

        bool __data_in_temp = false;

        using __value_type = oneapi::dpl::__internal::__value_t<_Range>;

        // Calculate nd-range params
        const nd_range_params __nd_range_params = eval_nd_range_params(__q, __n, __n_sorted);

        using __base_diagonals_sp_storage_t = __result_and_scratch_storage<_merge_split_point_t, /* _NResults */ 0>;

        const std::size_t __n_power2 = oneapi::dpl::__internal::__dpl_bit_ceil(__n);
        // ctz precisely calculates log2 of an integral value which is a power of 2, while
        // std::log2 may be prone to rounding errors on some architectures
        const std::int64_t __n_iter = sycl::ctz(__n_power2) - sycl::ctz(__leaf_size);

        // Storage to save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
        __base_diagonals_sp_storage_t* __p_base_diagonals_sp_global_storage = nullptr;

        // shared_ptr instance to return it in __future and extend the lifetime of the storage.
        std::shared_ptr<__result_and_scratch_storage_base> __p_result_and_scratch_storage_base;

        // Max amount of base diagonals
        const std::size_t __max_base_diags_count =
            get_max_base_diags_count(__q, __nd_range_params.chunk, __n) + __1_final_base_diag;

        for (std::int64_t __i = 0; __i < __n_iter; ++__i)
        {
            // TODO required to re-check threshold data size
            if (2 * __n_sorted < __get_starting_size_limit_for_large_submitter<__value_type>())
            {
                // Process parallel merge
                __event_chain = run_parallel_merge(__event_chain, __n_sorted, __data_in_temp, __q, __rng, __temp_buf,
                                                   __comp, __nd_range_params);
            }
            else
            {
                if (nullptr == __p_base_diagonals_sp_global_storage)
                {
                    // Create storage to save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
                    __p_base_diagonals_sp_global_storage =
                        new __base_diagonals_sp_storage_t(__q, __max_base_diags_count);

                    // Save the raw pointer into a shared_ptr to return it in __future and extend the lifetime of the storage.
                    __p_result_and_scratch_storage_base.reset(
                        static_cast<__result_and_scratch_storage_base*>(__p_base_diagonals_sp_global_storage));
                }

                nd_range_params __nd_range_params_this =
                    eval_nd_range_params(__q, std::size_t(2 * __n_sorted), __n_sorted);

                // Check that each base diagonal started from beginning of merge matrix
                assert(0 == (2 * __n_sorted) %
                                (__nd_range_params_this.steps_between_two_base_diags * __nd_range_params_this.chunk));

                const auto __portions = oneapi::dpl::__internal::__dpl_ceiling_div(__n, 2 * __n_sorted);
                __nd_range_params_this.base_diag_count =
                    __nd_range_params_this.base_diag_count * __portions + __1_final_base_diag;
                __nd_range_params_this.steps *= __portions;
                assert(__nd_range_params_this.base_diag_count <= __max_base_diags_count);

                // Calculation of split-points on each base diagonal
                __event_chain =
                    eval_split_points_for_groups(__event_chain, __n_sorted, __data_in_temp, __q, __rng, __temp_buf,
                                                 __comp, __nd_range_params_this, *__p_base_diagonals_sp_global_storage);

                // Process parallel merge with usage of split-points on base diagonals
                __event_chain = run_parallel_merge_from_diagonals(__event_chain, __n_sorted, __data_in_temp, __q, __rng,
                                                                  __temp_buf, __comp, __nd_range_params_this,
                                                                  *__p_base_diagonals_sp_global_storage);
            }

            __n_sorted *= 2;
            __data_in_temp = !__data_in_temp;
        }

        return {std::move(__event_chain), __data_in_temp, std::move(__p_result_and_scratch_storage_base)};
    }
};

template <typename _CopyBackName>
struct __merge_sort_copy_back_submitter;

template <typename... _CopyBackName>
struct __merge_sort_copy_back_submitter<__internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _Range, typename _TempBuf>
    sycl::event
    operator()(sycl::queue& __q, _Range& __rng, _TempBuf& __temp_buf, sycl::event __event_chain) const
    {
        return __q.submit([&__rng, &__temp_buf, &__event_chain](sycl::handler& __cgh) {
            __cgh.depends_on(__event_chain);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __temp_acc = __temp_buf.template get_access<access_mode::read>(__cgh);
            // We cannot use __cgh.copy here because of zip_iterator usage
            __cgh.parallel_for<_CopyBackName...>(sycl::range</*dim=*/1>(__rng.size()),
                                                 [=](sycl::item</*dim=*/1> __item_id) {
                                                     const std::size_t __idx = __item_id.get_linear_id();
                                                     __rng[__idx] = __temp_acc[__idx];
                                                 });
        });
    }
};

template <typename... _Name>
class __sort_leaf_kernel;

template <typename... _Name>
class __diagonals_kernel_name_for_merge_sort;

template <typename... _Name>
class __sort_global_kernel1;

template <typename... _Name>
class __sort_global_kernel2;

template <typename... _Name>
class __sort_copy_back_kernel;

template <typename _CustomName, typename _IndexT, typename _Range, typename _Compare, typename _LeafSorter>
__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
__merge_sort(sycl::queue& __q, _Range&& __rng, _Compare __comp, _LeafSorter& __leaf_sorter)
{
    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;

    using _LeafSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel<_CustomName>>;
    using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __diagonals_kernel_name_for_merge_sort<_CustomName, _IndexT>>;
    using _GlobalSortKernel1 = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __sort_global_kernel1<_CustomName, _IndexT>>;
    using _GlobalSortKernel2 = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __sort_global_kernel2<_CustomName, _IndexT>>;
    using _CopyBackKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName>>;

    assert(__rng.size() > 1);
    assert((__leaf_sorter.__process_size & (__leaf_sorter.__process_size - 1)) == 0 &&
           "Leaf size must be a power of 2");

    // 1. Perform sorting of the leaves of the merge sort tree
    sycl::event __event_leaf_sort = __merge_sort_leaf_submitter<_LeafSortKernel>()(__q, __rng, __leaf_sorter);

    // 2. Merge sorting
    oneapi::dpl::__par_backend_hetero::__buffer<_Tp> __temp(__rng.size());
    auto __temp_buf = __temp.get_buffer();
    auto [__event_sort, __data_in_temp, __temp_sp_storages] =
        __merge_sort_global_submitter<_IndexT, _DiagonalsKernelName, _GlobalSortKernel1, _GlobalSortKernel2>()(
            __q, __rng, __comp, __leaf_sorter.__process_size, __temp_buf, __event_leaf_sort);

    // 3. If the data remained in the temporary buffer then copy it back
    if (__data_in_temp)
    {
        __event_sort = __merge_sort_copy_back_submitter<_CopyBackKernel>()(__q, __rng, __temp_buf, __event_sort);
    }

    return __future{std::move(__event_sort), std::move(__temp_sp_storages)};
}

template <typename _CustomName, typename _IndexT, typename _Range, typename _Compare>
__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
__submit_selecting_leaf(sycl::queue& __q, _Range&& __rng, _Compare __comp)
{
    using _Leaf = __leaf_sorter<std::decay_t<_Range>, _Compare>;
    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;

    const std::size_t __n = __rng.size();
    sycl::device __device = __q.get_device();

    const std::size_t __max_wg_size = __device.template get_info<sycl::info::device::max_work_group_size>();

    const bool __is_cpu = __device.is_cpu();
    std::uint32_t __max_sg_size{};
    if (__is_cpu)
    {
        const auto __sg_sizes = __device.template get_info<sycl::info::device::sub_group_sizes>();
        __max_sg_size = *std::max_element(__sg_sizes.begin(), __sg_sizes.end());
    }
    // Assume CPUs handle one sub-group (SIMD) per CU;
    // Assume GPUs handle multiple sub-groups per CU,
    // while the maximum work-group size takes hardware multithreading (occupancy) into account
    const std::size_t __max_hw_wg_size = __is_cpu ? __max_sg_size : __max_wg_size;
    const auto __max_cu = __device.template get_info<sycl::info::device::max_compute_units>();
    // TODO: adjust the saturation point for Intel GPUs:
    // CU number is incorrect for Intel GPUs since it returns the number of VE instead of XC,
    // and max work-group size utilizes only a half of the XC resources for Data Center GPUs
    const std::uint32_t __saturation_point = __max_cu * __max_hw_wg_size;
    const std::uint32_t __desired_data_per_workitem = __n / __saturation_point;

    // 8 is the maximum reasonable value for bubble sub-group sorter due to algorithm complexity
    // 2 is the smallest reasonable value for merge-path group sorter since it loads 2 values at least
    // TODO: reconsider the values if other algorithms are used
    const std::uint16_t __data_per_workitem =
        __desired_data_per_workitem <= 2
            ? 2
            : std::min<std::uint32_t>(oneapi::dpl::__internal::__dpl_bit_floor(__desired_data_per_workitem), 8);

    // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
    // TODO: investigate if the adjustment can be less conservative
    const std::size_t __max_slm_items =
        __device.template get_info<sycl::info::device::local_mem_size>() / (sizeof(_Tp) * 2);

    const std::size_t __max_slm_wg_size = __max_slm_items / _Leaf::storage_size(__data_per_workitem, 1);
    // __n is taken as is because of the bit floor and processing at least 2 items per work-item
    // hence the processed size always fits a single work-group if __n is chosen
    std::size_t __wg_size = std::min<std::size_t>({__max_hw_wg_size, __max_slm_wg_size, __n});
    __wg_size = oneapi::dpl::__internal::__dpl_bit_floor(__wg_size);

    _Leaf __leaf(__rng, __comp, __data_per_workitem, __wg_size);
    return __merge_sort<_CustomName, _IndexT>(__q, std::forward<_Range>(__rng), __comp, __leaf);
};

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
__parallel_sort_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __rng,
                     _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    if (__rng.size() <= std::numeric_limits<std::uint32_t>::max())
    {
        return __submit_selecting_leaf<_CustomName, std::uint32_t>(__q_local, std::forward<_Range>(__rng), __comp);
    }
    else
    {
        return __submit_selecting_leaf<_CustomName, std::uint64_t>(__q_local, std::forward<_Range>(__rng), __comp);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H
