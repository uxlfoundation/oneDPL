// -*- C++ -*-
//===-- parallel_backend_sycl_merge.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H

#include <limits>    // std::numeric_limits
#include <cassert>   // assert
#include <cstdint>   // std::uint8_t, ...
#include <utility>   // std::make_pair, std::forward, std::declval
#include <algorithm> // std::min, std::lower_bound
#include <type_traits> // std::void_t, std::true_type, std::false_type

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "../../functional_impl.h" // for oneapi::dpl::identity

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{
template <typename _Index>
using _split_point_t = std::pair<_Index, _Index>;

//Searching for an intersection of a merge matrix (n1, n2) diagonal with the Merge Path to define sub-ranges
//to serial merge. For example, a merge matrix for [0,1,1,2,3] and [0,0,2,3] is shown below:
//     0   1  1  2   3
//    ------------------
//   |--->
// 0 | 0 | 1  1  1   1
//   |   |
// 0 | 0 | 1  1  1   1
//   |   ---------->
// 2 | 0   0  0  0 | 1
//   |             ---->
// 3 | 0   0  0  0   0 |
template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare, typename _Proj1 = oneapi::dpl::identity,
          typename _Proj2 = oneapi::dpl::identity>
_split_point_t<_Index>
__find_start_point(const _Rng1& __rng1, const _Index __rng1_from, _Index __rng1_to, const _Rng2& __rng2,
                   const _Index __rng2_from, _Index __rng2_to, const _Index __i_elem, _Compare __comp,
                   _Proj1 __proj1 = {}, _Proj2 __proj2 = {})
{
    // ----------------------- EXAMPLE ------------------------
    // Let's consider the following input data:
    //    rng1.size() = 10
    //    rng2.size() = 6
    //    i_diag = 9
    // Let's define the following ranges for processing:
    //    rng1: [3, ..., 9) -> __rng1_from = 3, __rng1_to = 9
    //    rng2: [1, ..., 4) -> __rng2_from = 1, __rng2_to = 4
    //
    // The goal: required to process only X' items of the merge matrix
    //           as intersection of rng1[3, ..., 9) and rng2[1, ..., 4)
    //
    // --------------------------------------------------------
    //
    //         __diag_it_begin(rng1)            __diag_it_end(rng1)
    //      (init state) (dest state)          (init state, dest state)
    //            |          |                       |
    //            V          V                       V
    //                       +   +   +   +   +   +
    //    \ rng1  0  1   2   3   4   5   6   7   8   9
    //   rng2   +--------------------------------------+
    //    0     |                    ^   ^   ^   X     |     <--- __diag_it_end(rng2) (init state)
    // +  1     | <----------------- +   +   X'2 ^     |     <--- __diag_it_end(rng2) (dest state)
    // +  2     | <----------------- +   X'1     |     |
    // +  3     | <----------------- X'0         |     |     <--- __diag_it_begin(rng2) (dest state)
    //    4     |                X   ^           |     |
    //    5     |            X       |           |     |     <--- __diag_it_begin(rng2) (init state)
    //          +-------AX-----------+-----------+-----+
    //              AX               |           |
    //           AX                  |           |
    //              Run lower_bound:[from = 5,   to = 8)
    //
    //  AX - absent items in rng2
    //
    //  We have three points on diagonal for call comparison:
    //      X'0 : call __comp(rng1[5], rng2[3])             // 5 + 3 == 9 - 1 == 8
    //      X'1 : call __comp(rng1[6], rng2[2])             // 6 + 2 == 9 - 1 == 8
    //      X'3 : call __comp(rng1[7], rng2[1])             // 7 + 1 == 9 - 1 == 8
    //   - where for every comparing pairs idx(rng1) + idx(rng2) == i_diag - 1

    using _IndexSigned = std::make_signed_t<_Index>;

    ////////////////////////////////////////////////////////////////////////////////////
    // Taking into account the specified constraints of the range of processed data
    const _IndexSigned __index_sum = __i_elem - 1;

    _IndexSigned idx1_from = __rng1_from;
    _IndexSigned idx1_to = __rng1_to;

    _IndexSigned idx2_from = __index_sum - (__rng1_to - 1);
    _IndexSigned idx2_to = __index_sum - __rng1_from + 1;

    const _IndexSigned idx2_from_diff =
        idx2_from < (_IndexSigned)__rng2_from ? (_IndexSigned)__rng2_from - idx2_from : 0;
    const _IndexSigned idx2_to_diff = idx2_to > (_IndexSigned)__rng2_to ? idx2_to - (_IndexSigned)__rng2_to : 0;

    idx1_to -= idx2_from_diff;
    idx1_from += idx2_to_diff;

    idx2_from = __index_sum - (idx1_to - 1);
    idx2_to = __index_sum - idx1_from + 1;

    ////////////////////////////////////////////////////////////////////////////////////
    // Run search of split point on diagonal

    using __it_t = oneapi::dpl::counting_iterator<_Index>;

    __it_t __diag_it_begin(idx1_from);
    __it_t __diag_it_end(idx1_to);

    const __it_t __res = std::lower_bound(
        __diag_it_begin, __diag_it_end, false,
        [&__rng1, &__rng2, __index_sum, __comp, __proj1, __proj2](_Index __idx, const bool __value) mutable {
            return __value == std::invoke(__comp, std::invoke(__proj2, __rng2[__index_sum - __idx]),
                                          std::invoke(__proj1, __rng1[__idx]));
        });

    return _split_point_t<_Index>{*__res, __index_sum - *__res + 1};
}

template <typename _Rng1DataType, typename _Rng2DataType, typename = void>
struct __can_use_ternary_op : std::false_type
{
};

template <typename _Rng1DataType, typename _Rng2DataType>
struct __can_use_ternary_op<_Rng1DataType, _Rng2DataType,
                            std::void_t<decltype(true ? std::declval<_Rng1DataType>() : std::declval<_Rng2DataType>())>>
    : std::true_type
{
};

template <typename _Rng1DataType, typename _Rng2DataType>
constexpr static bool __can_use_ternary_op_v = __can_use_ternary_op<_Rng1DataType, _Rng2DataType>::value;

// Do serial merge of the data from rng1 (starting from start1) and rng2 (starting from start2) and writing
// to rng3 (starting from start3) in 'chunk' steps, but do not exceed the total size of the sequences (n1 and n2)
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index, typename _Compare,
          typename _Proj1 = oneapi::dpl::identity, typename _Proj2 = oneapi::dpl::identity>
std::pair<_Index, _Index>
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3, const _Index __start1, const _Index __start2,
               const _Index __start3, const _Index __chunk, const _Index __n1, const _Index __n2, _Compare __comp,
               _Proj1 __proj1 = {}, _Proj2 __proj2 = {}, const _Index __n3 = 0)
{
    const _Index __rng1_size = std::min<_Index>(__n1 > __start1 ? __n1 - __start1 : _Index{0}, __chunk);
    const _Index __rng2_size = std::min<_Index>(__n2 > __start2 ? __n2 - __start2 : _Index{0}, __chunk);
    const _Index __rng3_size = std::min<_Index>(__rng1_size + __rng2_size, __chunk);

    const _Index __rng1_idx_end = __start1 + __rng1_size;
    const _Index __rng2_idx_end = __start2 + __rng2_size;
    const _Index __rng3_idx_end = __n3 > 0 ? std::min<_Index>(__n3, __start3 + __rng3_size) : __start3 + __rng3_size;

    _Index __rng1_idx = __start1;
    _Index __rng2_idx = __start2;

    bool __rng1_idx_less_n1 = false;
    bool __rng2_idx_less_n2 = false;

    for (_Index __rng3_idx = __start3; __rng3_idx < __rng3_idx_end; ++__rng3_idx)
    {
        __rng1_idx_less_n1 = __rng1_idx < __rng1_idx_end;
        __rng2_idx_less_n2 = __rng2_idx < __rng2_idx_end;

        // One of __rng1_idx_less_n1 and __rng2_idx_less_n2 should be true here
        // because 1) we should fill output data with elements from one of the input ranges
        // 2) we calculate __rng3_idx_end as std::min<_Index>(__rng1_size + __rng2_size, __chunk).
        if constexpr (__can_use_ternary_op_v<decltype(__rng1[__rng1_idx]), decltype(__rng2[__rng2_idx])>)
        {
            // This implementation is required for performance optimization
            __rng3[__rng3_idx] = (!__rng1_idx_less_n1 || (__rng1_idx_less_n1 && __rng2_idx_less_n2 &&
                                                          std::invoke(__comp, std::invoke(__proj2, __rng2[__rng2_idx]),
                                                                      std::invoke(__proj1, __rng1[__rng1_idx]))))
                                     ? __rng2[__rng2_idx++]
                                     : __rng1[__rng1_idx++];
        }
        else
        {
            // TODO required to understand why the usual if-else is slower then ternary operator
            if (!__rng1_idx_less_n1 || (__rng1_idx_less_n1 && __rng2_idx_less_n2 &&
                                        std::invoke(__comp, std::invoke(__proj2, __rng2[__rng2_idx]),
                                                    std::invoke(__proj1, __rng1[__rng1_idx]))))
                __rng3[__rng3_idx] = __rng2[__rng2_idx++];
            else
                __rng3[__rng3_idx] = __rng1[__rng1_idx++];
        }
    }
    return {__rng1_idx, __rng2_idx};
}

template <typename _IndexT>
using _split_points_device_storage_t = __device_storage<_split_point_t<_IndexT>>;

using _split_points_device_storage32_t = _split_points_device_storage_t<std::uint32_t>;
using _split_points_device_storage64_t = _split_points_device_storage_t<std::uint64_t>;

// Item 0 : event,
// Item 1 : split points storage for merge operations with _IdType = std::uint32_t
// Item 2 : split points storage for merge operations with _IdType = std::uint64_t
// Item 3 : optional result storage for merge operations (only if _OutSizeLimit is true)
template <typename _OutSizeLimit, typename _Range1, typename _Range2>
using __parallel_merge_return_data_t = std::conditional_t<
    _OutSizeLimit{},
    std::tuple<sycl::event,
               _split_points_device_storage32_t, _split_points_device_storage64_t,
               __result_storage<oneapi::dpl::__internal::__difference_tuple_t<_Range1, _Range2>>>,
    std::tuple<sycl::event,
               _split_points_device_storage32_t, _split_points_device_storage64_t>>;

template <typename _OutSizeLimit, typename _Range1, typename _Range2, typename _IdType>
__parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>
__create_parallel_merge_return_data(sycl::queue& __q, std::size_t __split_points_count)
{
    static_assert(std::is_same_v<_IdType, std::uint32_t> || std::is_same_v<_IdType, std::uint64_t>,
                  "The _IdType must be either std::uint32_t or std::uint64_t");

    // Optional create result storage for merge operations
    auto __create_result_storage = [&]() {
        if constexpr (_OutSizeLimit{})
            return __result_storage<oneapi::dpl::__internal::__difference_tuple_t<_Range1, _Range2>>(__q, 1);
    };

    // Create split points storage for merge operations with _IdType = std::uint32_t
    auto __create_sp_storage_32 = [&]() {
        if constexpr (std::is_same_v<_IdType, std::uint32_t>)
            return __split_points_count > 0 ? _split_points_device_storage32_t(__q, __split_points_count)
                                            : _split_points_device_storage32_t();
        else
            return _split_points_device_storage32_t();
    };

    // Create split points storage for merge operations with _IdType = std::uint64_t
    auto __create_sp_storage_64 = [&]() {
        if constexpr (std::is_same_v<_IdType, std::uint64_t>)
            return __split_points_count > 0 ? _split_points_device_storage64_t(__q, __split_points_count)
                                            : _split_points_device_storage64_t();
        else
            return _split_points_device_storage64_t();
    };

    if constexpr (_OutSizeLimit{})
        return {sycl::event(), __create_sp_storage_32(), __create_sp_storage_64(), __create_result_storage()};
    else
        return {sycl::event(), __create_sp_storage_32(), __create_sp_storage_64()};
}

// Sentinel type used as a stand-in for the stop-position accessor when _OutSizeLimit=false.
struct __no_parallel_merge_stop_pos_acc_tag
{
};

// Get the accessor to the result storage for merge operations if it is created, otherwise return __no_stop_pos_acc_tag
template <typename _OutSizeLimit, typename _Range1, typename _Range2, typename _ModeTagT>
auto
__get_parallel_merge_stop_pos_accessor_opt(_ModeTagT __mode, sycl::handler& __cgh,
                                           __parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>& __data,
                                           const sycl::property_list& __prop_list = {})
{
    if constexpr (_OutSizeLimit{})
        return __get_accessor(__mode, std::get<3>(__data), __cgh, __prop_list);
    else
        return __no_parallel_merge_stop_pos_acc_tag{};
}

template <typename _OutSizeLimit, typename _IdType, typename _Range1, typename _Range2>
auto&
__get_parallel_merge_sp_storage(__parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>& __data)
{
    static_assert(std::is_same_v<_IdType, std::uint32_t> || std::is_same_v<_IdType, std::uint64_t>,
                  "The _IdType must be either std::uint32_t or std::uint64_t");

    if constexpr (std::is_same_v<_IdType, std::uint32_t>)
        return std::get<1>(__data);
    else
        return std::get<2>(__data);
}

// Please see the comment for __parallel_for_small_submitter for optional kernel name explanation
template <typename _OutSizeLimit, typename _IdType, typename _Name>
struct __parallel_merge_submitter;

template <typename _OutSizeLimit, typename _IdType, typename... _Name>
struct __parallel_merge_submitter<_OutSizeLimit, _IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range1, typename _Range2, typename _Range3, typename _Compare, typename _Proj1, typename _Proj2>
    __parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>
    operator()(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp, _Proj1 __proj1,
               _Proj2 __proj2) const
    {
        const _IdType __n1 = oneapi::dpl::__ranges::__size(__rng1);
        const _IdType __n2 = oneapi::dpl::__ranges::__size(__rng2);
        const _IdType __n = std::min<_IdType>(__n1 + __n2, oneapi::dpl::__ranges::__size(__rng3));

        assert(__n1 > 0 || __n2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__q);

        // Empirical number of values to process per work-item
        const _IdType __chunk = __q.get_device().is_cpu() ? 128 : 4;

        const _IdType __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        auto __result = __create_parallel_merge_return_data<_OutSizeLimit, _Range1, _Range2, _IdType>(
            __q, /*__split_points_count*/ 0);

        // Save sycl::event instance into the first element of __result
        std::get<0>(__result) = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);

            auto __stop_pos_acc = __get_parallel_merge_stop_pos_accessor_opt<_OutSizeLimit, _Range1, _Range2>(
                sycl::write_only, __cgh, __result, __dpl_sycl::__no_init{});

            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item) {
                auto __id = __item.get_linear_id();
                const _IdType __i_elem = __id * __chunk;

                const auto __n_merge = std::min<_IdType>(__chunk, __n - __i_elem);
                const auto __start =
                    __find_start_point(__rng1, _IdType{0}, __n1, __rng2, _IdType{0}, __n2, __i_elem, __comp,
                                       __proj1, __proj2);

                [[maybe_unused]] const std::pair __ends =
                    __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __n_merge, __n1,
                                   __n2, __comp, __proj1, __proj2, __n);

                if constexpr (_OutSizeLimit{})
                {
                    // The last WI does additional work
                    if (__id == __steps - 1)
                        __stop_pos_acc.__data()[0] = {std::get<0>(__ends), std::get<1>(__ends)};
                }
            });
        });

        return std::move(__result);
    }
};

template <typename _OutSizeLimit, typename _IdType, typename _CustomName, typename _DiagonalsKernelName,
          typename _MergeKernelName>
struct __parallel_merge_submitter_large;

template <typename _OutSizeLimit, typename _IdType, typename _CustomName, typename... _DiagonalsKernelName,
          typename... _MergeKernelName>
struct __parallel_merge_submitter_large<_OutSizeLimit, _IdType, _CustomName,
                                        __internal::__optional_kernel_name<_DiagonalsKernelName...>,
                                        __internal::__optional_kernel_name<_MergeKernelName...>>
{
  private:
    struct nd_range_params
    {
        std::size_t base_diag_count = 0;
        std::size_t steps_between_two_base_diags = 0;
        _IdType chunk = 0;
        _IdType steps = 0;
    };

    // Calculate nd-range parameters
    nd_range_params
    eval_nd_range_params(const sycl::queue& __q, const std::size_t __n) const
    {
        // Empirical number of values to process per work-item
        const std::uint8_t __chunk = __q.get_device().is_cpu() ? 128 : 4;

        const _IdType __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);
        // TODO required to evaluate this value based on available SLM size for each work-group.
        const _IdType __base_diag_count = 32 * 1'024;
        const _IdType __steps_between_two_base_diags =
            oneapi::dpl::__internal::__dpl_ceiling_div(__steps, __base_diag_count);

        return {__base_diag_count, __steps_between_two_base_diags, __chunk, __steps};
    }

    // Calculation of split points on each base diagonal
    template <typename _Range1, typename _Range2, typename _Compare, typename _Proj1, typename _Proj2>
    sycl::event
    eval_split_points_for_groups(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _IdType __n, _Compare __comp,
                                 _Proj1 __proj1, _Proj2 __proj2, const nd_range_params& __nd_range_params,
                                 _split_points_device_storage_t<_IdType>& __base_diagonals_sp_global_storage) const
    {
        const _IdType __n1 = oneapi::dpl::__ranges::__size(__rng1);
        const _IdType __n2 = oneapi::dpl::__ranges::__size(__rng2);

        const _IdType __base_diag_chunk = __nd_range_params.steps_between_two_base_diags * __nd_range_params.chunk;

        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2);

            auto __base_diagonals_sp_global_acc =
                __get_accessor(sycl::write_only, __base_diagonals_sp_global_storage, __cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<_DiagonalsKernelName...>(
                sycl::range</*dim=*/1>(__nd_range_params.base_diag_count + 1), [=](sycl::item</*dim=*/1> __item) {
                    auto __global_idx = __item.get_linear_id();

                    const _IdType __i_elem = __global_idx * __base_diag_chunk;

                    __base_diagonals_sp_global_acc.__data()[__global_idx] =
                        __i_elem == 0
                            ? _split_point_t<_IdType>{0, 0}
                            : (__i_elem < __n ? __find_start_point(__rng1, _IdType{0}, __n1, __rng2, _IdType{0}, __n2,
                                                                   __i_elem, __comp, __proj1, __proj2)
                                              : _split_point_t<_IdType>{__n1, __n2});
                });
        });
    }

    // Process parallel merge
    template <typename _Range1, typename _Range2, typename _Range3, typename _Compare, typename _Proj1, typename _Proj2>
    sycl::event
    run_parallel_merge(const sycl::event& __event, sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2,
                       _Range3&& __rng3, _Compare __comp, _Proj1 __proj1, _Proj2 __proj2,
                       const nd_range_params& __nd_range_params,
                       __parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>& __result_data) const
    {
        const _IdType __n1 = oneapi::dpl::__ranges::__size(__rng1);
        const _IdType __n2 = oneapi::dpl::__ranges::__size(__rng2);
        const _IdType __n = std::min<_IdType>(__n1 + __n2, oneapi::dpl::__ranges::__size(__rng3));

        auto& __base_diagonals_sp_global_storage =
            __get_parallel_merge_sp_storage<_OutSizeLimit, _IdType, _Range1, _Range2>(__result_data);

        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);

            auto __base_diagonals_sp_global_acc =
                __get_accessor(sycl::read_only, __base_diagonals_sp_global_storage, __cgh);

            auto __stop_pos_acc = __get_parallel_merge_stop_pos_accessor_opt<_OutSizeLimit, _Range1, _Range2>(
                sycl::write_only, __cgh, __result_data, __dpl_sycl::__no_init{});

            __cgh.depends_on(__event);

            __cgh.parallel_for<_MergeKernelName...>(
                sycl::range</*dim=*/1>(__nd_range_params.steps), [=](sycl::item</*dim=*/1> __item) {
                    auto __global_idx = __item.get_linear_id();
                    const _IdType __i_elem = __global_idx * __nd_range_params.chunk;

                    auto __base_diagonals_sp_global_ptr = __base_diagonals_sp_global_acc.__data();
                    auto __diagonal_idx = __global_idx / __nd_range_params.steps_between_two_base_diags;

                    _split_point_t<_IdType> __start;
                    if (__global_idx % __nd_range_params.steps_between_two_base_diags != 0)
                    {
                        const _split_point_t<_IdType> __sp_left = __base_diagonals_sp_global_ptr[__diagonal_idx];
                        const _split_point_t<_IdType> __sp_right = __base_diagonals_sp_global_ptr[__diagonal_idx + 1];

                        __start = __find_start_point(__rng1, __sp_left.first, __sp_right.first, __rng2,
                                                     __sp_left.second, __sp_right.second, __i_elem, __comp,
                                                     __proj1, __proj2);
                    }
                    else
                    {
                        __start = __base_diagonals_sp_global_ptr[__diagonal_idx];
                    }

                    [[maybe_unused]] const std::pair __ends =
                        __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem,
                                       __nd_range_params.chunk, __n1, __n2, __comp, __proj1, __proj2, __n);

                    if constexpr (_OutSizeLimit{})
                    {
                        // The last WI does additional work
                        if (__global_idx == __nd_range_params.steps - 1)
                            __stop_pos_acc.__data()[0] = {std::get<0>(__ends), std::get<1>(__ends)};
                    }
                });
        });
    }

  public:
    template <typename _Range1, typename _Range2, typename _Range3, typename _Compare, typename _Proj1, typename _Proj2>
    __parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>
    operator()(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp, _Proj1 __proj1,
               _Proj2 __proj2) const
    {
        const _IdType __n1 = oneapi::dpl::__ranges::__size(__rng1);
        const _IdType __n2 = oneapi::dpl::__ranges::__size(__rng2);
        assert(__n1 > 0 || __n2 > 0);

        const _IdType __n = std::min<_IdType>(__n1 + __n2, oneapi::dpl::__ranges::__size(__rng3));

        _PRINT_INFO_IN_DEBUG_MODE(__q);

        // Calculate nd-range parameters
        const nd_range_params __nd_range_params = eval_nd_range_params(__q, __n);

        auto __result = __create_parallel_merge_return_data<_OutSizeLimit, _Range1, _Range2, _IdType>(
            __q, /*__split_points_count*/ __nd_range_params.base_diag_count + 1);

        // Find split-points on the base diagonals
        sycl::event __event = eval_split_points_for_groups(
            __q, __rng1, __rng2, __n, __comp, __proj1, __proj2, __nd_range_params,
            __get_parallel_merge_sp_storage<_OutSizeLimit, _IdType, _Range1, _Range2>(__result));

        // Merge data using split points on each diagonal
        // Save sycl::event instance into the first element of __result
        std::get<0>(__result) = run_parallel_merge(__event, __q, __rng1, __rng2, __rng3, __comp, __proj1, __proj2,
                                                   __nd_range_params, __result);

        return std::move(__result);
    }
};

template <typename... _Name>
class __merge_kernel_name;

template <typename... _Name>
class __merge_kernel_name_large;

template <typename... _Name>
class __diagonals_kernel_name;

template <typename _Tp>
constexpr std::size_t
__get_starting_size_limit_for_large_submitter()
{
    return 4 * 1'048'576; // 4 MB
}

template <>
constexpr std::size_t
__get_starting_size_limit_for_large_submitter<int>()
{
    return 16 * 1'048'576; // 16 MB
}

template <typename _CustomName, typename _OutSizeLimit = std::false_type, typename _Range1, typename _Range2,
          typename _Range3, typename _Compare, typename _Proj1, typename _Proj2>
__parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>
__parallel_merge_impl(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp,
                      _Proj1 __proj1, _Proj2 __proj2)
{
    using __value_type = oneapi::dpl::__internal::__value_t<_Range3>;
    const std::size_t __n =
        std::min<std::size_t>(oneapi::dpl::__ranges::__size(__rng1) + oneapi::dpl::__ranges::__size(__rng2),
                              oneapi::dpl::__ranges::__size(__rng3));
    if (__n < __get_starting_size_limit_for_large_submitter<__value_type>())
    {
        using _WiIndex = std::uint32_t;
        static_assert(__get_starting_size_limit_for_large_submitter<__value_type>() <=
                      std::numeric_limits<_WiIndex>::max());
        using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _WiIndex>>;
        return __parallel_merge_submitter<_OutSizeLimit, _WiIndex, _MergeKernelName>()(
            __q, std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2), std::forward<_Range3>(__rng3), __comp,
            __proj1, __proj2);
    }
    else
    {
        if (__n <= std::numeric_limits<std::uint32_t>::max())
        {
            using _WiIndex = std::uint32_t;
            using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __diagonals_kernel_name<_CustomName, _WiIndex>>;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_OutSizeLimit, _WiIndex, _CustomName, _DiagonalsKernelName,
                                                    _MergeKernelName>()(
                __q, std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2), std::forward<_Range3>(__rng3),
                __comp, __proj1, __proj2);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __diagonals_kernel_name<_CustomName, _WiIndex>>;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_OutSizeLimit, _WiIndex, _CustomName, _DiagonalsKernelName,
                                                    _MergeKernelName>()(
                __q, std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2), std::forward<_Range3>(__rng3),
                __comp, __proj1, __proj2);
        }
    }
}

template <typename _OutSizeLimit = std::false_type, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _Range3, typename _Compare, typename _Proj1, typename _Proj2>
__parallel_merge_return_data_t<_OutSizeLimit, _Range1, _Range2>
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();
    return __parallel_merge_impl<_CustomName, _OutSizeLimit>(__q_local, std::forward<_Range1>(__rng1),
                                                             std::forward<_Range2>(__rng2),
                                                             std::forward<_Range3>(__rng3), __comp, __proj1, __proj2);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
