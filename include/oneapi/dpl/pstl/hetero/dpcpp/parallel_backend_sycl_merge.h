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
template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
_split_point_t<_Index>
__find_start_point(const _Rng1& __rng1, const _Index __rng1_from, _Index __rng1_to, const _Rng2& __rng2,
                   const _Index __rng2_from, _Index __rng2_to, const _Index __i_elem, _Compare __comp)
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

    const __it_t __res =
        std::lower_bound(__diag_it_begin, __diag_it_end, false,
                         [&__rng1, &__rng2, __index_sum, __comp](_Index __idx, const bool __value) mutable {
                             return __value == __comp(__rng2[__index_sum - __idx], __rng1[__idx]);
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
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index, typename _Compare>
std::pair<_Index, _Index>
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3, const _Index __start1, const _Index __start2,
               const _Index __start3, const _Index __chunk, const _Index __n1, const _Index __n2, _Compare __comp,
               const _Index __n3 = 0)
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
                                                          __comp(__rng2[__rng2_idx], __rng1[__rng1_idx])))
                                     ? __rng2[__rng2_idx++]
                                     : __rng1[__rng1_idx++];
        }
        else
        {
            // TODO required to understand why the usual if-else is slower then ternary operator
            if (!__rng1_idx_less_n1 ||
                (__rng1_idx_less_n1 && __rng2_idx_less_n2 && __comp(__rng2[__rng2_idx], __rng1[__rng1_idx])))
                __rng3[__rng3_idx] = __rng2[__rng2_idx++];
            else
                __rng3[__rng3_idx] = __rng1[__rng1_idx++];
        }
    }
    return {__rng1_idx, __rng2_idx};
}

// Please see the comment for __parallel_for_small_submitter for optional kernel name explanation
template <typename _OutSizeLimit, typename _IdType, typename _Name>
struct __parallel_merge_submitter;

template <typename _OutSizeLimit, typename _IdType, typename... _Name>
struct __parallel_merge_submitter<_OutSizeLimit, _IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    __future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
    operator()(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        const _IdType __n = std::min<_IdType>(__n1 + __n2, __rng3.size());

        assert(__n1 > 0 || __n2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__q);

        // Empirical number of values to process per work-item
        const _IdType __chunk = __q.get_device().is_cpu() ? 128 : 4;

        const _IdType __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        using __val_t = _split_point_t<_IdType>;
        using _NResults = std::conditional_t<_OutSizeLimit{}, std::integral_constant<std::size_t, 1>,
                                             std::integral_constant<std::size_t, 0>>;
        using __result_and_scratch_storage_t = __result_and_scratch_storage<__val_t, _NResults::value>;
        __result_and_scratch_storage_t* __p_res_storage = nullptr;

        if constexpr (_OutSizeLimit{})
            __p_res_storage = new __result_and_scratch_storage_t(__q, 0);
        else
            assert(__rng3.size() >= __n1 + __n2);

        std::shared_ptr<__result_and_scratch_storage_base> __p_result_and_scratch_storage_base(
            static_cast<__result_and_scratch_storage_base*>(__p_res_storage));

        auto __event = __q.submit([&__rng1, &__rng2, &__rng3, __p_res_storage, __comp, __chunk, __steps, __n, __n1,
                                   __n2](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            auto __result_acc = __get_acc(__p_res_storage, __cgh);

            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                auto __id = __item_id.get_linear_id();
                const _IdType __i_elem = __id * __chunk;

                const auto __n_merge = std::min<_IdType>(__chunk, __n - __i_elem);
                const auto __start =
                    __find_start_point(__rng1, _IdType{0}, __n1, __rng2, _IdType{0}, __n2, __i_elem, __comp);

                [[maybe_unused]] const std::pair __ends =
                    __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __n_merge, __n1,
                                   __n2, __comp, __n);

                if constexpr (_OutSizeLimit{})
                    if (__id == __steps - 1) //the last WI does additional work
                    {
                        auto __res_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__result_acc);
                        *__res_ptr = __ends;
                    }
            });
        });

        // Save the raw pointer into a shared_ptr to return it in __future and extend the lifetime of the storage.
        // We should return the same thing in the second param of __future for compatibility
        // with the returning value in __parallel_merge_submitter_large::operator()
        return __future{std::move(__event), std::move(__p_result_and_scratch_storage_base)};
    }

  private:
    template <typename _Storage>
    static constexpr auto
    __get_acc(_Storage* __p_res_storage, sycl::handler& __cgh)
    {
        if constexpr (_OutSizeLimit{})
            return __p_res_storage->template __get_result_acc<sycl::access_mode::write>(__cgh, __dpl_sycl::__no_init{});
        else
            return int{0};
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
    template <typename _Range1, typename _Range2>
    nd_range_params
    eval_nd_range_params(const sycl::queue& __q, const _Range1& __rng1, const _Range2& __rng2,
                         const std::size_t __n) const
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
    template <typename _Range1, typename _Range2, typename _Compare, typename _Storage>
    sycl::event
    eval_split_points_for_groups(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _IdType __n, _Compare __comp,
                                 const nd_range_params& __nd_range_params,
                                 _Storage& __base_diagonals_sp_global_storage) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();

        const _IdType __base_diag_chunk = __nd_range_params.steps_between_two_base_diags * __nd_range_params.chunk;

        return __q.submit([&__rng1, &__rng2, __comp, __nd_range_params, __base_diagonals_sp_global_storage, __n1, __n2,
                           __n, __base_diag_chunk](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2);
            auto __base_diagonals_sp_global_acc =
                __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::write>(
                    __cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<_DiagonalsKernelName...>(
                sycl::range</*dim=*/1>(__nd_range_params.base_diag_count + 1), [=](sycl::item</*dim=*/1> __item_id) {
                    auto __global_idx = __item_id.get_linear_id();
                    auto __base_diagonals_sp_global_ptr =
                        _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    const _IdType __i_elem = __global_idx * __base_diag_chunk;

                    __base_diagonals_sp_global_ptr[__global_idx] =
                        __i_elem == 0 ? _split_point_t<_IdType>{0, 0}
                                      : (__i_elem < __n ? __find_start_point(__rng1, _IdType{0}, __n1, __rng2,
                                                                             _IdType{0}, __n2, __i_elem, __comp)
                                                        : _split_point_t<_IdType>{__n1, __n2});
                });
        });
    }

    // Process parallel merge
    template <typename _Range1, typename _Range2, typename _Range3, typename _Compare, typename _Storage>
    sycl::event
    run_parallel_merge(const sycl::event& __event, sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2,
                       _Range3&& __rng3, _Compare __comp, const nd_range_params& __nd_range_params,
                       const _Storage& __base_diagonals_sp_global_storage) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        const _IdType __n = std::min<_IdType>(__n1 + __n2, __rng3.size());

        return __q.submit([&__event, &__rng1, &__rng2, &__rng3, __n, __comp, __nd_range_params,
                           __base_diagonals_sp_global_storage, __n1, __n2](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            auto __base_diagonals_sp_global_acc =
                __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read>(__cgh);

            auto __result_acc = __get_acc(__base_diagonals_sp_global_storage, __cgh);

            __cgh.depends_on(__event);

            __cgh.parallel_for<_MergeKernelName...>(
                sycl::range</*dim=*/1>(__nd_range_params.steps), [=](sycl::item</*dim=*/1> __item_id) {
                    auto __global_idx = __item_id.get_linear_id();
                    const _IdType __i_elem = __global_idx * __nd_range_params.chunk;

                    auto __base_diagonals_sp_global_ptr =
                        _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);
                    auto __diagonal_idx = __global_idx / __nd_range_params.steps_between_two_base_diags;

                    _split_point_t<_IdType> __start;
                    if (__global_idx % __nd_range_params.steps_between_two_base_diags != 0)
                    {
                        const _split_point_t<_IdType> __sp_left = __base_diagonals_sp_global_ptr[__diagonal_idx];
                        const _split_point_t<_IdType> __sp_right = __base_diagonals_sp_global_ptr[__diagonal_idx + 1];

                        __start = __find_start_point(__rng1, __sp_left.first, __sp_right.first, __rng2,
                                                     __sp_left.second, __sp_right.second, __i_elem, __comp);
                    }
                    else
                    {
                        __start = __base_diagonals_sp_global_ptr[__diagonal_idx];
                    }

                    [[maybe_unused]] const std::pair __ends =
                        __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem,
                                       __nd_range_params.chunk, __n1, __n2, __comp, __n);

                    if constexpr (_OutSizeLimit{})
                        if (__global_idx == __nd_range_params.steps - 1)
                        {
                            auto __res_ptr = _Storage::__get_usm_or_buffer_accessor_ptr(__result_acc);
                            *__res_ptr = __ends;
                        }
                });
        });
    }

    template <typename _Storage>
    static constexpr auto
    __get_acc(const _Storage& __base_diagonals_sp_global_storage, sycl::handler& __cgh)
    {
        if constexpr (_OutSizeLimit{})
            return __base_diagonals_sp_global_storage.template __get_result_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});
        else
            return int{0};
    }

  public:
    template <typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    __future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
    operator()(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        assert(__n1 > 0 || __n2 > 0);

        const _IdType __n = std::min<_IdType>(__n1 + __n2, __rng3.size());

        _PRINT_INFO_IN_DEBUG_MODE(__q);

        // Calculate nd-range parameters
        const nd_range_params __nd_range_params = eval_nd_range_params(__q, __rng1, __rng2, __n);

        // Create storage to save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
        using __val_t = _split_point_t<_IdType>;
        using _NResults = std::conditional_t<_OutSizeLimit{}, std::integral_constant<std::size_t, 1>,
                                             std::integral_constant<std::size_t, 0>>;
        using __result_and_scratch_storage_t = __result_and_scratch_storage<__val_t, _NResults::value>;
        auto __p_base_diagonals_sp_global_storage =
            new __result_and_scratch_storage_t(__q, __nd_range_params.base_diag_count + 1);

        // Save the raw pointer into a shared_ptr to return it in __future and extend the lifetime of the storage.
        std::shared_ptr<__result_and_scratch_storage_base> __p_result_and_scratch_storage_base(
            static_cast<__result_and_scratch_storage_base*>(__p_base_diagonals_sp_global_storage));

        // Find split-points on the base diagonals
        sycl::event __event = eval_split_points_for_groups(__q, __rng1, __rng2, __n, __comp, __nd_range_params,
                                                           *__p_base_diagonals_sp_global_storage);

        // Merge data using split points on each diagonal
        __event = run_parallel_merge(__event, __q, __rng1, __rng2, __rng3, __comp, __nd_range_params,
                                     *__p_base_diagonals_sp_global_storage);

        return __future{std::move(__event), std::move(__p_result_and_scratch_storage_base)};
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

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
          typename _OutSizeLimit = std::false_type>
__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp, _OutSizeLimit = {})
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    using __value_type = oneapi::dpl::__internal::__value_t<_Range3>;

    sycl::queue __q_local = __exec.queue();

    const std::size_t __n = std::min<std::size_t>(__rng1.size() + __rng2.size(), __rng3.size());
    if (__n < __get_starting_size_limit_for_large_submitter<__value_type>())
    {
        using _WiIndex = std::uint32_t;
        static_assert(__get_starting_size_limit_for_large_submitter<__value_type>() <=
                      std::numeric_limits<_WiIndex>::max());
        using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _WiIndex>>;
        return __parallel_merge_submitter<_OutSizeLimit, _WiIndex, _MergeKernelName>()(
            __q_local, std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2), std::forward<_Range3>(__rng3),
            __comp);
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
                                                    _MergeKernelName>()(__q_local, std::forward<_Range1>(__rng1),
                                                                        std::forward<_Range2>(__rng2),
                                                                        std::forward<_Range3>(__rng3), __comp);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __diagonals_kernel_name<_CustomName, _WiIndex>>;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_OutSizeLimit, _WiIndex, _CustomName, _DiagonalsKernelName,
                                                    _MergeKernelName>()(__q_local, std::forward<_Range1>(__rng1),
                                                                        std::forward<_Range2>(__rng2),
                                                                        std::forward<_Range3>(__rng3), __comp);
        }
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
