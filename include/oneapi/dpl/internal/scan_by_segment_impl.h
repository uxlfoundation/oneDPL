/* Copyright (c) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *  
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _ONEDPL_SCAN_BY_SEGMENT_IMPL_H
#define _ONEDPL_SCAN_BY_SEGMENT_IMPL_H

#if _ONEDPL_BACKEND_SYCL

#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <algorithm>

#include "../pstl/algorithm_fwd.h"
#include "../pstl/parallel_backend.h"
#include "../pstl/hetero/utils_hetero.h"

#include "../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../pstl/hetero/dpcpp/unseq_backend_sycl.h"
#include "../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

#include "../pstl/hetero/dpcpp/sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace internal
{

// This function is responsible for performing an exclusive segmented scan between work items in shared local memory.
// Each work item passes __delta_local_id which is the distance to the closest lower indexed work item that
// processes a segment end. A SLM accessor that is twice the work-group size must be provided.
// Example across eight work items with addition as the binary operation :
// -----------------------------
// Local ID    : 0 1 2 3 4 5 6 7
// Accumulator : 2 3 1 3 0 5 2 1
// Delta       : 0 1 2 3 0 1 0 0
// -----------------------------
// Return Vals : 0 2 5 6 9 0 5 2
// -----------------------------
// __wg_segmented_scan is a derivative work and the reason for the additional copyright notice.
template <typename _NdItem, typename _LocalAcc, typename _IdxType, typename _ValueType, typename _BinaryOp>
_ValueType
__wg_segmented_scan(_NdItem __item, _LocalAcc __local_acc, _IdxType __local_id, _IdxType __delta_local_id,
                    _ValueType __accumulator, _ValueType __identity, _BinaryOp __binary_op, ::std::size_t __wgroup_size)
{
    _IdxType __first = 0;
    __local_acc[__local_id] = __accumulator;

    __dpl_sycl::__group_barrier(__item);

    for (::std::size_t __i = 1; __i < __wgroup_size; __i *= 2)
    {
        if (__delta_local_id >= __i)
            __accumulator = __binary_op(__local_acc[__first + __local_id - __i], __accumulator);

        __first = __wgroup_size - __first;
        __local_acc[__first + __local_id] = __accumulator;
        __dpl_sycl::__group_barrier(__item);
    }

    return (__local_id ? __local_acc[__first + __local_id - 1] : __identity);
}

template <bool __is_inclusive, typename... Name>
class __seg_scan_wg_kernel;

template <bool __is_inclusive, typename... Name>
class __seg_scan_prefix_kernel;

template <bool __is_inclusive>
struct __sycl_scan_by_segment_impl
{
    template <typename... _Name>
    using _SegScanWgPhase = __seg_scan_wg_kernel<__is_inclusive, _Name...>;

    template <typename... _Name>
    using _SegScanPrefixPhase = __seg_scan_prefix_kernel<__is_inclusive, _Name...>;

    template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
              typename _BinaryPredicate, typename _BinaryOperator, typename _T>
    void
    operator()(_BackendTag, _ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_values,
               _BinaryPredicate __binary_pred, _BinaryOperator __binary_op, _T __init, _T __identity)
    {
        using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

        using _SegScanWgKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            _SegScanWgPhase, _CustomName, _ExecutionPolicy, _Range1, _Range2, _Range3, _BinaryPredicate,
            _BinaryOperator>;
        using _SegScanPrefixKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            _SegScanPrefixPhase, _CustomName, _ExecutionPolicy, _Range1, _Range2, _Range3, _BinaryPredicate,
            _BinaryOperator>;

        using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;

        const ::std::size_t __n = __keys.size();

        constexpr ::std::uint16_t __vals_per_item =
            4; // Assigning 4 elements per work item resulted in best performance on gpu.

        // Limit the work-group size to prevent large sizes on CPUs. Empirically found value.
        // This value exceeds the current practical limit for GPUs, but may need to be re-evaluated in the future.
        std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec, (std::size_t)2048);

        // We require 2 * sizeof(__val_type) * __wgroup_size of SLM for the work group segmented scan. We add
        // an additional sizeof(__val_type) * __wgroup_size requirement to ensure sufficient SLM for the group algorithms.
        __wgroup_size =
            oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, 3 * sizeof(__val_type), __wgroup_size);

#if _ONEDPL_COMPILE_KERNEL
        auto __seg_scan_wg_kernel =
            __par_backend_hetero::__internal::__kernel_compiler<_SegScanWgKernel>::__compile(__exec);
        auto __seg_scan_prefix_kernel =
            __par_backend_hetero::__internal::__kernel_compiler<_SegScanPrefixKernel>::__compile(__exec);
        __wgroup_size =
            ::std::min({__wgroup_size, oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_scan_wg_kernel),
                        oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_scan_prefix_kernel)});
#endif

        ::std::size_t __n_groups = __internal::__dpl_ceiling_div(__n, __wgroup_size * __vals_per_item);

        auto __partials =
            oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups).get_buffer();

        // the number of segment ends found in each work group
        auto __seg_ends =
            oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, bool>(__exec, __n_groups).get_buffer();

        // 1. Work group reduction
        auto __wg_scan = __exec.queue().submit([&](sycl::handler& __cgh) {
            auto __partials_acc = __partials.template get_access<sycl::access_mode::write>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::write>(__cgh);

            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __values, __out_values);

            __dpl_sycl::__local_accessor<__val_type> __loc_acc(2 * __wgroup_size, __cgh);

#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
            __cgh.use_kernel_bundle(__seg_scan_wg_kernel.get_kernel_bundle());
#endif
            __cgh.parallel_for<_SegScanWgKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                __seg_scan_wg_kernel,
#endif
                sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                    __val_type __accumulator = __identity;

                    auto __group = __item.get_group();
                    ::std::size_t __global_id = __item.get_global_id(0);
                    ::std::size_t __local_id = __item.get_local_id(0);

                    // 1a. Perform a serial scan within the work item over assigned elements. Store partial
                    // reductions in work item memory, and write the accumulated value and number of counted
                    // segments into work group memory.
                    ::std::size_t __start = __global_id * __vals_per_item;
                    ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);

                    // First work item must set their accumulator to the provided init
                    if (__global_id == 0)
                    {
                        __accumulator = __init;
                    }
                    // TODO: We should use a more meaningful name through enum instead of -1
                    constexpr ::std::int32_t __no_segment_break = -1;
                    // signed to allow flag for no segment break found
                    ::std::int32_t __max_end = __no_segment_break;

                    for (::std::size_t __i = __start; __i < __end; ++__i)
                    {
                        if constexpr (__is_inclusive)
                        {
                            __accumulator = __binary_op(__accumulator, __values[__i]);
                            __out_values[__i] = __accumulator;
                        }
                        else // exclusive scan
                        {
                            __out_values[__i] = __accumulator;
                            __accumulator = __binary_op(__accumulator, __values[__i]);
                        }

                        // reset the accumulator to init if we reach the end of a segment
                        // (init for inclusive scan is the identity)
                        if (__n - 1 == __i || !__binary_pred(__keys[__i], __keys[__i + 1]))
                        {
                            __accumulator = __init;
                            __max_end = __local_id;
                        }
                    }

                    // 1b. Perform a work group scan to find the carry in value to apply to each item.
                    ::std::int32_t __closest_seg_id = __dpl_sycl::__inclusive_scan_over_group(
                        __group, __max_end, __dpl_sycl::__maximum<decltype(__max_end)>());

                    bool __group_has_segment_break = (__closest_seg_id != __no_segment_break);

                    //get rid of no segment end found flag
                    __closest_seg_id = ::std::max(::std::int32_t(0), __closest_seg_id);
                    __val_type __carry_in =
                        __wg_segmented_scan(__item, __loc_acc, __local_id, __local_id - __closest_seg_id, __accumulator,
                                            __identity, __binary_op, __wgroup_size); // need to use exclusive scan delta

                    // 1c. Update local partial reductions and write to global memory.
                    for (::std::size_t __i = __start; __i < __end; ++__i)
                    {
                        __out_values[__i] = __binary_op(__carry_in, __out_values[__i]);

                        if (__i >= __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                            break;
                    }

                    if (__local_id == __wgroup_size - 1) // last work item writes the group's carry out
                    {
                        ::std::size_t __group_id = __item.get_group(0);

                        __seg_ends_acc[__group_id] = __group_has_segment_break;

                        if (__max_end == __no_segment_break)
                        {
                            __partials_acc[__group_id] = __binary_op(__carry_in, __accumulator);
                        }
                        else
                        {
                            __partials_acc[__group_id] = __accumulator;
                        }
                    }
                });
        });

        // 2. Apply work group carry outs, calculate output indices, and load results into correct indices.
        __exec.queue()
            .submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_values);

                auto __partials_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
                auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);

                __cgh.depends_on(__wg_scan);

                __dpl_sycl::__local_accessor<__val_type> __loc_partials_acc(__wgroup_size, __cgh);

                __dpl_sycl::__local_accessor<bool> __loc_seg_ends_acc(__wgroup_size, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
                __cgh.use_kernel_bundle(__seg_scan_prefix_kernel.get_kernel_bundle());
#endif
                __cgh.parallel_for<_SegScanPrefixKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                    __seg_scan_prefix_kernel,
#endif
                    sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                        auto __group = __item.get_group();
                        ::std::size_t __group_id = __item.get_group(0);
                        ::std::size_t __global_id = __item.get_global_id(0);
                        ::std::size_t __local_id = __item.get_local_id(0);
                        ::std::size_t __start = __global_id * __vals_per_item;
                        ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);

                        ::std::int32_t __wg_agg_idx = __group_id - 1;
                        __val_type __agg_collector = __identity;

                        //TODO:  just launch with one fewer group and adjust indexing since group zero can skip phase
                        if (__group_id != 0)
                        {
                            // 2a. Calculate the work group's carry-in value.
                            bool __ag_exists = __start < __n;
                            // local reductions followed by a downsweep
                            // TODO: Generalize this value
                            constexpr ::std::int32_t __vals_to_explore = 16;
                            bool __last_it = false;
                            __loc_seg_ends_acc[__local_id] = false;
                            __loc_partials_acc[__local_id] = __identity;

                            for (::std::int32_t __i = __wg_agg_idx - __vals_to_explore * __local_id; !__last_it;
                                 __i -= __wgroup_size * __vals_to_explore)
                            {
                                __val_type __local_collector = __identity;
                                // Parallel exploration phase
                                for (::std::int32_t __j = __i;
                                     __j > __dpl_sycl::__maximum<::std::int32_t>{}(-1L, __i - __vals_to_explore); --__j)
                                {
                                    __local_collector = __binary_op(__partials_acc[__j], __local_collector);
                                    if (__seg_ends_acc[__j] || __j == 0)
                                    {
                                        __loc_seg_ends_acc[__local_id] = true;
                                        break;
                                    }
                                }
                                __loc_partials_acc[__local_id] = __local_collector;
                                __dpl_sycl::__group_barrier(__item);
                                // Serial aggregate collection and synchronization
                                if (__local_id == 0)
                                {
                                    for (::std::size_t __j = 0; __j < __wgroup_size; ++__j)
                                    {
                                        __agg_collector = __binary_op(__loc_partials_acc[__j], __agg_collector);
                                        if (__loc_seg_ends_acc[__j])
                                        {
                                            __last_it = true;
                                            break;
                                        }
                                    }
                                }
                                __agg_collector = __dpl_sycl::__group_broadcast(__item.get_group(), __agg_collector);
                                __last_it = __dpl_sycl::__group_broadcast(__item.get_group(), __last_it);
                            }
                            __ag_exists = __dpl_sycl::__any_of_group(__group, __ag_exists);

                            // If no aggregate exists, all work items return as no more work is needed.
                            if (!__ag_exists)
                                return;

                            // 2c. Second pass over the keys, reidentifying end segments and applying work group
                            // aggregates if appropriate.
                            ::std::size_t __end_nm1_cap =
                                __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n - 1);
                            ::std::size_t __local_min_key_idx = __n - 1;

                            // Find the smallest end index in the work group
                            for (::std::size_t __i = __end_nm1_cap - 1; __i >= __start; --__i)
                            {
                                if (!__binary_pred(__keys[__i], __keys[__i + 1]))
                                {
                                    __local_min_key_idx = __i;
                                }
                            }

                            ::std::size_t __wg_min_seg_end = __dpl_sycl::__reduce_over_group(
                                __group, __local_min_key_idx, __dpl_sycl::__minimum<::std::size_t>());

                            // apply work group aggregates
                            for (::std::size_t __i = __start;
                                 __i < __dpl_sycl::__minimum<decltype(__end)>{}(__wg_min_seg_end + 1, __end); ++__i)
                            {
                                __out_values[__i] = __binary_op(__agg_collector, __out_values[__i]);
                            }
                        }
                    });
            })
            .wait();
    }
};

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename T, typename BinaryPredicate, typename Operator, typename Inclusive>
OutputIterator
__scan_by_segment_impl_common(__internal::__hetero_tag<_BackendTag>, Policy&& policy, InputIterator1 first1,
                              InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init,
                              BinaryPredicate binary_pred, Operator binary_op, Inclusive)
{
    const auto n = ::std::distance(first1, last1);

    // Check for empty element ranges
    if (n <= 0)
        return result;

    namespace __bknd = oneapi::dpl::__par_backend_hetero;

    auto keep_keys = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto key_buf = keep_keys(first1, last1);
    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(first2, first2 + n);
    auto keep_value_outputs = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, OutputIterator>();
    auto value_output_buf = keep_value_outputs(result, result + n);
    auto buf_view = key_buf.all_view();
    using iter_value_t = typename ::std::iterator_traits<InputIterator2>::value_type;

    constexpr iter_value_t identity = unseq_backend::__known_identity<Operator, iter_value_t>;

    __sycl_scan_by_segment_impl<Inclusive::value>()(_BackendTag{}, ::std::forward<Policy>(policy), key_buf.all_view(),
                                                    value_buf.all_view(), value_output_buf.all_view(), binary_pred,
                                                    binary_op, init, identity);
    return result + n;
}

} // namespace internal
} // namespace dpl
} // namespace oneapi
#endif
#endif
