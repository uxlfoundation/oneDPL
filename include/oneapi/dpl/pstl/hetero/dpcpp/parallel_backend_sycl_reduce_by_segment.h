// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_by_segment.h ---------------------------------===//
/*  Copyright (c) Intel Corporation
 *
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 *  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_H

#include <cstdint>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <array>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "utils_ranges_sycl.h"
#include "sycl_traits.h"

#include "../../utils.h"
#include "../../../internal/scan_by_segment_impl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename... Name>
class __seg_reduce_count_kernel;
template <typename... Name>
class __seg_reduce_offset_kernel;
template <typename... Name>
class __seg_reduce_wg_kernel;
template <typename... Name>
class __seg_reduce_prefix_kernel;

namespace
{
template <typename... _Name>
using _SegReduceCountPhase = __seg_reduce_count_kernel<_Name...>;
template <typename... _Name>
using _SegReduceOffsetPhase = __seg_reduce_offset_kernel<_Name...>;
template <typename... _Name>
using _SegReduceWgPhase = __seg_reduce_wg_kernel<_Name...>;
template <typename... _Name>
using _SegReducePrefixPhase = __seg_reduce_prefix_kernel<_Name...>;
} // namespace

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__difference_t<_Range3>
__parallel_reduce_by_segment_fallback(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                      _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                                      _Range4&& __out_values, _BinaryPredicate __binary_pred,
                                      _BinaryOperator __binary_op,
                                      /*known_identity=*/std::true_type)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    using _SegReduceCountKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
        _SegReduceCountPhase, _CustomName, _ExecutionPolicy, _Range1, _Range2, _Range3, _Range4, _BinaryPredicate,
        _BinaryOperator>;
    using _SegReduceOffsetKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
        _SegReduceOffsetPhase, _CustomName, _ExecutionPolicy, _Range1, _Range2, _Range3, _Range4, _BinaryPredicate,
        _BinaryOperator>;
    using _SegReduceWgKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
        _SegReduceWgPhase, _CustomName, _ExecutionPolicy, _Range1, _Range2, _Range3, _Range4, _BinaryPredicate,
        _BinaryOperator>;
    using _SegReducePrefixKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
        _SegReducePrefixPhase, _CustomName, _ExecutionPolicy, _Range1, _Range2, _Range3, _Range4, _BinaryPredicate,
        _BinaryOperator>;

    using __diff_type = oneapi::dpl::__internal::__difference_t<_Range3>;
    using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
    using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;

    const std::size_t __n = __keys.size();

    constexpr std::uint16_t __vals_per_item =
        16; // Each work item serially processes 16 items. Best observed performance on gpu

    // Limit the work-group size to prevent large sizes on CPUs. Empirically found value.
    // This value exceeds the current practical limit for GPUs, but may need to be re-evaluated in the future.
    std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec, (std::size_t)2048);

    // adjust __wgroup_size according to local memory limit. Double the requirement on __val_type due to sycl group algorithm's use
    // of SLM.
    __wgroup_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(
        __exec, sizeof(__key_type) + 2 * sizeof(__val_type), __wgroup_size);

#if _ONEDPL_COMPILE_KERNEL
    auto __seg_reduce_count_kernel =
        __par_backend_hetero::__internal::__kernel_compiler<_SegReduceCountKernel>::__compile(__exec);
    auto __seg_reduce_offset_kernel =
        __par_backend_hetero::__internal::__kernel_compiler<_SegReduceOffsetKernel>::__compile(__exec);
    auto __seg_reduce_wg_kernel =
        __par_backend_hetero::__internal::__kernel_compiler<_SegReduceWgKernel>::__compile(__exec);
    auto __seg_reduce_prefix_kernel =
        __par_backend_hetero::__internal::__kernel_compiler<_SegReducePrefixKernel>::__compile(__exec);
    __wgroup_size =
        std::min({__wgroup_size, oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_count_kernel),
                  oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_offset_kernel),
                  oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_wg_kernel),
                  oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_prefix_kernel)});
#endif

    std::size_t __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __wgroup_size * __vals_per_item);

    // intermediate reductions within a workgroup
    auto __partials =
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups).get_buffer();

    auto __end_idx = oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __diff_type>(__exec, 1).get_buffer();

    // the number of segment ends found in each work group
    auto __seg_ends =
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __diff_type>(__exec, __n_groups).get_buffer();

    // buffer that stores an exclusive scan of the results
    auto __seg_ends_scanned =
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __diff_type>(__exec, __n_groups).get_buffer();

    // 1. Count the segment ends in each workgroup
    auto __seg_end_identification = __exec.queue().submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __keys);
        auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::write>(__cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__seg_reduce_count_kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_SegReduceCountKernel>(
            sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT && _ONEDPL_LIBSYCL_PROGRAM_PRESENT
                                                                              __seg_reduce_count_kernel,
#endif
                                                                              sycl::nd_item<1> __item) {
                auto __group = __item.get_group();
                std::size_t __group_id = __item.get_group(0);
                std::uint32_t __local_id = __item.get_local_id(0);
                std::size_t __global_id = __item.get_global_id(0);

                std::size_t __start = __global_id * __vals_per_item;
                std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);
                std::size_t __item_segments = 0;

                // 1a. Work item scan to identify segment ends
                for (std::size_t __i = __start; __i < __end; ++__i)
                    if (__n - 1 == __i || !__binary_pred(__keys[__i], __keys[__i + 1]))
                        ++__item_segments;

                // 1b. Work group reduction
                std::size_t __num_segs = __dpl_sycl::__reduce_over_group(
                    __group, __item_segments, __dpl_sycl::__plus<decltype(__item_segments)>());

                // 1c. First work item writes segment count to global memory
                if (__local_id == 0)
                    __seg_ends_acc[__group_id] = __num_segs;
            });
    });

    // 1.5 Small single-group kernel
    auto __single_group_scan = __exec.queue().submit([&](sycl::handler& __cgh) {
        __cgh.depends_on(__seg_end_identification);
        auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);
        auto __seg_ends_scan_acc = __seg_ends_scanned.template get_access<sycl::access_mode::read_write>(__cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__seg_reduce_offset_kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_SegReduceOffsetKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT && _ONEDPL_LIBSYCL_PROGRAM_PRESENT
            __seg_reduce_offset_kernel,
#endif
            sycl::nd_range<1>{__wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                auto __beg = __dpl_sycl::__get_accessor_ptr(__seg_ends_acc);
                auto __out_beg = __dpl_sycl::__get_accessor_ptr(__seg_ends_scan_acc);
                __dpl_sycl::__joint_exclusive_scan(__item.get_group(), __beg, __beg + __n_groups, __out_beg,
                                                   __diff_type(0), sycl::plus<__diff_type>());
            });
    });

    // 2. Work group reduction
    auto __wg_reduce = __exec.queue().submit([&](sycl::handler& __cgh) {
        __cgh.depends_on(__single_group_scan);
        oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_keys, __out_values, __values);

        auto __partials_acc = __partials.template get_access<sycl::access_mode::read_write>(__cgh);
        auto __seg_ends_scan_acc = __seg_ends_scanned.template get_access<sycl::access_mode::read>(__cgh);
        __dpl_sycl::__local_accessor<__val_type> __loc_acc(2 * __wgroup_size, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__seg_reduce_wg_kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_SegReduceWgKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT && _ONEDPL_LIBSYCL_PROGRAM_PRESENT
            __seg_reduce_wg_kernel,
#endif
            sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                std::array<__val_type, __vals_per_item> __loc_partials;

                auto __group = __item.get_group();
                std::size_t __group_id = __item.get_group(0);
                std::size_t __local_id = __item.get_local_id(0);
                std::size_t __global_id = __item.get_global_id(0);

                // 2a. Lookup the number of prior segs
                auto __wg_num_prior_segs = __seg_ends_scan_acc[__group_id];

                // 2b. Perform a serial scan within the work item over assigned elements. Store partial
                // reductions in work group local memory.
                std::size_t __start = __global_id * __vals_per_item;
                std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);

                std::size_t __max_end = 0;
                std::size_t __item_segments = 0;
                auto __identity = unseq_backend::__known_identity<_BinaryOperator, __val_type>;

                __val_type __accumulator = __identity;
                for (std::size_t __i = __start; __i < __end; ++__i)
                {
                    __accumulator = __binary_op(__accumulator, __values[__i]);
                    if (__n - 1 == __i || !__binary_pred(__keys[__i], __keys[__i + 1]))
                    {
                        __loc_partials[__i - __start] = __accumulator;
                        ++__item_segments;
                        __max_end = __local_id;
                        __accumulator = __identity;
                    }
                }

                // 2c. Count the number of prior work segments cooperatively over group
                std::size_t __prior_segs_in_wg = __dpl_sycl::__exclusive_scan_over_group(
                    __group, __item_segments, __dpl_sycl::__plus<std::size_t>());
                std::size_t __start_idx = __wg_num_prior_segs + __prior_segs_in_wg;

                // 2d. Find the greatest segment end less than the current index (inclusive)
                std::size_t __closest_seg_id = __dpl_sycl::__inclusive_scan_over_group(
                    __group, __max_end, __dpl_sycl::__maximum<std::size_t>());

                // __wg_segmented_scan is a derivative work and responsible for the third header copyright
                __val_type __carry_in = oneapi::dpl::internal::__wg_segmented_scan(
                    __item, __loc_acc, __local_id, __local_id - __closest_seg_id, __accumulator, __identity,
                    __binary_op, __wgroup_size);

                // 2e. Update local partial reductions in first segment and write to global memory.
                bool __apply_aggs = true;
                std::size_t __item_offset = 0;

                // first item in group does not have any work-group aggregates to apply
                if (__local_id == 0)
                {
                    __apply_aggs = false;
                    if (__global_id == 0)
                    {
                        // first segment identifier is always the first key
                        __out_keys[0] = __keys[0];
                    }
                }

                // apply the aggregates and copy the locally stored values to destination buffer
                for (std::size_t __i = __start; __i < __end; ++__i)
                {
                    if (__i == __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                    {
                        std::size_t __idx = __start_idx + __item_offset;
                        if (__apply_aggs)
                        {
                            __out_values[__idx] = __binary_op(__carry_in, __loc_partials[__i - __start]);
                            __apply_aggs = false;
                        }
                        else
                        {
                            __out_values[__idx] = __loc_partials[__i - __start];
                        }
                        if (__i != __n - 1)
                        {
                            __out_keys[__idx + 1] = __keys[__i + 1];
                        }
                        ++__item_offset;
                    }
                }

                // 2f. Output the work group aggregate and total number of segments for use in phase 3.
                if (__local_id == __wgroup_size - 1) // last work item writes the group's carry out
                {
                    // If no segment ends in the item, the aggregates from previous work groups must be applied.
                    if (__max_end == 0)
                    {
                        // needs to be inclusive with last element
                        __partials_acc[__group_id] = __binary_op(__carry_in, __accumulator);
                    }
                    else
                    {
                        __partials_acc[__group_id] = __accumulator;
                    }
                }
            });
    });

    // 3. Apply inter work-group aggregates
    __exec.queue()
        .submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_keys, __out_values);

            auto __partials_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
            auto __seg_ends_scan_acc = __seg_ends_scanned.template get_access<sycl::access_mode::read>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);
            auto __end_idx_acc = __end_idx.template get_access<sycl::access_mode::write>(__cgh);

            __dpl_sycl::__local_accessor<__val_type> __loc_partials_acc(__wgroup_size, __cgh);
            __dpl_sycl::__local_accessor<__diff_type> __loc_seg_ends_acc(__wgroup_size, __cgh);

            __cgh.depends_on(__wg_reduce);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
            __cgh.use_kernel_bundle(__seg_reduce_prefix_kernel.get_kernel_bundle());
#endif
            __cgh.parallel_for<_SegReducePrefixKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT && _ONEDPL_LIBSYCL_PROGRAM_PRESENT
                __seg_reduce_prefix_kernel,
#endif
                sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                    auto __group = __item.get_group();
                    std::int64_t __group_id = __item.get_group(0);
                    std::size_t __global_id = __item.get_global_id(0);
                    std::size_t __local_id = __item.get_local_id(0);

                    std::size_t __start = __global_id * __vals_per_item;
                    std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);
                    std::size_t __item_segments = 0;

                    std::int64_t __wg_agg_idx = __group_id - 1;
                    __val_type __agg_collector = unseq_backend::__known_identity<_BinaryOperator, __val_type>;

                    bool __ag_exists = false;
                    // 3a. Check to see if an aggregate exists and compute that value in the first
                    // work item.
                    if (__group_id != 0)
                    {
                        __ag_exists = __start < __n;
                        // local reductions followed by a sweep
                        constexpr std::int32_t __vals_to_explore = 16;
                        bool __last_it = false;
                        __loc_seg_ends_acc[__local_id] = false;
                        __loc_partials_acc[__local_id] = unseq_backend::__known_identity<_BinaryOperator, __val_type>;
                        for (std::int32_t __i = __wg_agg_idx - __vals_to_explore * __local_id; !__last_it;
                             __i -= __wgroup_size * __vals_to_explore)
                        {
                            __val_type __local_collector = unseq_backend::__known_identity<_BinaryOperator, __val_type>;
                            // exploration phase
                            for (std::int32_t __j = __i;
                                 __j > __dpl_sycl::__maximum<std::int32_t>{}(-1L, __i - __vals_to_explore); --__j)
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
                            // serial aggregate collection and synchronization
                            if (__local_id == 0)
                            {
                                for (std::size_t __j = 0; __j < __wgroup_size; ++__j)
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

                        // Check to see if aggregates exist.
                        // The last group must always stay to write the final index
                        __ag_exists = __dpl_sycl::__any_of_group(__group, __ag_exists);
                        if (!__ag_exists && __group_id != __n_groups - 1)
                            return;
                    }
                    // 3b. count the segment ends
                    for (std::size_t __i = __start; __i < __end; ++__i)
                        if (__i == __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                            ++__item_segments;

                    std::size_t __prior_segs_in_wg = __dpl_sycl::__exclusive_scan_over_group(
                        __group, __item_segments, __dpl_sycl::__plus<decltype(__item_segments)>());

                    // 3c. Determine prior index
                    std::size_t __wg_num_prior_segs = __seg_ends_scan_acc[__group_id];

                    // 3d. Second pass over the keys, reidentifying end segments and applying work group
                    // aggregates if appropriate. Both the key and reduction value are written to the final output at the
                    // computed index
                    std::size_t __item_offset = 0;
                    for (std::size_t __i = __start; __i < __end; ++__i)
                    {
                        if (__i == __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                        {
                            std::size_t __idx = __wg_num_prior_segs + __prior_segs_in_wg + __item_offset;

                            // apply the aggregate if it is the first segment end in the workgroup only
                            if (__prior_segs_in_wg == 0 && __item_offset == 0 && __ag_exists)
                                __out_values[__idx] = __binary_op(__agg_collector, __out_values[__idx]);

                            ++__item_offset;
                            // the last item must write the last index's position to return
                            if (__i == __n - 1)
                                __end_idx_acc[0] = __idx;
                        }
                    }
                });
        })
        .wait();

    return __end_idx.get_host_access()[0] + 1;
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif
