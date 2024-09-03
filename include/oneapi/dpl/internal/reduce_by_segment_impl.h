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

#ifndef _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H
#define _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H

#include <cstdint>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <array>

#include "../pstl/iterator_impl.h"
#include "function.h"
#include "by_segment_extension_defs.h"
#include "../pstl/utils.h"

#if _ONEDPL_BACKEND_SYCL
#include "../pstl/utils_ranges.h"
#include "../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../pstl/ranges_defs.h"
#include "../pstl/glue_algorithm_ranges_impl.h"
#include "../pstl/hetero/dpcpp/sycl_traits.h" //SYCL traits specialization for some oneDPL types.
#include "scan_by_segment_impl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename Name>
class Reduce1;
template <typename Name>
class Reduce2;
template <typename Name>
class Reduce3;
template <typename Name>
class Reduce4;

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
::std::pair<OutputIterator1, OutputIterator2>
reduce_by_segment_impl(_Tag, Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                       OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred,
                       BinaryOperator binary_op)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key. This function's implementation is a derivative work
    // and responsible for the second copyright notice in this header.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    const auto n = ::std::distance(first1, last1);

    if (n <= 0)
        return ::std::make_pair(result1, result2);
    else if (n == 1)
    {
        *result1 = *first1;
        *result2 = *first2;
        return ::std::make_pair(result1 + 1, result2 + 1);
    }

    typedef uint64_t FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef uint64_t CountType;

    // buffer that is used to store a flag indicating if the associated key is not equal to
    // the next key, and thus its associated sum should be part of the final result
    oneapi::dpl::__par_backend::__buffer<Policy, FlagType> _mask(policy, n + 1);
    auto mask = _mask.get();
    mask[0] = 1;

    // instead of copying mask, use shifted sequence:
    mask[n] = 1;

    // Identify where the first key in a sequence of equivalent keys is located
    transform(policy, first1, last1 - 1, first1 + 1, _mask.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPred>(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    oneapi::dpl::__par_backend::__buffer<Policy, ValueType> _scanned_values(policy, n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    oneapi::dpl::__par_backend::__buffer<Policy, FlagType> _scanned_tail_flags(policy, n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    inclusive_scan(policy, make_zip_iterator(first2, _mask.get()), make_zip_iterator(first2, _mask.get()) + n,
                   make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
                   internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op));

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indices each segment sum should be written
    oneapi::dpl::exclusive_scan(policy, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                                ::std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    auto scanned_tail_flags = _scanned_tail_flags.get();
    auto scanned_values = _scanned_values.get();

    // number of unique segments
    CountType N = scanned_tail_flags[n - 1] + 1;

    // scatter the keys and accumulated values
    oneapi::dpl::for_each(::std::forward<Policy>(policy),
                          make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1),
                          make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1) + n,
                          internal::scatter_and_accumulate_fun<OutputIterator1, OutputIterator2>(result1, result2));

    // for example: result1 = {1, 2, 3, 4, 1, 3, 1, 3, 0}
    // for example: result2 = {1, 2, 3, 4, 2, 6, 2, 6, 0}

    return ::std::make_pair(result1 + N, result2 + N);
}

#if _ONEDPL_BACKEND_SYCL

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

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _Range4, typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__difference_t<_Range3>
__sycl_reduce_by_segment(__internal::__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __keys,
                         _Range2&& __values, _Range3&& __out_keys, _Range4&& __out_values,
                         _BinaryPredicate __binary_pred, _BinaryOperator __binary_op,
                         ::std::false_type /* has_known_identity */)
{
    return oneapi::dpl::experimental::ranges::reduce_by_segment(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys), ::std::forward<_Range2>(__values),
        ::std::forward<_Range3>(__out_keys), ::std::forward<_Range4>(__out_values), __binary_pred, __binary_op);
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _Range4, typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__difference_t<_Range3>
__sycl_reduce_by_segment(__internal::__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __keys,
                         _Range2&& __values, _Range3&& __out_keys, _Range4&& __out_values,
                         _BinaryPredicate __binary_pred, _BinaryOperator __binary_op,
                         ::std::true_type /* has_known_identity */)
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

    const ::std::size_t __n = __keys.size();

    constexpr ::std::uint16_t __vals_per_item =
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
        ::std::min({__wgroup_size,
                    oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_count_kernel),
                    oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_offset_kernel),
                    oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_wg_kernel),
                    oneapi::dpl::__internal::__kernel_work_group_size(__exec, __seg_reduce_prefix_kernel)});
#endif

    ::std::size_t __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __wgroup_size * __vals_per_item);

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
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__seg_reduce_count_kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_SegReduceCountKernel>(
            sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                                                                              __seg_reduce_count_kernel,
#endif
                                                                              sycl::nd_item<1> __item) {
                auto __group = __item.get_group();
                ::std::size_t __group_id = __item.get_group(0);
                ::std::size_t __local_id = __item.get_local_id(0);
                ::std::size_t __global_id = __item.get_global_id(0);

                ::std::size_t __start = __global_id * __vals_per_item;
                ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);
                ::std::size_t __item_segments = 0;

                // 1a. Work item scan to identify segment ends
                for (::std::size_t __i = __start; __i < __end; ++__i)
                    if (__n - 1 == __i || !__binary_pred(__keys[__i], __keys[__i + 1]))
                        ++__item_segments;

                // 1b. Work group reduction
                ::std::size_t __num_segs = __dpl_sycl::__reduce_over_group(
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
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__seg_reduce_offset_kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_SegReduceOffsetKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
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
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__seg_reduce_wg_kernel.get_kernel_bundle());
#endif
        __cgh.parallel_for<_SegReduceWgKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
            __seg_reduce_wg_kernel,
#endif
            sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                ::std::array<__val_type, __vals_per_item> __loc_partials;

                auto __group = __item.get_group();
                ::std::size_t __group_id = __item.get_group(0);
                ::std::size_t __local_id = __item.get_local_id(0);
                ::std::size_t __global_id = __item.get_global_id(0);

                // 2a. Lookup the number of prior segs
                auto __wg_num_prior_segs = __seg_ends_scan_acc[__group_id];

                // 2b. Perform a serial scan within the work item over assigned elements. Store partial
                // reductions in work group local memory.
                ::std::size_t __start = __global_id * __vals_per_item;
                ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);

                ::std::size_t __max_end = 0;
                ::std::size_t __item_segments = 0;
                auto __identity = unseq_backend::__known_identity<_BinaryOperator, __val_type>;

                __val_type __accumulator = __identity;
                for (::std::size_t __i = __start; __i < __end; ++__i)
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
                ::std::size_t __prior_segs_in_wg = __dpl_sycl::__exclusive_scan_over_group(
                    __group, __item_segments, __dpl_sycl::__plus<decltype(__item_segments)>());
                ::std::size_t __start_idx = __wg_num_prior_segs + __prior_segs_in_wg;

                // 2d. Find the greatest segment end less than the current index (inclusive)
                ::std::size_t __closest_seg_id = __dpl_sycl::__inclusive_scan_over_group(
                    __group, __max_end, __dpl_sycl::__maximum<decltype(__max_end)>());

                // __wg_segmented_scan is a derivative work and responsible for the third header copyright
                __val_type __carry_in = oneapi::dpl::internal::__wg_segmented_scan(
                    __item, __loc_acc, __local_id, __local_id - __closest_seg_id, __accumulator, __identity,
                    __binary_op, __wgroup_size);

                // 2e. Update local partial reductions in first segment and write to global memory.
                bool __apply_aggs = true;
                ::std::size_t __item_offset = 0;

                // first item in group does not have any work-group aggregates to apply
                if (__local_id == 0)
                {
                    __apply_aggs = false;
                    if (__global_id == 0 && __n > 0)
                    {
                        // first segment identifier is always the first key
                        __out_keys[0] = __keys[0];
                    }
                }

                // apply the aggregates and copy the locally stored values to destination buffer
                for (::std::size_t __i = __start; __i < __end; ++__i)
                {
                    if (__i == __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                    {
                        ::std::size_t __idx = __start_idx + __item_offset;
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
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
            __cgh.use_kernel_bundle(__seg_reduce_prefix_kernel.get_kernel_bundle());
#endif
            __cgh.parallel_for<_SegReducePrefixKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                __seg_reduce_prefix_kernel,
#endif
                sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                    auto __group = __item.get_group();
                    ::std::int64_t __group_id = __item.get_group(0);
                    ::std::size_t __global_id = __item.get_global_id(0);
                    ::std::size_t __local_id = __item.get_local_id(0);

                    ::std::size_t __start = __global_id * __vals_per_item;
                    ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);
                    ::std::size_t __item_segments = 0;

                    ::std::int64_t __wg_agg_idx = __group_id - 1;
                    __val_type __agg_collector = unseq_backend::__known_identity<_BinaryOperator, __val_type>;

                    bool __ag_exists = false;
                    // 3a. Check to see if an aggregate exists and compute that value in the first
                    // work item.
                    if (__group_id != 0)
                    {
                        __ag_exists = __start < __n;
                        // local reductions followed by a sweep
                        constexpr ::std::int32_t __vals_to_explore = 16;
                        bool __last_it = false;
                        __loc_seg_ends_acc[__local_id] = false;
                        __loc_partials_acc[__local_id] = unseq_backend::__known_identity<_BinaryOperator, __val_type>;
                        for (::std::int32_t __i = __wg_agg_idx - __vals_to_explore * __local_id; !__last_it;
                             __i -= __wgroup_size * __vals_to_explore)
                        {
                            __val_type __local_collector = unseq_backend::__known_identity<_BinaryOperator, __val_type>;
                            // exploration phase
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
                            // serial aggregate collection and synchronization
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

                        // Check to see if aggregates exist.
                        // The last group must always stay to write the final index
                        __ag_exists = __dpl_sycl::__any_of_group(__group, __ag_exists);
                        if (!__ag_exists && __group_id != __n_groups - 1)
                            return;
                    }
                    // 3b. count the segment ends
                    for (::std::size_t __i = __start; __i < __end; ++__i)
                        if (__i == __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                            ++__item_segments;

                    ::std::size_t __prior_segs_in_wg = __dpl_sycl::__exclusive_scan_over_group(
                        __group, __item_segments, __dpl_sycl::__plus<decltype(__item_segments)>());

                    // 3c. Determine prior index
                    ::std::size_t __wg_num_prior_segs = __seg_ends_scan_acc[__group_id];

                    // 3d. Second pass over the keys, reidentifying end segments and applying work group
                    // aggregates if appropriate. Both the key and reduction value are written to the final output at the
                    // computed index
                    ::std::size_t __item_offset = 0;
                    for (::std::size_t __i = __start; __i < __end; ++__i)
                    {
                        if (__i == __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                        {
                            ::std::size_t __idx = __wg_num_prior_segs + __prior_segs_in_wg + __item_offset;

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

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator1, typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
::std::pair<OutputIterator1, OutputIterator2>
reduce_by_segment_impl(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                       InputIterator1 last1, InputIterator2 first2, OutputIterator1 result1, OutputIterator2 result2,
                       BinaryPred binary_pred, BinaryOperator binary_op)
{
    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    using _CountType = ::std::uint64_t;

    namespace __bknd = __par_backend_hetero;

    const auto n = ::std::distance(first1, last1);

    if (n == 0)
        return ::std::make_pair(result1, result2);

    auto keep_keys = __ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto key_buf = keep_keys(first1, last1);
    auto keep_values = __ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(first2, first2 + n);
    auto keep_key_outputs = __ranges::__get_sycl_range<__bknd::access_mode::write, OutputIterator1>();
    auto key_output_buf = keep_key_outputs(result1, result1 + n);
    auto keep_value_outputs = __ranges::__get_sycl_range<__bknd::access_mode::write, OutputIterator2>();
    auto value_output_buf = keep_value_outputs(result2, result2 + n);

    using has_known_identity =
        typename unseq_backend::__has_known_identity<BinaryOperator,
                                                     typename ::std::iterator_traits<InputIterator2>::value_type>::type;

    // number of unique keys
    _CountType __n = __sycl_reduce_by_segment(
        __tag, ::std::forward<Policy>(policy), key_buf.all_view(), value_buf.all_view(), key_output_buf.all_view(),
        value_output_buf.all_view(), binary_pred, binary_op, has_known_identity{});

    return ::std::make_pair(result1 + __n, result2 + __n);
}
#endif
} // namespace internal

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                  OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, first1, first2, result1, result2);

    return internal::reduce_by_segment_impl(__dispatch_tag, ::std::forward<Policy>(policy), first1, last1, first2,
                                            result1, result2, binary_pred, binary_op);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2,
          typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_segment(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1,
                  OutputIt2 result2, BinaryPred binary_pred)
{
    typedef typename ::std::iterator_traits<InputIt2>::value_type T;

    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred,
                             ::std::plus<T>());
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_segment(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1,
                  OutputIt2 result2)
{
    typedef typename ::std::iterator_traits<InputIt1>::value_type T;

    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2,
                             ::std::equal_to<T>());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
              OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op)
{
    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred,
                             binary_op);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2,
          typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_key(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1, OutputIt2 result2,
              BinaryPred binary_pred)
{
    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_key(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1, OutputIt2 result2)
{
    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2);
}
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H
