// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename... _Name>
class __reduce_small_kernel;

template <typename... _Name>
class __reduce_mid_device_kernel;

template <typename... _Name>
class __reduce_mid_work_group_kernel;

template <typename... _Name>
class __reduce_kernel;

// Adjust number of sequential operations per work-item based on the vector size. Single elements are kept to
// improve performance of small arrays or remainder loops.
template <std::uint8_t _VecSize, typename _Size>
_Size
__adjust_iters_per_work_item(_Size __iters_per_work_item)
{
    if (__iters_per_work_item > 1)
        return oneapi::dpl::__internal::__dpl_ceiling_div(__iters_per_work_item, _VecSize) * _VecSize;
    return __iters_per_work_item;
}

// Single work group kernel that transforms and reduces __n elements to the single result.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _InitType, typename _AccLocal, typename _Res, typename... _Acc>
void
__work_group_reduce_kernel(const _NDItemId __item_id, const _Size __n, const _Size __iters_per_work_item,
                           const bool __is_full, _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern,
                           _InitType __init, const _AccLocal& __local_mem, _Res* __res_ptr, const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    const _Size __group_size = __item_id.get_local_range().size();
    oneapi::dpl::__internal::__lazy_ctor_storage<_Tp> __result;
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, __iters_per_work_item, /*global_offset*/ (_Size)0, __is_full,
                        /*__n_groups*/ (_Size)1, __result, __acc...);

    const _Size __n_items = __transform_pattern.output_size(__n, __group_size, __iters_per_work_item);
    // 2. Reduce within work group using local memory
    __result.__v = __reduce_pattern(__item_id, __n_items, __result.__v, __local_mem);
    if (__local_idx == 0)
    {
        __reduce_pattern.apply_init(__init, __result.__v);
        __res_ptr[0] = __result.__v;
    }
    __result.__destroy();
}

// Device kernel that transforms and reduces __n elements to the number of work groups preliminary results.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _AccLocal, typename _Tmp, typename... _Acc>
void
__device_reduce_kernel(const _NDItemId __item_id, const _Size __n, const _Size __iters_per_work_item,
                       const bool __is_full, const _Size __n_groups, _TransformPattern __transform_pattern,
                       _ReducePattern __reduce_pattern, const _AccLocal& __local_mem, _Tmp* __reduce_result_ptr,
                       const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_idx = __item_id.get_group(0);
    const _Size __group_size = __item_id.get_local_range().size();
    oneapi::dpl::__internal::__lazy_ctor_storage<_Tp> __result;
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, __iters_per_work_item, /*global_offset*/ (_Size)0, __is_full, __n_groups,
                        __result, __acc...);

    const _Size __n_items = __transform_pattern.output_size(__n, __group_size, __iters_per_work_item);
    // 2. Reduce within work group using local memory
    __result.__v = __reduce_pattern(__item_id, __n_items, __result.__v, __local_mem);
    if (__local_idx == 0)
        __reduce_result_ptr[__group_idx] = __result.__v;
    __result.__destroy();
}

//------------------------------------------------------------------------
// parallel_transform_reduce - async patterns
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Parallel_transform_reduce for a small arrays using a single work group.
// Transforms and reduces __work_group_size * __iters_per_work_item elements.
template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename _KernelName>
struct __parallel_transform_reduce_small_submitter;

template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename... _Name>
struct __parallel_transform_reduce_small_submitter<_Tp, _Commutative, _VecSize,
                                                   __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              typename... _Ranges>
    auto
    operator()(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, const _Size __n,
               const _Size __work_group_size, const _Size __iters_per_work_item, _ReduceOp __reduce_op,
               _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp, _Tp, _Commutative, _VecSize>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};
        const bool __is_full = __n == __work_group_size * __iters_per_work_item;

        using __result_and_scratch_storage_t = __result_and_scratch_storage<_ExecutionPolicy, _Tp>;
        __result_and_scratch_storage_t __scratch_container{__exec, 1, 0};

        sycl::event __reduce_event = __exec.queue().submit([&, __n](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            auto __res_acc = __scratch_container.__get_result_acc(__cgh);
            std::size_t __local_mem_size = __reduce_pattern.local_mem_req(__work_group_size);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__local_mem_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __res_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__res_acc);
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, __is_full,
                                                    __transform_pattern, __reduce_pattern, __init, __temp_local,
                                                    __res_ptr, __rngs...);
                });
        });

        return __future(__reduce_event, __scratch_container);
    }
}; // struct __parallel_transform_reduce_small_submitter

template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename _ExecutionPolicy, typename _Size,
          typename _ReduceOp, typename _TransformOp, typename _InitType, typename... _Ranges>
auto
__parallel_transform_reduce_small_impl(oneapi::dpl::__internal::__device_backend_tag __backend_tag,
                                       _ExecutionPolicy&& __exec, const _Size __n, const _Size __work_group_size,
                                       const _Size __iters_per_work_item, _ReduceOp __reduce_op,
                                       _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_small_kernel<_CustomName>>;

    return __parallel_transform_reduce_small_submitter<_Tp, _Commutative, _VecSize, _ReduceKernel>()(
        __backend_tag, std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size, __iters_per_work_item,
        __reduce_op, __transform_op, __init, std::forward<_Ranges>(__rngs)...);
}

// Submits the first kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses multiple work groups that each reduce __work_group_size * __iters_per_work_item items and store the preliminary
// results in __temp.
template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter;

template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename... _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter<_Tp, _Commutative, _VecSize,
                                                           __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp,
              typename _ExecutionPolicy2, typename... _Ranges>
    auto
    operator()(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, const _Size __n,
               const _Size __work_group_size, const _Size __iters_per_work_item, _ReduceOp __reduce_op,
               _TransformOp __transform_op,
               const __result_and_scratch_storage<_ExecutionPolicy2, _Tp>& __scratch_container,
               _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp, _Tp, _Commutative, _VecSize>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // number of buffer elements processed within workgroup
        const _Size __size_per_work_group = __iters_per_work_item * __work_group_size;
        const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        const bool __is_full = __n == __size_per_work_group * __n_groups;

        return __exec.queue().submit([&, __n](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            std::size_t __local_mem_size = __reduce_pattern.local_mem_req(__work_group_size);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__local_mem_size), __cgh);
            auto __temp_acc = __scratch_container.__get_scratch_acc(__cgh);
            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __temp_ptr =
                        __result_and_scratch_storage<_ExecutionPolicy2, _Tp>::__get_usm_or_buffer_accessor_ptr(
                            __temp_acc);
                    __device_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, __is_full, __n_groups,
                                                __transform_pattern, __reduce_pattern, __temp_local, __temp_ptr,
                                                __rngs...);
                });
        });
    }
}; // struct __parallel_transform_reduce_device_kernel_submitter

// Submits the second kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses a single work groups to reduce __n preliminary results stored in __temp and returns a future object with the
// result buffer.
template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter;

template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename... _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter<_Tp, _Commutative, _VecSize,
                                                               __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _InitType,
              typename _ExecutionPolicy2>
    auto
    operator()(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, sycl::event& __reduce_event,
               const _Size __n, const _Size __work_group_size, const _Size __iters_per_work_item, _ReduceOp __reduce_op,
               _InitType __init, const __result_and_scratch_storage<_ExecutionPolicy2, _Tp>& __scratch_container) const
    {
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _NoOpFunctor, _Tp, _Commutative, _VecSize>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        const bool __is_full = __n == __work_group_size * __iters_per_work_item;

        using __result_and_scratch_storage_t = __result_and_scratch_storage<_ExecutionPolicy2, _Tp>;

        __reduce_event = __exec.queue().submit([&, __n](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            auto __temp_acc = __scratch_container.__get_scratch_acc(__cgh);
            auto __res_acc = __scratch_container.__get_result_acc(__cgh);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);

            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __temp_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                    auto __res_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__res_acc, __n);
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __iters_per_work_item, __is_full,
                                                    __transform_pattern, __reduce_pattern, __init, __temp_local,
                                                    __res_ptr, __temp_ptr);
                });
        });

        return __future(__reduce_event, __scratch_container);
    }
}; // struct __parallel_transform_reduce_work_group_kernel_submitter

template <typename _Tp, typename _Commutative, std::uint8_t _VecSize, typename _ExecutionPolicy, typename _Size,
          typename _ReduceOp, typename _TransformOp, typename _InitType, typename... _Ranges>
auto
__parallel_transform_reduce_mid_impl(oneapi::dpl::__internal::__device_backend_tag __backend_tag,
                                     _ExecutionPolicy&& __exec, const _Size __n, const _Size __work_group_size,
                                     const _Size __iters_per_work_item_device_kernel,
                                     const _Size __iters_per_work_item_work_group_kernel, _ReduceOp __reduce_op,
                                     _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceDeviceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_mid_device_kernel<_CustomName>>;
    using _ReduceWorkGroupKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_mid_work_group_kernel<_CustomName>>;

    // number of buffer elements processed within workgroup
    const _Size __size_per_work_group = __iters_per_work_item_device_kernel * __work_group_size;
    const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
    __result_and_scratch_storage<_ExecutionPolicy, _Tp> __scratch_container{__exec, 1, __n_groups};

    sycl::event __reduce_event =
        __parallel_transform_reduce_device_kernel_submitter<_Tp, _Commutative, _VecSize, _ReduceDeviceKernel>()(
            __backend_tag, __exec, __n, __work_group_size, __iters_per_work_item_device_kernel, __reduce_op,
            __transform_op, __scratch_container, std::forward<_Ranges>(__rngs)...);

    // __n_groups preliminary results from the device kernel.
    return __parallel_transform_reduce_work_group_kernel_submitter<_Tp, _Commutative, _VecSize,
                                                                   _ReduceWorkGroupKernel>()(
        __backend_tag, std::forward<_ExecutionPolicy>(__exec), __reduce_event, __n_groups, __work_group_size,
        __iters_per_work_item_work_group_kernel, __reduce_op, __init, __scratch_container);
}

// General implementation using a tree reduction
template <typename _Tp, typename _Commutative, std::uint8_t _VecSize>
struct __parallel_transform_reduce_impl
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              typename... _Ranges>
    static auto
    submit(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Size __n, _Size __work_group_size,
           const _Size __iters_per_work_item, _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init,
           _Ranges&&... __rngs)
    {
        using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __reduce_kernel, _CustomName, _ReduceOp, _TransformOp, _NoOpFunctor, _Ranges...>;

        auto __transform_pattern1 =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _TransformOp, _Tp, _Commutative, _VecSize>{
                __reduce_op, __transform_op};
        auto __transform_pattern2 =
            unseq_backend::transform_reduce<_ExecutionPolicy, _ReduceOp, _NoOpFunctor, _Tp, _Commutative, _VecSize>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

#if _ONEDPL_COMPILE_KERNEL
        auto __kernel = __internal::__kernel_compiler<_ReduceKernel>::__compile(__exec);
        _Size __adjusted_work_group_size = oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel);
        __work_group_size = std::min(__work_group_size, __adjusted_work_group_size);
#endif

        const _Size __size_per_work_group =
            __iters_per_work_item * __work_group_size; // number of buffer elements processed within workgroup
        _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);

        // Create temporary global buffers to store temporary values
        const std::size_t __n_scratch = 2 * __n_groups;
        using __result_and_scratch_storage_t = __result_and_scratch_storage<_ExecutionPolicy, _Tp>;
        __result_and_scratch_storage_t __scratch_container{__exec, 1, __n_scratch};

        // __is_first == true. Reduce over each work_group
        // __is_first == false. Reduce between work groups
        bool __is_first = true;

        // For memory utilization it's better to use one big buffer instead of two small because size of the buffer is
        // close to a few MB
        _Size __offset_1 = 0;
        _Size __offset_2 = __n_groups;

        sycl::event __reduce_event;
        do
        {
            __reduce_event = __exec.queue().submit([&, __is_first, __offset_1, __offset_2, __n,
                                                    __n_groups](sycl::handler& __cgh) {
                __cgh.depends_on(__reduce_event);
                auto __temp_acc = __scratch_container.__get_scratch_acc(__cgh);
                auto __res_acc = __scratch_container.__get_result_acc(__cgh);

                // get an access to data under SYCL buffer
                oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);
                std::size_t __local_mem_size = __reduce_pattern.local_mem_req(__work_group_size);
                __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__local_mem_size), __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
                __cgh.use_kernel_bundle(__kernel.get_kernel_bundle());
#endif
                __cgh.parallel_for<_ReduceKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                    __kernel,
#endif
                    sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item_id) {
                        auto __temp_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                        auto __res_ptr =
                            __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__res_acc, 2 * __n_groups);
                        auto __local_idx = __item_id.get_local_id(0);
                        auto __group_idx = __item_id.get_group(0);
                        // 1. Initialization (transform part). Fill local memory
                        _Size __n_items;
                        const bool __is_full = __n == __size_per_work_group * __n_groups;
                        oneapi::dpl::__internal::__lazy_ctor_storage<_Tp> __result;
                        if (__is_first)
                        {
                            __transform_pattern1(__item_id, __n, __iters_per_work_item, /*global_offset*/ (_Size)0,
                                                 __is_full, __n_groups, __result, __rngs...);
                            __n_items = __transform_pattern1.output_size(__n, __work_group_size, __iters_per_work_item);
                        }
                        else
                        {
                            __transform_pattern2(__item_id, __n, __iters_per_work_item, __offset_2, __is_full,
                                                 __n_groups, __result, __temp_ptr);
                            __n_items = __transform_pattern2.output_size(__n, __work_group_size, __iters_per_work_item);
                        }
                        // 2. Reduce within work group using local memory
                        __result.__v = __reduce_pattern(__item_id, __n_items, __result.__v, __temp_local);
                        if (__local_idx == 0)
                        {
                            // final reduction
                            if (__n_groups == 1)
                            {
                                __reduce_pattern.apply_init(__init, __result.__v);
                                __res_ptr[0] = __result.__v;
                            }

                            __temp_ptr[__offset_1 + __group_idx] = __result.__v;
                        }
                        __result.__destroy();
                    });
            });
            __is_first = false;
            std::swap(__offset_1, __offset_2);
            __n = __n_groups;
            __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        } while (__n > 1);

        return __future(__reduce_event, __scratch_container);
    }
}; // struct __parallel_transform_reduce_impl

// General version of parallel_transform_reduce.
// The binary operator must be associative but commutativity is only required by some of the algorithms using
// __parallel_transform_reduce. This is provided by the _Commutative parameter. Commutative algorithms use
// coalesced loads from global memory if beneficial. Non-commutative algorithms processes elements in order.
//
// Each work item transforms and reduces __iters_per_work_item elements from global memory and stores the result in SLM.
// Each work group of size __work_group_size reduces the preliminary results of each work item in a group reduction
// using SLM.
// A single-work group implementation is used for small arrays.
// Mid-sized arrays use two tree reductions with independent __iters_per_work_item.
// Big arrays are processed with a recursive tree reduction. __work_group_size * __iters_per_work_item elements are
// reduced in each step.
template <typename _Tp, typename _Commutative, typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp,
          typename _InitType, typename... _Ranges>
auto
__parallel_transform_reduce(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                            _ReduceOp __reduce_op, _TransformOp __transform_op, _InitType __init, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);
    using _Size = decltype(__n);

    // Empirically found tuning parameters for typical devices.
    constexpr _Size __max_iters_per_work_item = 32;
    constexpr std::size_t __max_work_group_size = 256;
    static_assert(__max_work_group_size * __max_iters_per_work_item <= std::numeric_limits<std::uint16_t>::max(),
                  "Out of 16-bit addressing range");
    constexpr std::uint8_t __vector_size = 4;
    constexpr std::uint32_t __oversubscription = 2;

    // Get the work group size adjusted to the local memory limit.
    // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
    // TODO: find a way to generalize getting of reliable work-group size.
    std::size_t __work_group_size =
        oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, static_cast<std::size_t>(sizeof(_Tp) * 2));

    // Limit work-group size to __max_work_group_size for performance on GPUs. Empirically tested.
    __work_group_size = std::min(__work_group_size, __max_work_group_size);
    const _Size __max_elements_per_wg = __work_group_size * __max_iters_per_work_item;

    // Use single work group implementation if less than __max_iters_per_work_item elements per work-group.
    // We can use 16-bit addressing since we have at most __max_work_group_size * __max_iters_per_work_item elements.
    // Update this in case of changing __max_work_group_size or __max_iters_per_work_item.
    if (__n <= __max_elements_per_wg)
    {
        const auto __n_short = static_cast<std::uint16_t>(__n);
        const auto __work_group_size_short = static_cast<std::uint16_t>(__work_group_size);
        std::uint16_t __iters_per_work_item = oneapi::dpl::__internal::__dpl_ceiling_div(__n_short, __work_group_size);
        __iters_per_work_item = __adjust_iters_per_work_item<__vector_size>(__iters_per_work_item);
        return __parallel_transform_reduce_small_impl<_Tp, _Commutative, __vector_size>(
            __backend_tag, std::forward<_ExecutionPolicy>(__exec), __n_short, __work_group_size_short,
            __iters_per_work_item, __reduce_op, __transform_op, __init, std::forward<_Ranges>(__rngs)...);
    }
    // Use two-step tree reduction.
    // First step reduces __work_group_size * __iters_per_work_item_device_kernel elements.
    // Second step reduces __work_group_size * __iters_per_work_item_work_group_kernel elements.
    // We can use 32-bit addressing since we have at most (__max_work_group_size * __max_iters_per_work_item) ^ 2
    // elements.
    // Update this in case of changing __max_work_group_size or __max_iters_per_work_item.
    if (__n <= __max_elements_per_wg * __max_elements_per_wg)
    {
        const auto __n_short = static_cast<std::uint32_t>(__n);
        const auto __work_group_size_short = static_cast<std::uint32_t>(__work_group_size);
        // Fully-utilize the device by running a work-group per compute unit.
        // Add a factor more work-groups than compute units to fully utilizes the device and hide latencies.
        const std::uint32_t __max_cu = oneapi::dpl::__internal::__max_compute_units(__exec);
        std::uint32_t __n_groups = __max_cu * __oversubscription;
        std::uint32_t __iters_per_work_item_device_kernel =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n_short, __n_groups * __work_group_size_short);
        __iters_per_work_item_device_kernel =
            __adjust_iters_per_work_item<__vector_size>(__iters_per_work_item_device_kernel);

        // Lower the number of iterations to not exceed the empirically found limit.
        // This increases the number of work-groups up to the limit of work-group size times __max_iters_per_work_item.
        if (__iters_per_work_item_device_kernel > __max_iters_per_work_item)
        {
            __iters_per_work_item_device_kernel = __max_iters_per_work_item;
            __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n_short, __max_elements_per_wg);
        }
        std::uint32_t __iters_per_work_item_work_group_kernel =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n_groups, __work_group_size_short);
        __iters_per_work_item_work_group_kernel =
            __adjust_iters_per_work_item<__vector_size>(__iters_per_work_item_work_group_kernel);
        return __parallel_transform_reduce_mid_impl<_Tp, _Commutative, __vector_size>(
            __backend_tag, std::forward<_ExecutionPolicy>(__exec), __n_short, __work_group_size_short,
            __iters_per_work_item_device_kernel, __iters_per_work_item_work_group_kernel, __reduce_op, __transform_op,
            __init, std::forward<_Ranges>(__rngs)...);
    }
    // Otherwise use a recursive tree reduction with __max_iters_per_work_item __iters_per_work_item.
    const auto __work_group_size_long = static_cast<_Size>(__work_group_size);
    return __parallel_transform_reduce_impl<_Tp, _Commutative, __vector_size>::submit(
        __backend_tag, std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size_long, __max_iters_per_work_item,
        __reduce_op, __transform_op, __init, std::forward<_Ranges>(__rngs)...);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
