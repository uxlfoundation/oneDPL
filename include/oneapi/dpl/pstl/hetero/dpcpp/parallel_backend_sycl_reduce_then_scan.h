// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_then_scan.h ---------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__exclusive_sub_group_masked_scan(const __dpl_sycl::__sub_group& __sub_group, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    _LazyValueType __old_init;
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        if (__sub_group_local_id == 0)
            __old_init.__setup(__init_and_carry.__v);
        __init_and_carry.__v = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    }

    __value = sycl::shift_group_right(__sub_group, __value, 1);
    if constexpr (__init_present)
    {
        if (__sub_group_local_id == 0)
        {
            __value = __old_init.__v;
            __old_init.__destroy();
        }
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__inclusive_sub_group_masked_scan(const __dpl_sycl::__sub_group& __sub_group, _MaskOp __mask_fn,
                                  _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                                  _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        _ValueType __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        __init_and_carry.__v = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _MaskOp,
          typename _InitBroadcastId, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__sub_group_masked_scan(const __dpl_sycl::__sub_group& __sub_group, _MaskOp __mask_fn,
                        _InitBroadcastId __init_broadcast_id, _ValueType& __value, _BinaryOp __binary_op,
                        _LazyValueType& __init_and_carry)
{
    if constexpr (__is_inclusive)
    {
        __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
    else
    {
        __exclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType>
void
__sub_group_scan(const __dpl_sycl::__sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                 _LazyValueType& __init_and_carry)
{
    auto __mask_fn = [](auto __sub_group_local_id, auto __offset) { return __sub_group_local_id >= __offset; };
    constexpr std::uint8_t __init_broadcast_id = __sub_group_size - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry);
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _BinaryOp,
          typename _ValueType, typename _LazyValueType, typename _SizeType>
void
__sub_group_scan_partial(const __dpl_sycl::__sub_group& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                         _LazyValueType& __init_and_carry, _SizeType __elements_to_process)
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    std::uint8_t __init_broadcast_id = __elements_to_process - 1;
    __sub_group_masked_scan<__sub_group_size, __is_inclusive, __init_present>(
        __sub_group, __mask_fn, __init_broadcast_id, __value, __binary_op, __init_and_carry);
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, bool __capture_output,
          std::uint16_t __max_inputs_per_item, typename _GenInput, typename _ScanInputTransform, typename _BinaryOp,
          typename _WriteOp, typename _LazyValueType, typename _InRng, typename _OutRng>
void
__scan_through_elements_helper(const __dpl_sycl::__sub_group& __sub_group, _GenInput __gen_input,
                               _ScanInputTransform __scan_input_transform, _BinaryOp __binary_op, _WriteOp __write_op,
                               _LazyValueType& __sub_group_carry, const _InRng& __in_rng, _OutRng& __out_rng,
                               std::size_t __start_id, std::size_t __n, std::uint32_t __iters_per_item,
                               std::size_t __subgroup_start_id, std::uint32_t __sub_group_id,
                               std::uint32_t __active_subgroups)
{
    using _GenInputType = std::invoke_result_t<_GenInput, _InRng, std::size_t>;

    bool __is_full_block = (__iters_per_item == __max_inputs_per_item);
    bool __is_full_thread = __subgroup_start_id + __iters_per_item * __sub_group_size <= __n;
    if (__is_full_thread)
    {
        _GenInputType __v = __gen_input(__in_rng, __start_id);
        __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(__sub_group, __scan_input_transform(__v),
                                                                           __binary_op, __sub_group_carry);
        if constexpr (__capture_output)
        {
            __write_op(__out_rng, __start_id, __v);
        }

        if (__is_full_block)
        {
            // For full block and full thread, we can unroll the loop
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __j = 1; __j < __max_inputs_per_item; __j++)
            {
                __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size);
                __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __write_op(__out_rng, __start_id + __j * __sub_group_size, __v);
                }
            }
        }
        else
        {
            // For full thread but not full block, we can't unroll the loop, but we
            // can proceed without special casing for partial subgroups.
            for (std::uint32_t __j = 1; __j < __iters_per_item; __j++)
            {
                __v = __gen_input(__in_rng, __start_id + __j * __sub_group_size);
                __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __write_op(__out_rng, __start_id + __j * __sub_group_size, __v);
                }
            }
        }
    }
    else
    {
        // For partial thread, we need to handle the partial subgroup at the end of the range
        if (__sub_group_id < __active_subgroups)
        {
            std::uint32_t __iters =
                oneapi::dpl::__internal::__dpl_ceiling_div(__n - __subgroup_start_id, __sub_group_size);

            if (__iters == 1)
            {
                std::size_t __local_id = (__start_id < __n) ? __start_id : __n - 1;
                _GenInputType __v = __gen_input(__in_rng, __local_id);
                __sub_group_scan_partial<__sub_group_size, __is_inclusive, __init_present>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                    __n - __subgroup_start_id);
                if constexpr (__capture_output)
                {
                    if (__start_id < __n)
                        __write_op(__out_rng, __start_id, __v);
                }
            }
            else
            {
                _GenInputType __v = __gen_input(__in_rng, __start_id);
                __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __write_op(__out_rng, __start_id, __v);
                }

                for (std::uint32_t __j = 1; __j < __iters - 1; __j++)
                {
                    std::size_t __local_id = __start_id + __j * __sub_group_size;
                    __v = __gen_input(__in_rng, __local_id);
                    __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                        __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry);
                    if constexpr (__capture_output)
                    {
                        __write_op(__out_rng, __local_id, __v);
                    }
                }

                std::size_t __offset = __start_id + (__iters - 1) * __sub_group_size;
                std::size_t __local_id = (__offset < __n) ? __offset : __n - 1;
                __v = __gen_input(__in_rng, __local_id);
                __sub_group_scan_partial<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __scan_input_transform(__v), __binary_op, __sub_group_carry,
                    __n - (__subgroup_start_id + (__iters - 1) * __sub_group_size));
                if constexpr (__capture_output)
                {
                    if (__offset < __n)
                        __write_op(__out_rng, __offset, __v);
                }
            }
        }
    }
}

template <typename... _Name>
class __reduce_then_scan_reduce_kernel;

template <typename... _Name>
class __reduce_then_scan_scan_kernel;

template <std::uint8_t __sub_group_size, std::uint16_t __max_inputs_per_item, bool __is_inclusive,
          bool __is_unique_pattern_v, typename _GenReduceInput, typename _ReduceOp, typename _InitType,
          typename _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter;

template <std::uint8_t __sub_group_size, std::uint16_t __max_inputs_per_item, bool __is_inclusive,
          bool __is_unique_pattern_v, typename _GenReduceInput, typename _ReduceOp, typename _InitType,
          typename... _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter<__sub_group_size, __max_inputs_per_item, __is_inclusive,
                                                    __is_unique_pattern_v, _GenReduceInput, _ReduceOp, _InitType,
                                                    __internal::__optional_kernel_name<_KernelName...>>
{
    // Step 1 - SubGroupReduce is expected to perform sub-group reductions to global memory
    // input buffer
    template <typename _ExecutionPolicy, typename _InRng, typename _TmpStorageAcc>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, const sycl::nd_range<1> __nd_range, _InRng&& __in_rng,
               _TmpStorageAcc& __scratch_container, const sycl::event& __prior_event,
               const std::uint32_t __inputs_per_sub_group, const std::uint32_t __inputs_per_item,
               const std::size_t __block_num) const
    {
        using _InitValueType = typename _InitType::__value_type;
        return __exec.queue().submit([&, this](sycl::handler& __cgh) {
            __dpl_sycl::__local_accessor<_InitValueType> __sub_group_partials(__num_sub_groups_local, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng);
            auto __temp_acc = __scratch_container.__get_scratch_acc(__cgh);
            __cgh.parallel_for<_KernelName...>(
                    __nd_range, [=, *this](sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(__sub_group_size)]] {
                _InitValueType* __temp_ptr = _TmpStorageAcc::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                std::size_t __group_id = __ndi.get_group(0);
                __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
                std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
                std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

                oneapi::dpl::__internal::__lazy_ctor_storage<_InitValueType> __sub_group_carry;
                std::size_t __group_start_id =
                    (__block_num * __max_block_size) + (__group_id * __inputs_per_sub_group * __num_sub_groups_local);
                if constexpr (__is_unique_pattern_v)
                {
                    // for unique patterns, the first element is always copied to the output, so we need to skip it
                    __group_start_id += 1;
                }
                std::size_t __max_inputs_in_group = __inputs_per_sub_group * __num_sub_groups_local;
                std::uint32_t __inputs_in_group = std::min(__n - __group_start_id, __max_inputs_in_group);
                std::uint32_t __active_subgroups =
                    oneapi::dpl::__internal::__dpl_ceiling_div(__inputs_in_group, __inputs_per_sub_group);
                std::size_t __subgroup_start_id = __group_start_id + (__sub_group_id * __inputs_per_sub_group);

                std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

                if (__sub_group_id < __active_subgroups)
                {
                    // adjust for lane-id
                    // compute sub-group local prefix on T0..63, K samples/T, send to accumulator kernel
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/false,
                                                   /*__capture_output=*/false, __max_inputs_per_item>(
                        __sub_group, __gen_reduce_input, oneapi::dpl::__internal::__no_op{}, __reduce_op, nullptr,
                        __sub_group_carry, __in_rng, /*unused*/ __in_rng, __start_id, __n, __inputs_per_item,
                        __subgroup_start_id, __sub_group_id, __active_subgroups);
                    if (__sub_group_local_id == 0)
                        __sub_group_partials[__sub_group_id] = __sub_group_carry.__v;
                    __sub_group_carry.__destroy();
                }
                __dpl_sycl::__group_barrier(__ndi);

                // compute sub-group local prefix sums on (T0..63) carries
                // and store to scratch space at the end of dst; next
                // accumulator kernel takes M thread carries from scratch
                // to compute a prefix sum on global carries
                if (__sub_group_id == 0)
                {
                    __start_id = (__group_id * __num_sub_groups_local);
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    if (__iters == 1)
                    {
                        // fill with unused dummy values to avoid overruning input
                        std::uint32_t __load_id = std::min(std::uint32_t{__sub_group_local_id}, __active_subgroups - 1);
                        _InitValueType __v = __sub_group_partials[__load_id];
                        __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                            __sub_group, __v, __reduce_op, __sub_group_carry, __active_subgroups);
                        if (__sub_group_local_id < __active_subgroups)
                            __temp_ptr[__start_id + __sub_group_local_id] = __v;
                    }
                    else
                    {
                        std::uint32_t __reduction_scan_id = __sub_group_local_id;
                        // need to pull out first iteration tp avoid identity
                        _InitValueType __v = __sub_group_partials[__reduction_scan_id];
                        __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                            __sub_group, __v, __reduce_op, __sub_group_carry);
                        __temp_ptr[__start_id + __reduction_scan_id] = __v;
                        __reduction_scan_id += __sub_group_size;

                        for (std::uint32_t __i = 1; __i < __iters - 1; __i++)
                        {
                            __v = __sub_group_partials[__reduction_scan_id];
                            __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                                __sub_group, __v, __reduce_op, __sub_group_carry);
                            __temp_ptr[__start_id + __reduction_scan_id] = __v;
                            __reduction_scan_id += __sub_group_size;
                        }
                        // If we are past the input range, then the previous value of v is passed to the sub-group scan.
                        // It does not affect the result as our sub_group_scan will use a mask to only process in-range elements.

                        // fill with unused dummy values to avoid overruning input
                        std::uint32_t __load_id = std::min(__reduction_scan_id, __num_sub_groups_local - 1);

                        __v = __sub_group_partials[__load_id];
                        __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                            __sub_group, __v, __reduce_op, __sub_group_carry,
                            __active_subgroups - ((__iters - 1) * __sub_group_size));
                        if (__reduction_scan_id < __num_sub_groups_local)
                            __temp_ptr[__start_id + __reduction_scan_id] = __v;
                    }

                    __sub_group_carry.__destroy();
                }
            });
        });
    }

    // Constant parameters throughout all blocks
    const std::uint32_t __max_block_size;
    const std::uint32_t __num_sub_groups_local;
    const std::uint32_t __num_sub_groups_global;
    const std::uint32_t __num_work_items;
    const std::size_t __n;

    const _GenReduceInput __gen_reduce_input;
    const _ReduceOp __reduce_op;
    _InitType __init;
};

template <std::uint8_t __sub_group_size, std::uint16_t __max_inputs_per_item, bool __is_inclusive,
          bool __is_unique_pattern_v, typename _ReduceOp, typename _GenScanInput, typename _ScanInputTransform,
          typename _WriteOp, typename _InitType, typename _KernelName>
struct __parallel_reduce_then_scan_scan_submitter;

template <std::uint8_t __sub_group_size, std::uint16_t __max_inputs_per_item, bool __is_inclusive,
          bool __is_unique_pattern_v, typename _ReduceOp, typename _GenScanInput, typename _ScanInputTransform,
          typename _WriteOp, typename _InitType, typename... _KernelName>
struct __parallel_reduce_then_scan_scan_submitter<
    __sub_group_size, __max_inputs_per_item, __is_inclusive, __is_unique_pattern_v, _ReduceOp, _GenScanInput,
    _ScanInputTransform, _WriteOp, _InitType, __internal::__optional_kernel_name<_KernelName...>>
{
    using _InitValueType = typename _InitType::__value_type;

    _InitValueType
    __get_block_carry_in(const std::size_t __block_num, _InitValueType* __tmp_ptr) const
    {
        return __tmp_ptr[__num_sub_groups_global + (__block_num % 2)];
    }

    template <typename _ValueType>
    void
    __set_block_carry_out(const std::size_t __block_num, _InitValueType* __tmp_ptr,
                          const _ValueType __block_carry_out) const
    {
        __tmp_ptr[__num_sub_groups_global + 1 - (__block_num % 2)] = __block_carry_out;
    }

    template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _TmpStorageAcc>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, const sycl::nd_range<1> __nd_range, _InRng&& __in_rng, _OutRng&& __out_rng,
               _TmpStorageAcc& __scratch_container, const sycl::event& __prior_event,
               const std::uint32_t __inputs_per_sub_group, const std::uint32_t __inputs_per_item,
               const std::size_t __block_num) const
    {
        std::uint32_t __inputs_in_block = std::min(__n - __block_num * __max_block_size, std::size_t{__max_block_size});
        std::uint32_t __active_groups = oneapi::dpl::__internal::__dpl_ceiling_div(
            __inputs_in_block, __inputs_per_sub_group * __num_sub_groups_local);
        return __exec.queue().submit([&, this](sycl::handler& __cgh) {
            // We need __num_sub_groups_local + 1 temporary SLM locations to store intermediate results:
            //   __num_sub_groups_local for each sub-group partial from the reduce kernel +
            //   1 element for the accumulated block-local carry-in from previous groups in the block
            __dpl_sycl::__local_accessor<_InitValueType> __sub_group_partials(__num_sub_groups_local + 1, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng, __out_rng);
            auto __temp_acc = __scratch_container.__get_scratch_acc(__cgh);
            auto __res_acc = __scratch_container.__get_result_acc(__cgh);

            __cgh.parallel_for<_KernelName...>(
                    __nd_range, [=, *this] (sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(__sub_group_size)]] {
                _InitValueType* __tmp_ptr = _TmpStorageAcc::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                _InitValueType* __res_ptr =
                    _TmpStorageAcc::__get_usm_or_buffer_accessor_ptr(__res_acc, __num_sub_groups_global + 2);
                std::uint32_t __group_id = __ndi.get_group(0);
                __dpl_sycl::__sub_group __sub_group = __ndi.get_sub_group();
                std::uint32_t __sub_group_id = __sub_group.get_group_linear_id();
                std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();

                std::size_t __group_start_id =
                    (__block_num * __max_block_size) + (__group_id * __inputs_per_sub_group * __num_sub_groups_local);
                if constexpr (__is_unique_pattern_v)
                {
                    // for unique patterns, the first element is always copied to the output, so we need to skip it
                    __group_start_id += 1;
                }

                std::size_t __max_inputs_in_group = __inputs_per_sub_group * __num_sub_groups_local;
                std::uint32_t __inputs_in_group = std::min(__n - __group_start_id, __max_inputs_in_group);
                std::uint32_t __active_subgroups =
                    oneapi::dpl::__internal::__dpl_ceiling_div(__inputs_in_group, __inputs_per_sub_group);
                oneapi::dpl::__internal::__lazy_ctor_storage<_InitValueType> __carry_last;

                // propagate carry in from previous block
                oneapi::dpl::__internal::__lazy_ctor_storage<_InitValueType> __sub_group_carry;

                // on the first sub-group in a work-group (assuming S subgroups in a work-group):
                // 1. load S sub-group local carry prefix sums (T0..TS-1) to SLM
                // 2. load 32, 64, 96, etc. TS-1 work-group carry-outs (32 for WG num<32, 64 for WG num<64, etc.),
                //    and then compute the prefix sum to generate global carry out
                //    for each WG, i.e., prefix sum on TS-1 carries over all WG.
                // 3. on each WG select the adjacent neighboring WG carry in
                // 4. on each WG add the global carry-in to S sub-group local prefix sums to
                //    get a T-local global carry in
                // 5. recompute T-local prefix values, add the T-local global carries,
                //    and then write back the final values to memory
                if (__sub_group_id == 0)
                {
                    // step 1) load to SLM the WG-local S prefix sums
                    //         on WG T-local carries
                    //            0: T0 carry, 1: T0 + T1 carry, 2: T0 + T1 + T2 carry, ...
                    //           S: sum(T0 carry...TS carry)
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    std::size_t __subgroups_before_my_group = __group_id * __num_sub_groups_local;
                    std::uint32_t __load_reduction_id = __sub_group_local_id;
                    std::uint8_t __i = 0;
                    for (; __i < __iters - 1; __i++)
                    {
                        __sub_group_partials[__load_reduction_id] =
                            __tmp_ptr[__subgroups_before_my_group + __load_reduction_id];
                        __load_reduction_id += __sub_group_size;
                    }
                    if (__load_reduction_id < __active_subgroups)
                    {
                        __sub_group_partials[__load_reduction_id] =
                            __tmp_ptr[__subgroups_before_my_group + __load_reduction_id];
                    }

                    // step 2) load 32, 64, 96, etc. work-group carry outs on every work-group; then
                    //         compute the prefix in a sub-group to get global work-group carries
                    //         memory accesses: gather(63, 127, 191, 255, ...)
                    std::uint32_t __offset = __num_sub_groups_local - 1;
                    // only need 32 carries for WGs0..WG32, 64 for WGs32..WGs64, etc.
                    if (__group_id > 0)
                    {
                        // only need the last element from each scan of num_sub_groups_local subgroup reductions
                        const std::size_t __elements_to_process = __subgroups_before_my_group / __num_sub_groups_local;
                        const std::size_t __pre_carry_iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__elements_to_process, __sub_group_size);
                        if (__pre_carry_iters == 1)
                        {
                            // single partial scan
                            std::size_t __proposed_id = __num_sub_groups_local * __sub_group_local_id + __offset;
                            std::size_t __remaining_elements = __elements_to_process;
                            std::size_t __reduction_id = (__proposed_id < __subgroups_before_my_group)
                                                             ? __proposed_id
                                                             : __subgroups_before_my_group - 1;
                            _InitValueType __value = __tmp_ptr[__reduction_id];
                            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true,
                                                     /*__init_present=*/false>(__sub_group, __value, __reduce_op,
                                                                               __carry_last, __remaining_elements);
                        }
                        else
                        {
                            // multiple iterations
                            // first 1 full
                            std::uint32_t __reduction_id = __num_sub_groups_local * __sub_group_local_id + __offset;
                            std::uint32_t __reduction_id_increment = __num_sub_groups_local * __sub_group_size;
                            _InitValueType __value = __tmp_ptr[__reduction_id];
                            __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                                __sub_group, __value, __reduce_op, __carry_last);
                            __reduction_id += __reduction_id_increment;
                            // then some number of full iterations
                            for (std::uint32_t __i = 1; __i < __pre_carry_iters - 1; __i++)
                            {
                                __value = __tmp_ptr[__reduction_id];
                                __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                                    __sub_group, __value, __reduce_op, __carry_last);
                                __reduction_id += __reduction_id_increment;
                            }

                            // final partial iteration

                            std::size_t __remaining_elements =
                                __elements_to_process - ((__pre_carry_iters - 1) * __sub_group_size);
                            // fill with unused dummy values to avoid overruning input
                            std::size_t __final_reduction_id =
                                std::min(std::size_t{__reduction_id}, __subgroups_before_my_group - 1);
                            __value = __tmp_ptr[__final_reduction_id];
                            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true,
                                                     /*__init_present=*/true>(__sub_group, __value, __reduce_op,
                                                                              __carry_last, __remaining_elements);
                        }

                        // steps 3+4) load global carry in from neighbor work-group
                        //            and apply to local sub-group prefix carries
                        std::size_t __carry_offset = __sub_group_local_id;

                        std::uint8_t __iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);

                        std::uint8_t __i = 0;
                        for (; __i < __iters - 1; ++__i)
                        {
                            __sub_group_partials[__carry_offset] =
                                __reduce_op(__carry_last.__v, __sub_group_partials[__carry_offset]);
                            __carry_offset += __sub_group_size;
                        }
                        if (__i * __sub_group_size + __sub_group_local_id < __active_subgroups)
                        {
                            __sub_group_partials[__carry_offset] =
                                __reduce_op(__carry_last.__v, __sub_group_partials[__carry_offset]);
                            __carry_offset += __sub_group_size;
                        }
                        if (__sub_group_local_id == 0)
                            __sub_group_partials[__active_subgroups] = __carry_last.__v;
                        __carry_last.__destroy();
                    }
                }

                __dpl_sycl::__group_barrier(__ndi);

                // Get inter-work group and adjusted for intra-work group prefix
                bool __sub_group_carry_initialized = true;
                if (__block_num == 0)
                {
                    if (__sub_group_id > 0)
                    {
                        _InitValueType __value =
                            __sub_group_partials[std::min(__sub_group_id - 1, __active_subgroups - 1)];
                        oneapi::dpl::unseq_backend::__init_processing<_InitValueType>{}(__init, __value, __reduce_op);
                        __sub_group_carry.__setup(__value);
                    }
                    else if (__group_id > 0)
                    {
                        _InitValueType __value = __sub_group_partials[__active_subgroups];
                        oneapi::dpl::unseq_backend::__init_processing<_InitValueType>{}(__init, __value, __reduce_op);
                        __sub_group_carry.__setup(__value);
                    }
                    else // zeroth block, group and subgroup
                    {
                        if constexpr (__is_unique_pattern_v)
                        {
                            if (__sub_group_local_id == 0)
                            {
                                // For unique patterns, always copy the 0th element to the output
                                __write_op.__assign(__in_rng[0], __out_rng[0]);
                            }
                        }

                        if constexpr (std::is_same_v<_InitType,
                                                     oneapi::dpl::unseq_backend::__no_init_value<_InitValueType>>)
                        {
                            // This is the only case where we still don't have a carry in.  No init value, 0th block,
                            // group, and subgroup. This changes the final scan through elements below.
                            __sub_group_carry_initialized = false;
                        }
                        else
                        {
                            __sub_group_carry.__setup(__init.__value);
                        }
                    }
                }
                else
                {
                    if (__sub_group_id > 0)
                    {
                        _InitValueType __value =
                            __sub_group_partials[std::min(__sub_group_id - 1, __active_subgroups - 1)];
                        __sub_group_carry.__setup(__reduce_op(__get_block_carry_in(__block_num, __tmp_ptr), __value));
                    }
                    else if (__group_id > 0)
                    {
                        __sub_group_carry.__setup(__reduce_op(__get_block_carry_in(__block_num, __tmp_ptr),
                                                              __sub_group_partials[__active_subgroups]));
                    }
                    else
                    {
                        __sub_group_carry.__setup(__get_block_carry_in(__block_num, __tmp_ptr));
                    }
                }

                // step 5) apply global carries
                std::size_t __subgroup_start_id = __group_start_id + (__sub_group_id * __inputs_per_sub_group);
                std::size_t __start_id = __subgroup_start_id + __sub_group_local_id;

                if (__sub_group_carry_initialized)
                {
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/true,
                                                   /*__capture_output=*/true, __max_inputs_per_item>(
                        __sub_group, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op,
                        __sub_group_carry, __in_rng, __out_rng, __start_id, __n, __inputs_per_item, __subgroup_start_id,
                        __sub_group_id, __active_subgroups);
                }
                else // first group first block, no subgroup carry
                {
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/false,
                                                   /*__capture_output=*/true, __max_inputs_per_item>(
                        __sub_group, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op,
                        __sub_group_carry, __in_rng, __out_rng, __start_id, __n, __inputs_per_item, __subgroup_start_id,
                        __sub_group_id, __active_subgroups);
                }
                // If within the last active group and sub-group of the block, use the 0th work-item of the sub-group
                // to write out the last carry out for either the return value or the next block
                if (__sub_group_local_id == 0 && (__active_groups == __group_id + 1) &&
                    (__active_subgroups == __sub_group_id + 1))
                {
                    if (__block_num + 1 == __num_blocks)
                    {
                        if constexpr (__is_unique_pattern_v)
                        {
                            // unique patterns automatically copy the 0th element and scan starting at index 1
                            __res_ptr[0] = __sub_group_carry.__v + 1;
                        }
                        else
                        {
                            __res_ptr[0] = __sub_group_carry.__v;
                        }
                    }
                    else
                    {
                        // capture the last carry out for the next block
                        __set_block_carry_out(__block_num, __tmp_ptr, __sub_group_carry.__v);
                    }
                }
                __sub_group_carry.__destroy();
            });
        });
    }

    const std::uint32_t __max_block_size;
    const std::uint32_t __num_sub_groups_local;
    const std::uint32_t __num_sub_groups_global;
    const std::uint32_t __num_work_items;
    const std::size_t __num_blocks;
    const std::size_t __n;

    const _ReduceOp __reduce_op;
    const _GenScanInput __gen_scan_input;
    const _ScanInputTransform __scan_input_transform;
    const _WriteOp __write_op;
    _InitType __init;
};

// reduce_then_scan requires subgroup size of 32, and performs well only on devices with fast coordinated subgroup
// operations.  We do not want to run this scan on CPU targets, as they are not performant with this algorithm.
template <typename _ExecutionPolicy>
bool
__is_gpu_with_sg_32(const _ExecutionPolicy& __exec)
{
    const bool __dev_has_sg32 = oneapi::dpl::__internal::__supports_sub_group_size(__exec, 32);
    return (__exec.queue().get_device().is_gpu() && __dev_has_sg32);
}

// General scan-like algorithm helpers
// _GenReduceInput - a function which accepts the input range and index to generate the data needed by the main output
//                   used in the reduction operation (to calculate the global carries)
// _GenScanInput - a function which accepts the input range and index to generate the data needed by the final scan
//                 and write operations, for scan patterns
// _ScanInputTransform - a unary function applied to the output of `_GenScanInput` to extract the component used in the
//             scan, but not the part only required for the final write operation
// _ReduceOp - a binary function which is used in the reduction and scan operations
// _WriteOp - a function which accepts output range, index, and output of `_GenScanInput` applied to the input range
//            and performs the final write to output operation
template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _GenReduceInput, typename _ReduceOp,
          typename _GenScanInput, typename _ScanInputTransform, typename _WriteOp, typename _InitType,
          typename _Inclusive, typename _IsUniquePattern>
auto
__parallel_transform_reduce_then_scan(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                      _InRng&& __in_rng, _OutRng&& __out_rng, _GenReduceInput __gen_reduce_input,
                                      _ReduceOp __reduce_op, _GenScanInput __gen_scan_input,
                                      _ScanInputTransform __scan_input_transform, _WriteOp __write_op, _InitType __init,
                                      _Inclusive, _IsUniquePattern)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_reduce_kernel<_CustomName>>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_scan_kernel<_CustomName>>;
    using _ValueType = typename _InitType::__value_type;

    constexpr std::uint8_t __sub_group_size = 32;
    constexpr std::uint8_t __block_size_scale = std::max(std::size_t{1}, sizeof(double) / sizeof(_ValueType));
    // Empirically determined maximum. May be less for non-full blocks.
    constexpr std::uint16_t __max_inputs_per_item = 64 * __block_size_scale;
    constexpr bool __inclusive = _Inclusive::value;
    constexpr bool __is_unique_pattern_v = _IsUniquePattern::value;

    const std::uint32_t __max_work_group_size = oneapi::dpl::__internal::__max_work_group_size(__exec, 8192);
    // Round down to nearest multiple of the subgroup size
    const std::uint32_t __work_group_size = (__max_work_group_size / __sub_group_size) * __sub_group_size;

    // TODO: Investigate potentially basing this on some scale of the number of compute units. 128 work-groups has been
    // found to be reasonable number for most devices.
    constexpr std::uint32_t __num_work_groups = 128;
    const std::uint32_t __num_work_items = __num_work_groups * __work_group_size;
    const std::uint32_t __num_sub_groups_local = __work_group_size / __sub_group_size;
    const std::uint32_t __num_sub_groups_global = __num_sub_groups_local * __num_work_groups;
    const std::size_t __n = __in_rng.size();
    const std::uint32_t __max_inputs_per_block = __work_group_size * __max_inputs_per_item * __num_work_groups;
    std::size_t __inputs_remaining = __n;
    if constexpr (__is_unique_pattern_v)
    {
        // skip scan of zeroth element in unique patterns
        __inputs_remaining -= 1;
    }
    // reduce_then_scan kernel is not built to handle "empty" scans which includes `__n == 1` for unique patterns.
    // These trivial end cases should be handled at a higher level.
    assert(__inputs_remaining > 0);
    const std::uint32_t __max_inputs_per_subgroup = __max_inputs_per_block / __num_sub_groups_global;
    std::uint32_t __evenly_divided_remaining_inputs =
        std::max(std::size_t{__sub_group_size},
                 oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining) / __num_sub_groups_global);
    std::uint32_t __inputs_per_sub_group =
        __inputs_remaining >= __max_inputs_per_block ? __max_inputs_per_subgroup : __evenly_divided_remaining_inputs;
    std::uint32_t __inputs_per_item = __inputs_per_sub_group / __sub_group_size;
    const std::size_t __block_size = std::min(__inputs_remaining, std::size_t{__max_inputs_per_block});
    const std::size_t __num_blocks = __inputs_remaining / __block_size + (__inputs_remaining % __block_size != 0);

    // We need temporary storage for reductions of each sub-group (__num_sub_groups_global).
    // Additionally, we need two elements for the block carry-out to prevent a race condition
    // between reading and writing the block carry-out within a single kernel.
    __result_and_scratch_storage<_ExecutionPolicy, _ValueType> __result_and_scratch{__exec, 1,
                                                                                    __num_sub_groups_global + 2};

    // Reduce and scan step implementations
    using _ReduceSubmitter =
        __parallel_reduce_then_scan_reduce_submitter<__sub_group_size, __max_inputs_per_item, __inclusive,
                                                     __is_unique_pattern_v, _GenReduceInput, _ReduceOp, _InitType,
                                                     _ReduceKernel>;
    using _ScanSubmitter =
        __parallel_reduce_then_scan_scan_submitter<__sub_group_size, __max_inputs_per_item, __inclusive,
                                                   __is_unique_pattern_v, _ReduceOp, _GenScanInput, _ScanInputTransform,
                                                   _WriteOp, _InitType, _ScanKernel>;
    _ReduceSubmitter __reduce_submitter{__max_inputs_per_block,
                                        __num_sub_groups_local,
                                        __num_sub_groups_global,
                                        __num_work_items,
                                        __n,
                                        __gen_reduce_input,
                                        __reduce_op,
                                        __init};
    _ScanSubmitter __scan_submitter{__max_inputs_per_block,
                                    __num_sub_groups_local,
                                    __num_sub_groups_global,
                                    __num_work_items,
                                    __num_blocks,
                                    __n,
                                    __reduce_op,
                                    __gen_scan_input,
                                    __scan_input_transform,
                                    __write_op,
                                    __init};
    sycl::event __event;
    // Data is processed in 2-kernel blocks to allow contiguous input segment to persist in LLC between the first and second kernel for accelerators
    // with sufficiently large L2 / L3 caches.
    for (std::size_t __b = 0; __b < __num_blocks; ++__b)
    {
        std::uint32_t __workitems_in_block = oneapi::dpl::__internal::__dpl_ceiling_div(
            std::min(__inputs_remaining, std::size_t{__max_inputs_per_block}), __inputs_per_item);
        std::uint32_t __workitems_in_block_round_up_workgroup =
            oneapi::dpl::__internal::__dpl_ceiling_div(__workitems_in_block, __work_group_size) * __work_group_size;
        auto __global_range = sycl::range<1>(__workitems_in_block_round_up_workgroup);
        auto __local_range = sycl::range<1>(__work_group_size);
        auto __kernel_nd_range = sycl::nd_range<1>(__global_range, __local_range);
        // 1. Reduce step - Reduce assigned input per sub-group, compute and apply intra-wg carries, and write to global memory.
        __event = __reduce_submitter(__exec, __kernel_nd_range, __in_rng, __result_and_scratch, __event,
                                     __inputs_per_sub_group, __inputs_per_item, __b);
        // 2. Scan step - Compute intra-wg carries, determine sub-group carry-ins, and perform full input block scan.
        __event = __scan_submitter(__exec, __kernel_nd_range, __in_rng, __out_rng, __result_and_scratch, __event,
                                   __inputs_per_sub_group, __inputs_per_item, __b);
        __inputs_remaining -= std::min(__inputs_remaining, __block_size);
        // We only need to resize these parameters prior to the last block as it is the only non-full case.
        if (__b + 2 == __num_blocks)
        {
            __evenly_divided_remaining_inputs =
                std::max(std::size_t{__sub_group_size},
                         oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining) / __num_sub_groups_global);
            __inputs_per_sub_group = __inputs_remaining >= __max_inputs_per_block ? __max_inputs_per_subgroup
                                                                                  : __evenly_divided_remaining_inputs;
            __inputs_per_item = __inputs_per_sub_group / __sub_group_size;
        }
    }
    return __future(__event, __result_and_scratch);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H
